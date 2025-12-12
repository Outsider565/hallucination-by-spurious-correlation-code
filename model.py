# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model
import torch
from torch import nn
import torch.nn.functional as F
import tiktoken
from dataclasses import dataclass
import numpy as np
from einops import einsum


EOS_TOKEN = 50256
NEWLINE_TOKEN = 198


class Rotary(torch.nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_()
        self.rotary = Rotary(self.head_dim)

    def create_packing_mask(self, x, eos_token=50256):
        B, T = x.size()
        device = x.device

        # Identify EOS tokens
        eos_mask = (x == eos_token).to(torch.int32)  # Shape: (B, T)

        # Compute cumulative sum to assign segment IDs
        # As our sequence are begin with eos token, we can use the cumulative sum to assign segment IDs
        segment_ids = torch.cumsum(eos_mask, dim=1)

        # Create mask where tokens can attend to others in the same segment
        mask = segment_ids.unsqueeze(1) == segment_ids.unsqueeze(2)  # Shape: (B, T, T)

        # Create causal mask to prevent attending to future tokens
        causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
        combined_mask = mask & causal_mask.unsqueeze(0)  # Shape: (B, T, T)
        return combined_mask



    def forward(self, x, idx=None):
        B, T, C = x.size()
        #TODO: could optimize this by sharing the mask across layers
        # Create attention mask if indices are provided
        if idx is not None:
            attention_mask = self.create_packing_mask(idx)
        else:
            attention_mask = None

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        
        cos, sin = self.rotary(q)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        # Apply attention with the bio mask
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=attention_mask.unsqueeze(1),
            is_causal=attention_mask is None  # Only use built-in causal mask if no bio mask
        )

        y = y.transpose(1, 2).contiguous().view_as(x)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_()  # zero init suggested by @Grad62304977

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(
            x
        ).square()  # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        # Get idx from attention layer if it exists
        idx = getattr(self.attn, 'idx', None)
        x = x + self.attn(F.rms_norm(x, (x.size(-1),)), idx=idx)
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x


# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
class GPTConfig:
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6
    n_embd: int = 768


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = (
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

    def forward(self, idx, targets=None, return_logits=True, full_logits=False, mask_q=False, random_align=False):
        # forward the GPT model itself
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        
        hs = []
        # Pass through transformer blocks with idx for attention masking
        for block in self.transformer.h:
            # Pass idx to the attention layer
            block.attn.idx = idx  # Store idx temporarily
            x = block(x)
            block.attn.idx = None  # Clean up
            hs.append(x)

        x = F.rms_norm(x, (x.size(-1),))
        if targets is not None:
            logits = self.lm_head(x)
            logits = logits.float()

            if mask_q:
                run_mask = torch.zeros(targets.size(0), dtype=torch.bool, device=targets.device)
                bg = torch.tensor([[EOS_TOKEN, 48, 25]] * targets.size(0), device=targets.device)
                ed = torch.tensor([[317, 25]] * targets.size(0), device=targets.device)
                mask = torch.zeros_like(targets, dtype=torch.bool, device=targets.device)
                K = targets.size(1)
                for i in range(K):
                    if i <= K-3:
                        run_mask |= (targets[:, i:i+3] == bg).all(dim=-1)
                    mask[:, i] |= run_mask
                    if i>=1:
                        run_mask &= ~((targets[:, i-1:i+1] == ed).all(dim=-1))

            if mask_q:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.contiguous().view(-1), ignore_index=-100, reduction='none'
                )
                loss = loss.view(targets.size(0), targets.size(1))
                loss = loss.masked_fill(mask, 0.0).sum(dim=1) / (mask.size(1) - mask.sum(dim=1))
                loss = loss.mean()
            else:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.contiguous().view(-1), ignore_index=-100, reduction='mean'
                )
            if random_align:
                # hs = torch.stack(hs, dim=1)
                # r_hs = torch.randn_like(hs)
                # loss += 10*torch.dot(hs.view(-1), r_hs.view(-1)) / hs.numel()

                # hs = torch.stack(hs, dim=1)
                # r_hs = torch.randperm(hs.size(1), device=hs.device)
                # r_hs = hs[:, r_hs, :]
                # loss += -0.1*torch.mean(torch.clip(torch.square(hs - r_hs), min=0, max=1))

                # hs=hs[1:-1]
                # r_hs = hs[::-1]
                # hs = torch.stack(hs, dim=1)
                # r_hs = torch.stack(r_hs, dim=1)
                # loss += torch.mean(torch.clip(torch.square(hs - r_hs), min=0, max=1))

                # hs = torch.stack(hs, dim=1)
                # hs.transpose(1,2)
                # hs = hs.reshape(hs.size(0), hs.size(1), -1)
                # hs = F.normalize(hs, dim=-1)
                # hs = einsum(hs, hs, 'a b d, e b d -> b a e')
                # # loss += torch.mean(torch.square(hs - torch.eye(hs.size(-1), device=hs.device)))
                # loss += torch.mean(torch.exp(5*torch.abs(hs - torch.eye(hs.size(-1), device=hs.device))))-1

                hs = torch.stack(hs, dim=1)
                hs = F.normalize(hs, dim=-1)
                hs = einsum(hs, hs, 'a b c d, e b c d -> b c a e')
                # loss += torch.mean(torch.square(hs - torch.eye(hs.size(-1), device=hs.device)))
                loss += torch.mean(torch.exp(5*torch.abs(hs - torch.eye(hs.size(-1), device=hs.device))))-1
        elif full_logits:
            logits = self.lm_head(x)
            logits = logits.float()
            loss = None
        else:
            logits = self.lm_head(x[:, [-1], :])
            logits = logits.float()
            loss = None

        if not return_logits:
            logits = None

        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, stop_token=50256):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx)
            if temperature == 0:
                idx_next = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            else:
                # pluck the logits at the final step and scale by desired temperature
                logits = logits[:, -1, :] / temperature
                # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                # apply softmax to convert logits to (normalized) probabilities
                probs = F.softmax(logits, dim=-1)
                # sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)
            if idx_next == stop_token:
                break
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
    @classmethod
    def from_pretrained(cls, path, device='cpu'):
        """
        Load a saved model from a file.

        Args:
            path (str): The path to the saved model file.
            device: The device to load the model onto (e.g., 'cpu' or 'cuda').

        Returns:
            GPT: The loaded model.

        Example:
            >>> model_path = "path/to/saved_model.pt"
            >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            >>> loaded_model = GPT.from_pretrained(model_path, device)
        """
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model_state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        
        if 'config' not in checkpoint:
            # Try to infer model parameters from the state dict
            config = {}
            for key in model_state_dict.keys():
                if 'wte.weight' in key:
                    config['vocab_size'] = model_state_dict[key].shape[0]
                elif 'wpe.weight' in key:
                    config['block_size'] = model_state_dict[key].shape[0]
                elif '.attn.' in key and '.weight' in key:
                    if 'attn.c_attn.weight' in key:
                        config['n_embd'] = model_state_dict[key].shape[0] // 3
                    elif 'attn.c_proj.weight' in key:
                        config['n_embd'] = model_state_dict[key].shape[1]
                elif '.mlp.' in key and '.weight' in key:
                    if 'mlp.c_fc.weight' in key:
                        config['n_embd'] = model_state_dict[key].shape[1]
            
            # Count the number of transformer blocks
            # breakpoint()
            config['n_layer'] = len([k for k in model_state_dict.keys() if 'h.' in k and '.attn.c_q.weight' in k])
            
            # Infer n_head if possible
            if 'n_embd' in config:
                for key in model_state_dict.keys():
                    if '.attn.bias' in key:
                        n_head = model_state_dict[key].shape[0]
                        if n_head * (config['n_embd'] // n_head) == config['n_embd']:
                            config['n_head'] = n_head
                            break
            print("Warning, this code might produce wrong n_embd")
        else:
            config = checkpoint['config']
        
        # Create a GPTConfig object from the inferred configuration
        gpt_config = GPTConfig(**config)
        print(gpt_config)
        
        model = cls(gpt_config)
        model.load_state_dict(model_state_dict, strict=True)
        model = model.to(device)
        return model
    
    @property
    def config(self):
        return {
            'vocab_size': self.transformer.wte.weight.shape[0],
            'n_embd': self.transformer.wte.weight.shape[1],
            'n_layer': len(self.transformer.h),
            'n_head': self.transformer.h[0].attn.n_head if self.transformer.h else 0
        }
        
    
def tokenize(s, enc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    eot = enc._special_tokens['<|endoftext|>'] # end of text token
    tokens = [eot] # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(s))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def inference(model, input_text: str, tokenizer=None, max_new_tokens=100, **kwargs):
    """
    Perform inference using a GPT model.

    Args:
        model: Either a path to a saved model file or a GPT model instance.
        input_text (str): The input text to start the generation from.
        tokenizer: Either a tokenizer instance or None. If None, load from tiktoken.
        max_new_tokens (int): Maximum number of new tokens to generate.
        **kwargs: Additional keyword arguments for text generation.

    Returns:
        str: The generated text.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model if a path is provided
    if isinstance(model, str):
        model = GPT.from_pretrained(model, device)
    elif isinstance(model, GPT):
        model = model.to(device)
    else:
        raise ValueError("model must be either a path to a saved model or a GPT instance")

    model.eval()

    # Use provided tokenizer or load from tiktoken
    if tokenizer is None:
        tokenizer = tiktoken.get_encoding("gpt2")

    # Encode the input text using the tokenize function
    input_ids = tokenize(input_text, tokenizer)
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    # Generate text
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, **kwargs)

    # Decode the generated ids back to text
    output_text = tokenizer.decode(output_ids[0].tolist())
    # exclude the input text from the output
    output_text = output_text[len(input_text+'<|endoftext|>'):]

    return output_text

def eval_loss(model, texts, tokenizer=None):
    """
    Evaluate loss for given text(s) using a GPT model.
    
    Args:
        model: A GPT model instance
        texts: Either a single string or list of strings to evaluate
        tokenizer: Either a tokenizer instance or None. If None, load from tiktoken.
        
    Returns:
        tuple: (loss value as float, list of log probabilities for each token)
        If texts is a list, returns lists of losses and log probabilities
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use provided tokenizer or load from tiktoken
    if tokenizer is None:
        tokenizer = tiktoken.get_encoding("gpt2")
    
    # Handle single string case
    if isinstance(texts, str):
        texts = [texts]
        
    # Encode all texts and get max length
    encoded_texts = [tokenize(text, tokenizer) for text in texts]
    max_len = max(len(tokens) for tokens in encoded_texts)
    
    # Pad with -100 token
    padded_texts = [np.concatenate([tokens, np.array([-100] * (max_len - len(tokens)))]) for tokens in encoded_texts]
    
    # Convert to tensor
    input_ids = torch.tensor(np.array(padded_texts), dtype=torch.long, device=device)
    
    target_ids = input_ids[:, 1:].clone()
    input_ids = input_ids[:, :-1]
    # replace -100 with EOT token in input_ids
    input_ids[input_ids == -100] = EOS_TOKEN
    # Get loss and logits
    model.eval()
    with torch.no_grad():
        loss = []
        logits, _ = model(input_ids, target_ids)
        
        # Calculate log probabilities for each sequence
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        all_token_log_probs = []
        
        for i in range(len(texts)):
            seq_len = len(encoded_texts[i])
            token_log_probs = log_probs[i, range(seq_len-1), target_ids[i][:seq_len-1]]
            all_token_log_probs.append(token_log_probs.tolist())
            loss.append(sum(token_log_probs).item()/(seq_len-1))
    
    # Return single result for string input, list of results for list input
    if len(texts) == 1:
        return loss[0], all_token_log_probs[0]
    else:
        return loss, all_token_log_probs




if __name__ == "__main__":
    PATH = "logs/345ab7f3-1ba5-4518-b02d-ff1a7f8c3b64/state_step005100.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT.from_pretrained(PATH, device)
    # Initialize the tokenizer
    enc = tiktoken.get_encoding("gpt2")
    print(inference(model, "Jade Emiliano Long entered life on", tokenizer=enc))
