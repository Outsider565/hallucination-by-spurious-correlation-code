import os
import argparse
import multiprocessing as mp
import numpy as np
import tiktoken
from tqdm import tqdm
import json

def write_datafile(filename, toks):
    """ 
    Saves token data as a .bin file, for reading in C.
    - First comes a header with 256 int32s
    - The tokens follow, each as a uint16
    """
    assert len(toks) < 2**31, "token count too large" # ~2.1B tokens
    
    # construct the header
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520 # magic
    header[1] = 1 # version
    header[2] = len(toks) # number of tokens after the 256*4 bytes of header (each 2 bytes as uint16)
    
    if not isinstance(toks, np.ndarray) or not toks.dtype == np.uint16:
        maxtok = 2**16
        assert all(0 <= t < maxtok for t in toks), "token dictionary too large for uint16"
        toks_np = np.array(toks, dtype=np.uint16)
    else:
        toks_np = toks
        
    print(f"writing {len(toks):,} tokens to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())
    return len(toks)

def tokenize(s):
    """
    Tokenize a string using the tokenizer.
    """
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(s))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def save_shard(output_dir, input_file_name, all_tokens_np, token_count, shard_index, split, metrics):
    """Process and write a single data shard"""
    filename = os.path.join(output_dir, f"{input_file_name}_{split}_{shard_index:06d}.bin")
    tokens_to_write = all_tokens_np[:token_count]
    tokens_written = write_datafile(filename, tokens_to_write)
    
    # Update metrics
    metrics['files'].append({
        "filename": os.path.basename(filename),
        "tokens": tokens_written
    })
    if split == "val":
        metrics['val_tokens'] += tokens_written
    else:
        metrics['train_tokens'] += tokens_written
        
    return tokens_written

def convert_binary(input_file, output_dir, val_shard_size=10**7, align_length=512, n_proc=16):
    """
    Convert a text file to binary format for efficient loading during training.
    
    Args:
        input_file (str): Path to the input text file
        output_dir (str): Directory to save the binary files
        val_shard_size (int): Size of each validation shard in tokens
        align_length (int): Every align_length tokens will be in the same sequence
        n_proc (int): Number of processes to use for tokenization
    
    Returns:
        dict: Metrics about the conversion process
    """
    global enc, eot
    
    # Setup output directory and get input filename
    input_file_name = os.path.splitext(os.path.basename(input_file))[0]

    # Initialize metrics
    metrics = {
        "train_tokens": 0,
        "val_tokens": 0,
        "files": [],
    }

    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens['<|endoftext|>']
    eot = 50256

    # Process file with multiprocessing
    nprocs = max(1, os.cpu_count() - 2)
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        all_tokens_np = np.empty((10**9,), dtype=np.uint16)
        token_count = 0
        progress_bar = None
        sequence_token_count = 0 
        saved_val = False
        with open(input_file, 'r', encoding='utf-8') as f:
            for tokens in pool.imap(tokenize, f, chunksize=n_proc):
                # tokens: np.array, start with eos and end with \n(198)
                orig_token_length = len(tokens)
                if align_length > 0 and sequence_token_count + orig_token_length > align_length:
                    tokens = np.concatenate([np.full(align_length - sequence_token_count, eot, dtype=np.uint16), tokens])
                    sequence_token_count = 0
                sequence_token_count += orig_token_length
                if saved_val or (token_count + len(tokens) < val_shard_size):
                    # if can be fit in the current shard, add it
                    all_tokens_np[token_count:token_count+len(tokens)] = tokens
                    token_count += len(tokens)
                    if progress_bar is None:
                        progress_bar = tqdm(total=val_shard_size, unit="tokens", desc=f"Shard {shard_index}")
                    progress_bar.update(len(tokens))
                else:
                    # if can't be fit in the current shard, save it and start a new shard
                    # so not every shard is a full shard and of the same size
                    save_shard(output_dir, input_file_name, all_tokens_np, token_count, shard_index, "val", metrics)
                    shard_index += 1
                    progress_bar = None
                    all_tokens_np[0:len(tokens)] = tokens
                    token_count = len(tokens)
                    saved_val = True

        # Process final partial shard
        if token_count != 0:
            save_shard(output_dir, input_file_name, all_tokens_np, token_count, shard_index, "train", metrics)

    # Add align_length to metrics
    metrics['align_length'] = align_length
    
    # Save metrics
    metrics_file = os.path.join(output_dir, f"metadata.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Print summary
    print(f"\nMetrics saved to {metrics_file}")
    print(f"Total train tokens: {metrics['train_tokens']:,}")
    print(f"Total val tokens: {metrics['val_tokens']:,}")
    print(f"Total files: {len(metrics['files'])}")
    
    return metrics

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Text file to binary conversion")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input text file path")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output directory for binary files")
    parser.add_argument("-s", "--val_shard_size", type=int, default=10**7, help="Size of each val shard in tokens")
    parser.add_argument("-l", "--align_length", type=int, default=-1, help="Every align_length tokens will be in the same sequence in later training")
    parser.add_argument("--n_proc", type=int, default=16, help="Number of processes to use for tokenization")
    args = parser.parse_args()

    # Call the convert_binary function
    convert_binary(
        input_file=args.input,
        output_dir=args.output,
        val_shard_size=args.val_shard_size,
        align_length=args.align_length,
        n_proc=args.n_proc
    )

if __name__ == "__main__":
    main()
