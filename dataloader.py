import glob
import numpy as np
import torch
import tiktoken

from fire import Fire
# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader
IGNORE_INDEX = -100

def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print(
            "---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README"
        )
        print(
            "---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try"
        )
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2]  # number of tokens (claimed)
    return ntok  # for now just return the number of tokens


def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2]  # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens


class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T
        self.eos_token = 50256  # GPT2's EOS token

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total

        # kick things off
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self):
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B, T = self.B, self.T
        if self.current_position + B * T <= len(self.tokens):
            x = torch.tensor(self.tokens[self.current_position:self.current_position + B * T], dtype=torch.long) # shape: (B * T)
            y = torch.tensor(self.tokens[self.current_position + 1:self.current_position + B * T + 1], dtype=torch.long) # shape: (B * T) or (B * T - 1) if last batch
            if self.current_position + B * T == len(self.tokens):
                print("warning: the last batch's y is not complete")
                self.advance()
                # one more token for y
                y = torch.cat([y, torch.tensor(self.tokens[self.current_position:self.current_position+1], dtype=torch.long)])
        else:
            temp_x = torch.tensor(self.tokens[self.current_position:], dtype=torch.long)
            temp_y = torch.tensor(self.tokens[self.current_position + 1:], dtype=torch.long)
            self.advance()
            x = torch.cat([temp_x, torch.tensor(self.tokens[self.current_position:self.current_position + B * T - len(temp_x)], dtype=torch.long)])
            y = torch.cat([temp_y, torch.tensor(self.tokens[self.current_position:self.current_position + B * T - len(temp_y)], dtype=torch.long)])
            assert len(x) == len(y) == B * T, f"x and y should be of the same length {B*T}, but got {len(x)} and {len(y)}"
        x = x.reshape(B, T)
        y = y.reshape(B, T)
        # mask the consecutive eos tokens in y
        consecutive_eos_mask = (x == self.eos_token) & (y  == self.eos_token)
        y[consecutive_eos_mask] = IGNORE_INDEX
        self.current_position += B * T * self.num_processes # move to next process chunk of (B * T * N)
        return x.cuda(), y.cuda()

def print_file(filename):
    dataloader = DistributedDataLoader(filename, 8, 512, 1, 2)
    enc = tiktoken.get_encoding("gpt2")
    for i in range(3):
        x, y = dataloader.next_batch()
        print(x.shape)
        print(y.shape)
        print("-----------------------")
        print(enc.decode(x[0].tolist()))
        print(y[0].tolist())
        print(enc.decode(x[1].tolist()))
        print("-----------------------")

if __name__ == "__main__":
    Fire(print_file)