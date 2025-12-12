import os
import random
from typing import Callable, Union
from tqdm import tqdm
from fire import Fire

random.seed(42)

def dynamic_mix_data(mixed_in_file: str, mixed_from_file: str, every_n_lines: Union[int, Callable[[float], int]], repeat_mixed_in_file: int = 1):
    with open(mixed_in_file, 'r') as f:
        original_mixed_in_lines = f.readlines()
    with open(mixed_from_file, 'r') as f:
        mixed_from_lines = f.readlines()
    mixed_in_lines = []
    for _ in range(repeat_mixed_in_file):
        # shuffle the original mixed_in_lines every time
        mixed_in_lines.extend(original_mixed_in_lines)
        random.shuffle(original_mixed_in_lines)
    if isinstance(every_n_lines, int):
        every_n_lines = lambda x: every_n_lines
    result_lines = []
    insert_count = every_n_lines(0)
    for i, line in enumerate(tqdm(mixed_in_lines)):
        result_lines.append(line)
        insert_count -= 1
        if insert_count == 0:
            result_lines.append(random.choice(mixed_from_lines))
            insert_count = every_n_lines(i / len(mixed_in_lines))
    return result_lines
    
def make_linear_every_n_lines(start_n: int, end_n: int, start_ratio: float, end_ratio: float):
    def linear_every_n_lines(ratio: float):
        # Linear interpolation between start_n and end_n based on ratio
        ratio_range = end_ratio - start_ratio
        if ratio <= start_ratio:
            return start_n
        elif ratio >= end_ratio:
            return -1
        normalized_ratio = (ratio - start_ratio) / ratio_range
        return int(start_n + (end_n - start_n) * normalized_ratio)
    return linear_every_n_lines

def simple_mix_data(file_a: str, file_b: str, file_a_ratio: float, file_b_ratio: float, save_path: str):
    # file_a_ratio and file_b_ratio could > 1
    with open(file_a, 'r') as f:
        file_a_lines = f.readlines()
    with open(file_b, 'r') as f:
        file_b_lines = f.readlines()
    result_lines = []
    
    num_a_lines = int(len(file_a_lines) * file_a_ratio)
    num_b_lines = int(len(file_b_lines) * file_b_ratio)
    print(num_a_lines, num_b_lines)
    
    sampled_a_lines = (file_a_lines * (num_a_lines // len(file_a_lines))) + random.sample(file_a_lines, num_a_lines % len(file_a_lines))
    sampled_b_lines = (file_b_lines * (num_b_lines // len(file_b_lines))) + random.sample(file_b_lines, num_b_lines % len(file_b_lines))
    total_lines = num_a_lines + num_b_lines
    print(f'sampled {len(sampled_a_lines)}({len(sampled_a_lines) / total_lines:.2%}) lines from {file_a}')
    print(f'sampled {len(sampled_b_lines)}({len(sampled_b_lines) / total_lines:.2%}) lines from {file_b}')
    
    result_lines = sampled_a_lines + sampled_b_lines
    random.shuffle(result_lines)
    
    with open(save_path, 'w') as f:
        f.writelines(result_lines)

    
def mix_data_balanced_1_to_1(file_a: str, file_b: str, save_path: str):
    """
    Mixes two files with a 1:1 ratio, based on the length of the shorter file.
    """
    print(f"Mixing {os.path.basename(file_a)} and {os.path.basename(file_b)} with 1:1 balanced samples.")
    
    with open(file_a, 'r') as f:
        lines_a = f.readlines()
    with open(file_b, 'r') as f:
        lines_b = f.readlines()
        
    min_len = min(len(lines_a), len(lines_b))
    
    if min_len == 0:
        print(f"Warning: One of the files is empty. Cannot perform balanced mix. Output will be empty.")
        with open(save_path, 'w') as f:
            f.write("")
        return

    ratio_a = min_len / len(lines_a) if len(lines_a) > 0 else 0
    ratio_b = min_len / len(lines_b) if len(lines_b) > 0 else 0
    
    print(f"Shorter file has {min_len} lines. Mixing {min_len} lines from each file.")
    simple_mix_data(file_a, file_b, ratio_a, ratio_b, save_path)


def test_mix():
    mixed_from_file = 'bioS_single/pretrain.txt'
    mixed_in_file = 'bioS_single/SFT.txt'
    result_lines = dynamic_mix_data(mixed_in_file, mixed_from_file, make_linear_every_n_lines(1, 10, 0, 0.8), repeat_mixed_in_file=10)
    with open('bioS_single/SFT_mix_pretrain_10x.txt', 'w') as f:
        f.writelines(result_lines)

if __name__ == '__main__':
    Fire(simple_mix_data)