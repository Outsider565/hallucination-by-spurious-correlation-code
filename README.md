# Code for “When Bias Pretends to Be Truth”

This folder contains the research code accompanying the paper on spurious‑correlation‑driven hallucinations in LLMs.  
The pipeline has three parts: (1) synthetic biographical data generation with controllable spurious correlation, (2) GPT‑2‑style pretraining/SFT and evaluation, and (3) hallucination‑detection benchmarking (synthetic + real LLMs), plus a toy‑model script for the theory section.


## Expected data files

Synthetic biographical data generation expects a `data/` directory (path configurable) containing:

```
data/
  first_names.txt
  middle_names.txt
  last_names.txt
  cities.txt
  universities.txt
  majors.txt
  companies.csv          # two columns: company_name, headquarters_city
```

The generator writes datasets into `hallucinate_small/` by default.

## Scripts overview

- `generate_bios_correlation_clean.py`  
  Generates synthetic profiles and QA datasets with controllable family‑name spurious correlation.
  Key flag: `--probability` (ρ) controls correlation strength.

- `convert_binary.py`  
  Tokenizes a `.txt` dataset into sharded `.bin` files plus `metadata.json` for fast loading.
  Produces files like `*_val_000000.bin` and `*_train_000000.bin`.

- `train_gpt2_clean.py`  
  Distributed training for GPT‑2‑like models on `.bin` shards produced above.

- `inference_SFT_clean.py`  
  Evaluates a trained model on QA datasets (accuracy on known entities, refusal on unknown entities)
  and optionally computes AUPRC for simple filters.

- `run_detection.py`  
  Computes hallucination‑detection scores (logit‑based, hidden‑state probes, attention heuristics)
  and trains/evaluates linear probes. Supports loading HF models from `--model_dir`.

- `halluc_detect_utils.py`, `detect.py`  
  Utilities for detection, including co‑occurrence counting on Wikipedia dumps (used as a proxy for spurious correlation).

- `synthetic.py`  
  Toy “shortcut vs noisy region” data model and membership‑probing experiments used in the theory section.

- `experiment_manager.py`  
  Optional orchestrator that runs data → train → eval from a JSON config.

## Quickstart: synthetic pipeline

1) **Generate synthetic datasets**

```bash
python generate_bios_correlation_clean.py \
  --data_dir data \
  --output_dir hallucinate_small \
  --probability 0.9 \
  --K 10000 \
  --ratio 0.12
```

This creates (among others):
- `hallucinate_small/pretrain_perturbed_mixed/` (binary shards + `metadata.json`)
- `hallucinate_small/SFT*.txt` and `hallucinate_small/refuse_*.txt`

2) **Pretrain / continue pretraining**

```bash
torchrun --nproc_per_node=NUM_GPUS train_gpt2_clean.py \
  --input_folder hallucinate_small/pretrain_perturbed_mixed \
  --model_size nano \
  --sequence_length 512 \
  --batch_size 512 \
  --device_batch_size 64 \
  --num_epochs 1 \
  --output_dir experiments \
  --run_name pretrain_rho0p9
```

Checkpoints and logs are written under `experiments/pretrain_rho0p9/`.

3) **Evaluate recall / refusal**

```bash
python inference_SFT_clean.py \
  --model_path experiments/pretrain_rho0p9/checkpoints/ckpt.pt \
  --input_path hallucinate_small/SFT_test.txt \
  --output_path experiments/pretrain_rho0p9/results/sft_test.json
```

Use `SFT_unknown_refused_test.txt` to measure refusal on unknown entities.

## Hallucination‑detection benchmarking

After you have a HF‑loadable model directory (or a converted checkpoint), run:

```bash
python run_detection.py \
  --model_dir PATH_TO_MODEL_DIR \
  --final_jsonl PATH_TO_FINAL.jsonl \
  --mt logit --mt hidden --mt attns \
  --first_n 500
```

See flags in `run_detection.py` to choose score types and caching behavior.

## Toy‑model / theory experiments

```bash
python synthetic.py --rho 0.05 --d 64 --d_shortcut 1 --n_train 600 --num_seed 5
```

Outputs are saved under `output/` and include AUROC curves for shortcut‑strength sweeps.

## Reproducibility

All scripts accept `--seed` (or equivalent) for deterministic runs when possible.  
Heavy experiments in the paper require multiple GPUs and long training; adjust model size and dataset scale as needed.

# hallucination-by-spurious-correlation-code
