import argparse

from six.moves import cPickle as pkl
from tqdm import tqdm
import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from itertools import islice
from collections import defaultdict

import json
import re

import hashlib
import time
from pathlib import Path

from fastchat.model import get_conversation_template
from halluc_detect_utils import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import numpy as np
import os
import pickle

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, required=True, help="Path to saved model directory")
parser.add_argument("--correct_samples_path", type=str, help="Path to correct samples file")
parser.add_argument("--hallucination_samples_path", type=str, help="Path to hallucination samples file")
parser.add_argument("--final_jsonl", type=str, help="Path to final.jsonl containing question/label and prediction fields")
parser.add_argument("--first_n", type=int, default=500, help="First N samples to consider from each file")
parser.add_argument(
    "--use_toklens", action="store_true", help="remove prompt prefix before computing hidden and eigen scores"
)
parser.add_argument(
    "--mt", choices=["logit", "hidden", "attns"], action="append", help="choose method types for detection scores"
)
parser.add_argument("--system_prompt", type=str, default="You are a helpful assistant.", help="System prompt for generation")
parser.add_argument("--reasoning", type=str, default="low", choices=["low", "medium", "high"], help="Harmony reasoning effort for prompt building")
parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"], help="Model dtype")
parser.add_argument("--tag", type=str, default="gpt-oss", help="Tag for saved feature filenames")
parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()
set_seed(args.seed)

def parse_qa_line(line):
    if "A:" not in line:
        return None, None
    q_part, a_part = line.split("A:", 1)
    prompt = q_part.replace("Q:", "").strip()
    response = a_part.strip()
    return prompt, response

def build_custom_prompt(system_prompt, user_prompt, assistant_response=None):
    parts = []
    if system_prompt:
        parts.append(f"<|system|>\n{system_prompt}")
    if user_prompt:
        parts.append(f"<|user|>\n{user_prompt}")
    if assistant_response is not None:
        parts.append(f"<|assistant|>\n{assistant_response}")
    else:
        parts.append("<|assistant|>\n")
    return "\n".join(parts)

def compute_linear_probe_metrics(X, y, name="Linear Probe"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = LogisticRegression(max_iter=50000, solver="saga")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)[:, 1]

    auc_val = roc_auc_score(y_test, y_score)
    acc_val = accuracy_score(y_test, y_pred)

    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    mask = fpr <= 0.05
    tpr_at_5 = tpr[mask].max() if np.any(mask) else 0.0

    print(f"[{name}] AUROC: {auc_val*100:.2f}%, "
          f"Acc: {acc_val*100:.2f}%, "
          f"TPR@5%FPR: {tpr_at_5*100:.2f}%")

PREFERRED_CACHE_ROOT = Path("/dev/shm/run_detection_cache")
FALLBACK_CACHE_ROOT = Path("features_cache")


def ensure_cache_root() -> Path:
    """Return a writable cache directory, preferring /dev/shm."""
    for candidate in (PREFERRED_CACHE_ROOT, FALLBACK_CACHE_ROOT):
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except OSError:
            continue
    # As a last resort, fall back to current working directory cache folder
    fallback = Path.cwd() / "features_cache"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def describe_path(path: str) -> dict:
    """Capture basic file metadata for cache keys."""
    if not path:
        return {}
    abs_path = os.path.abspath(path)
    info = {"path": abs_path}
    try:
        stat_result = os.stat(abs_path)
    except FileNotFoundError:
        info.update({"exists": False})
    else:
        info.update({
            "exists": True,
            "size": stat_result.st_size,
            "mtime": stat_result.st_mtime,
        })
    return info


def gather_dataset_metadata(args) -> dict:
    if args.final_jsonl:
        return {"final_jsonl": describe_path(args.final_jsonl)}
    meta = {}
    if args.correct_samples_path:
        meta["correct_samples_path"] = describe_path(args.correct_samples_path)
    if args.hallucination_samples_path:
        meta["hallucination_samples_path"] = describe_path(args.hallucination_samples_path)
    return meta


def build_cache_payload(args, dataset_meta: dict, sample_count: int, mt_list) -> dict:
    return {
        "model_dir": os.path.abspath(args.model_dir),
        "dtype": args.dtype,
        "tag": args.tag,
        "reasoning": args.reasoning,
        "use_toklens": bool(args.use_toklens),
        "mt": sorted(mt_list),
        "dataset": dataset_meta,
        "first_n": args.first_n,
        "sample_count": sample_count,
    }


def cache_key_from_payload(payload: dict) -> str:
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_cached_artifacts(cache_root: Path, cache_key: str, expected_payload: dict):
    cache_dir = cache_root / cache_key
    metadata_path = cache_dir / "metadata.json"
    artifacts_path = cache_dir / "artifacts.pkl"
    if not metadata_path.exists() or not artifacts_path.exists():
        return None
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception:
        return None
    if metadata.get("payload") != expected_payload:
        return None
    try:
        with open(artifacts_path, "rb") as f:
            artifacts = pickle.load(f)
    except Exception:
        return None
    return metadata, artifacts


def save_cached_artifacts(cache_root: Path, cache_key: str, payload: dict, artifacts: dict) -> Path:
    cache_dir = cache_root / cache_key
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_dir / "metadata.json"
    artifacts_path = cache_dir / "artifacts.pkl"
    metadata = {"payload": payload, "saved_at": time.time()}
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f)
    with open(artifacts_path, "wb") as f:
        pickle.dump(artifacts, f)
    return cache_dir

if __name__ == "__main__":
    print(args.model_dir)

    mt_list = args.mt

    # dtype selection for model loading
    if args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, trust_remote_code=True, use_fast=True, padding_side='left'
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True
    ).eval()
    if getattr(model.config, 'pad_token_id', None) is None:
        model.config.pad_token_id = tokenizer.eos_token_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = device

    # Build data from final.jsonl if provided, else fall back to legacy text files
    records = []
    labels = []
    if args.final_jsonl and os.path.isfile(args.final_jsonl):
        with open(args.final_jsonl, "r", encoding="utf-8") as f:
            for line in islice(f, args.first_n):
                line = line.strip()
                if not line:
                    continue
                try:
                    ex = json.loads(line)
                except Exception:
                    continue
                q = ex.get("question") or ex.get("problem") or ""
                pred = (
                    ex.get("prediction")
                    or ex.get("largest_cluster_prediction")
                    or ex.get("model_answer")
                    or ex.get("answer")
                    or ""
                )
                gold = ex.get("label") or ex.get("gold") or ex.get("reference") or ""
                if "em" in ex:
                    lab = int(ex["em"]) if isinstance(ex["em"], (int, bool)) else int(bool(ex["em"]))
                elif "is_largest_cluster_correct" in ex:
                    lab = int(bool(ex["is_largest_cluster_correct"]))
                else:
                    lab = int(re.sub(r"\s+", " ", str(pred).strip()).lower() == re.sub(r"\s+", " ", str(gold).strip()).lower()) if gold else 0
                records.append({"prompt": q, "response": pred})
                labels.append(lab)
    else:
        all_qas = []
        if args.correct_samples_path:
            with open(args.correct_samples_path, "r", encoding="utf-8") as f:
                correct_qas = list(islice(f, args.first_n))
            all_qas.extend(correct_qas)
            labels.extend([1] * len(correct_qas))

        if args.hallucination_samples_path:
            with open(args.hallucination_samples_path, "r", encoding="utf-8") as f:
                halluc_qas = list(islice(f, args.first_n))
            all_qas.extend(halluc_qas)
            labels.extend([0] * len(halluc_qas))
    
    scores = []
    indiv_scores = {}
    for mt in mt_list:
        indiv_scores[mt] = defaultdict(def_dict_value)
    
    tok_lens = []
    # get scores for sample data

    num_layers = model.config.num_hidden_layers
    features_dict = {l: {
        "in_avg": [], "in_last": [],
        "out_avg": [], "out_last": [],
    } for l in range(num_layers+1)}

    y_labels = []
    y_labels = []


    if args.final_jsonl and os.path.isfile(args.final_jsonl):
        iterator = enumerate(records)
        total_len = len(records)
    else:
        iterator = enumerate(all_qas)
        total_len = len(all_qas)

    for i, item in tqdm(iterator, total=total_len):
        if args.final_jsonl and os.path.isfile(args.final_jsonl):
            prompt = item["prompt"]
            response = item["response"]
        else:
            prompt, response = parse_qa_line(item)
            if response is None:
                continue

        # Harmony-style prompts using GPT-OSS chat template
        dev_msg = {"role": "developer", "content": "Only output the final answer. "}
        usr_msg = {"role": "user", "content": prompt}
        messages_user = [dev_msg, usr_msg]
        messages_full = [dev_msg, usr_msg, {"role": "assistant", "content": response}]

        user_prompt = tokenizer.apply_chat_template(
            messages_user,
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort=args.reasoning
        )
        full_prompt = tokenizer.apply_chat_template(
            messages_full,
            tokenize=False,
            add_generation_prompt=False,
            reasoning_effort=args.reasoning
        )

        tok_in_u = tokenizer(user_prompt, return_tensors="pt", add_special_tokens=False).input_ids
        tok_in = tokenizer(full_prompt, return_tensors="pt", add_special_tokens=False).input_ids
        tok_lens.append([tok_in_u.shape[1], tok_in.shape[1]])

        logit, hidden_act, attn = get_model_vals(model, tok_in.to(model_device))
        logit = logit[0].cpu()
        hidden_act = [x[0].to(torch.float32).detach().cpu() for x in hidden_act]
        attn = [x[0].to(torch.float32).detach().cpu() for x in attn]
        tok_in = tok_in.cpu()

        tok_len = [tok_in_u.shape[1], tok_in.shape[1]]

        for l in range(len(hidden_act)):
            h = hidden_act[l]

            #print(l,h.shape)

            h_in = h[:tok_len[0]]
            features_dict[l]["in_avg"].append(h_in.mean(axis=0))
            features_dict[l]["in_last"].append(h_in[-1])

            if tok_len[1] > tok_len[0]:
                h_out = h[tok_len[0]:tok_len[1]]
                features_dict[l]["out_avg"].append(h_out.mean(axis=0))
                features_dict[l]["out_last"].append(h_out[-1])
            else:
                d = h_in.shape[1]
                features_dict[l]["out_avg"].append(np.zeros(d))
                features_dict[l]["out_last"].append(np.zeros(d))

        y_labels.append(labels[i])

        compute_scores(
            [logit],
            [hidden_act],
            [attn],
            scores,
            indiv_scores,
            mt_list,
            [tok_in],
            [tok_len],
            use_toklens=args.use_toklens,
        )
    y_labels = np.array(y_labels)

    os.makedirs("features", exist_ok=True)

    # Save the entire features dictionary
    with open(f"features/merged_features_{args.tag}.pkl", 'wb') as f:
        pickle.dump(features_dict, f)

    # Save the y_labels array
    with open(f"features/y_labels_{args.tag}.pkl", 'wb') as f:
        pickle.dump(y_labels, f)

    # for l, feats in features_dict.items():
    #     for k, X_list in feats.items():
    #         if len(X_list) == 0:
    #             continue
    #         X = np.array(X_list)
    #         print(f"Layer {l} - Feature {k} - Shape: {X.shape}")
    #         compute_linear_probe_metrics(X, y_labels, name=f"Linear Probing Layer {l+1} [{k}]")
    if 'logit' in mt_list:
        ly_scores = -np.array(indiv_scores['logit']["perplexity"])
        arc, acc, low, fpr, tpr, thresh_ind, thresh = get_roc_auc_scores(*get_balanced_scores(ly_scores,labels))
        print(f"PPL & {arc*100:.2f} & {acc*100:.2f} & {low*100:.2f} \\")

        ly_scores = np.array(indiv_scores['logit']["window_entropy"])
        arc, acc, low, fpr, tpr, thresh_ind, thresh = get_roc_auc_scores(*get_balanced_scores(ly_scores,labels))
        print(f"Window Entropy & {arc*100:.2f} & {acc*100:.2f} & {low*100:.2f} \\")

        ly_scores = np.array(indiv_scores['logit']["logit_entropy"])
        arc, acc, low, fpr, tpr, thresh_ind, thresh = get_roc_auc_scores(*get_balanced_scores(ly_scores,labels))
        print(f"Logit Entropy & {arc*100:.2f} & {acc*100:.2f} & {low*100:.2f} \\")

    if 'attns' in mt_list and len(indiv_scores.get('attns', {})) > 0:
        num_layers_report = len(indiv_scores['attns'].keys())
        arc_list, acc_list, low_list = [], [], []

        samp_preds = []
        thresh_vals = []

        for layer_num in range(1, num_layers_report + 1):
            scores = -np.array(indiv_scores['attns']["Attn"+str(layer_num)])
            bal_sc, bal_labels = get_balanced_scores(scores,labels)
            arc, acc, low, fpr, tpr, thresh_ind, thresh = get_roc_auc_scores(bal_sc,bal_labels)
            thresh_val, pred_list = get_thresh_val(thresh, acc, bal_sc)
            samp_preds.append(pred_list)
            thresh_vals.append(thresh_val)
            print(f"Layer:{layer_num+1} - AUROC:{arc:.4f}, Acc:{acc:.4f}, TPR@5%FPR:{low:.4f}")
            arc_list.append(arc)
            acc_list.append(acc)
            low_list.append(low)
