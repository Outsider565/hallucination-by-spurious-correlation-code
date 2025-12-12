import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import time

import torch
import torch.multiprocessing as mp
import pandas as pd
import numpy as np
from tqdm import tqdm
import tiktoken

from model import GPT, inference
from utils import get_info


@dataclass
class InferenceConfig:
    """Configuration for model inference and evaluation."""
    model_path: str
    input_path: str
    output_path: str
    
    # Inference parameters
    first_n: int = -1  # Number of samples to evaluate (-1 for all)
    processes_per_gpu: int = 4
    max_new_tokens: int = 200
    temperature: float = 0.0
    
    # Data parameters
    individuals_file: str = "hallucinate_small/individuals.json"
    
    # Gradient-based filtering (experimental)
    use_gradient_filter: bool = False
    gradient_threshold: float = 300.0
    
    # Loss-based filtering (experimental)  
    use_loss_filter: bool = False
    loss_threshold: float = 3.0
    
    # AUPRC calculation
    calculate_auprc: bool = False
    auprc_filter_types: List[str] = None # List of filters to apply, e.g., ["gradient", "loss"]
    
    # Logging
    log_level: str = "INFO"
    save_detailed_results: bool = True


class GPT2Evaluator:
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.setup_logging()
        self.setup_tokenizer()
        self.load_known_persons()
        
    def setup_logging(self):
        """Setup logging configuration."""
        output_dir = Path(self.config.output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = output_dir / "evaluation.log"
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_tokenizer(self):
        """Initialize tokenizer."""
        os.environ["TIKTOKEN_CACHE_DIR"] = "./tiktoken_cache"
        self.enc = tiktoken.get_encoding("gpt2")
        self.eot = 50256
        
    def load_known_persons(self):
        """Load known persons from individuals file."""
        self.known_persons = set()
        try:
            with open(self.config.individuals_file, 'r', encoding='utf-8') as f:
                individuals = json.load(f)
                for person in individuals:
                    self.known_persons.add(person['full_name'])
            self.logger.info(f"Loaded {len(self.known_persons)} known persons")
        except FileNotFoundError:
            self.logger.warning(f"Individuals file not found: {self.config.individuals_file}")
            
    def tokenize(self, text: str) -> np.ndarray:
        """Tokenize a string."""
        tokens = [self.eot]
        tokens.extend(self.enc.encode_ordinary(text))
        tokens_np = np.array(tokens)
        assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "Token dictionary too large for uint16"
        return tokens_np.astype(np.uint16)
        
    def get_gradient(self, model: GPT, text: str) -> torch.Tensor:
        """Calculate gradient for a given text."""
        model.train()
        model.zero_grad()
        
        device = next(model.parameters()).device
        inputs = torch.from_numpy(self.tokenize(text)).to(device).unsqueeze(0).long()
        target = inputs.roll(-1, dims=1)
        loss = model(inputs, targets=target)[1]
        loss.backward(retain_graph=True)

        grad_list = []
        for param in model.parameters():
            if param.grad is not None:
                grad_list.append(param.grad.view(-1))
                
        flat_grads = torch.cat(grad_list)
        model.zero_grad()
        model.eval()
        
        return flat_grads
        
    def get_loss(self, model: GPT, text: str) -> float:
        """Calculate loss for a given text."""
        model.eval()
        model.zero_grad()
        
        device = next(model.parameters()).device
        inputs = torch.from_numpy(self.tokenize(text)).to(device).unsqueeze(0).long()
        target = inputs.roll(-1, dims=1)
        loss = model(inputs, targets=target)[1]
        
        return loss.item()
        
    def apply_gradient_filter(self, model: GPT, question: str, answer: str) -> str:
        """Apply gradient-based filtering to detect potential hallucinations."""
        if "I don't know" in answer:
            return answer
            
        qa_text = f"Q: {question} A: {answer}"
        qa_idk_text = f"Q: {question} A: I don't know."
        
        g1 = self.get_gradient(model, qa_text)
        g2 = self.get_gradient(model, qa_idk_text)
        
        c1 = torch.dot(g1, g1).item()
        c3 = torch.dot(g2, g2).item()
        
        if c1 > c3 and c1 >= self.config.gradient_threshold:
            self.logger.debug(f"Gradient filter triggered: c1={c1:.2f}, c3={c3:.2f}")
            return "I don't know."
            
        return answer
        
    def apply_loss_filter(self, model: GPT, question: str, answer: str) -> str:
        """Apply loss-based filtering to detect potential hallucinations."""
        if "I don't know" in answer:
            return answer
            
        qa_text = f"Q: {question} A: {answer}"
        qa_idk_text = f"Q: {question} A: I don't know"
        
        loss_answer = self.get_loss(model, qa_text)
        loss_idk = self.get_loss(model, qa_idk_text)
        
        if loss_answer > loss_idk and loss_idk >= self.config.loss_threshold:
            self.logger.debug(f"Loss filter triggered: loss_answer={loss_answer:.2f}, loss_idk={loss_idk:.2f}")
            return "I don't know"
            
        return answer
        
    def process_chunk(self, process_id: int, gpu_id: int, model: GPT, 
                     qas: List[str], start_idx: int, end_idx: int, 
                     return_dict: Dict, progress_queue: mp.Queue):
        """Process a chunk of QA pairs."""
        torch.cuda.set_device(gpu_id)
        results = []
        correct = 0
        idk_count = 0
        auprc_samples_by_type = {ft: [] for ft in self.config.auprc_filter_types} if self.config.calculate_auprc else None
        
        for i in range(start_idx, end_idx):
            qa = qas[i]
            
            parts = qa.split("A:")
            if len(parts) != 2:
                self.logger.warning(f"Invalid QA format at index {i}: {qa}")
                continue
                
            prompt = parts[0] + "A:"
            question = parts[0].split("Q:")[1].strip() if "Q:" in parts[0] else ""
            golden_answer = parts[1].strip()
            
            question_type, person = get_info(question)
            
            generated_answer = inference(
                model, prompt, 
                tokenizer=self.enc,
                max_new_tokens=self.config.max_new_tokens,
                stop_token=198,
                temperature=self.config.temperature
            ).strip()
            
            if self.config.calculate_auprc:
                for filter_type in self.config.auprc_filter_types:
                    try:
                        filter_value = calculate_filter_value(
                            model, question, generated_answer,
                            filter_type,
                            {
                                'get_gradient': lambda m, t: self.get_gradient(m, t),
                                'get_loss': lambda m, t: self.get_loss(m, t)
                            }
                        )
                        sample = ThresholdSample(
                            sample_id=i,
                            is_positive=person not in self.known_persons,
                            original_answer=golden_answer,
                            generated_answer=generated_answer,
                            transition_threshold=filter_value,
                            filter_type=filter_type,
                            filter_value=filter_value
                        )
                        auprc_samples_by_type[filter_type].append(sample)
                    except Exception as e:
                        self.logger.warning(f"Failed to calculate filter value for sample {i} with filter {filter_type}: {e}")
            
            filtered_answer = generated_answer
            if self.config.use_gradient_filter:
                filtered_answer = self.apply_gradient_filter(model, question, generated_answer)
            elif self.config.use_loss_filter:
                filtered_answer = self.apply_loss_filter(model, question, generated_answer)
                
            if "I don't know" in golden_answer:
                golden_answer = "I don't know"
            if "I don't know" in generated_answer:
                generated_answer = "I don't know"
                idk_count += 1
            if "I don't know" in filtered_answer:
                filtered_answer = "I don't know"
                
            is_correct = (golden_answer == generated_answer)
            
            result = {
                "prompt": prompt,
                "golden_answer": golden_answer,
                "generated_answer": generated_answer,
                "filtered_answer": filtered_answer,
                "question_type": question_type,
                "person": person,
                "correct": is_correct,
                "person_known": person in self.known_persons
            }
            results.append(result)
            
            if is_correct:
                correct += 1
                
            progress_queue.put(1)
            
        if self.config.calculate_auprc:
            return_dict[process_id] = (results, correct, idk_count, auprc_samples_by_type)
        else:
            return_dict[process_id] = (results, correct, idk_count, None)
        
    def evaluate(self) -> Dict[str, Any]:
        """Run evaluation on the dataset."""
        self.logger.info(f"Starting evaluation of {self.config.model_path}")
        self.logger.info(f"Input: {self.config.input_path}")
        self.logger.info(f"Output: {self.config.output_path}")
        
        self.logger.info("Loading model...")
        n_gpus = torch.cuda.device_count()
        if n_gpus == 0:
            raise RuntimeError("No GPUs available for inference")
            
        model = GPT.from_pretrained(self.config.model_path, f"cuda:0")
        models = {0: model}
        
        for gpu_id in range(1, n_gpus):
            models[gpu_id] = model.to(f"cuda:{gpu_id}")
            
        with open(self.config.input_path, "r", encoding="utf-8") as f:
            qas = f.readlines()
            
        total_samples = len(qas)
        if self.config.first_n > 0:
            total_samples = min(total_samples, self.config.first_n)
            
        self.logger.info(f"Evaluating {total_samples} samples")
        
        total_processes = n_gpus * self.config.processes_per_gpu
        chunk_size = (total_samples + total_processes - 1) // total_processes
        
        processes = []
        manager = mp.Manager()
        return_dict = manager.dict()
        progress_queue = manager.Queue()
        
        self.logger.info(f"Using {n_gpus} GPUs with {self.config.processes_per_gpu} processes per GPU")
        
        for gpu_id in range(n_gpus):
            for proc_idx in range(self.config.processes_per_gpu):
                process_id = gpu_id * self.config.processes_per_gpu + proc_idx
                start_idx = process_id * chunk_size
                end_idx = min(start_idx + chunk_size, total_samples)
                
                if start_idx >= total_samples:
                    continue
                    
                p = mp.Process(
                    target=self.process_chunk,
                    args=(process_id, gpu_id, models[gpu_id], qas, 
                          start_idx, end_idx, return_dict, progress_queue)
                )
                processes.append(p)
                p.start()
                
        pbar = tqdm(total=total_samples, desc="Evaluation Progress")
        completed = 0
        while completed < total_samples:
            progress_queue.get()
            completed += 1
            pbar.update(1)
        pbar.close()
        
        for p in processes:
            p.join()
            
        all_results = []
        total_correct = 0
        total_idk = 0
        all_auprc_samples_by_type = {ft: [] for ft in self.config.auprc_filter_types} if self.config.calculate_auprc else None
        
        for process_id in sorted(return_dict.keys()):
            results_chunk, correct_chunk, idk_chunk, auprc_samples_chunk = return_dict[process_id]
            all_results.extend(results_chunk)
            total_correct += correct_chunk
            total_idk += idk_chunk
            if auprc_samples_chunk:
                for filter_type, samples in auprc_samples_chunk.items():
                    all_auprc_samples_by_type[filter_type].extend(samples)
            
        accuracy = total_correct / total_samples
        idk_ratio = total_idk / total_samples
        self.logger.info(f"Overall Accuracy: {accuracy:.4f}")
        self.logger.info(f"IDK Ratio: {idk_ratio:.4f}")
        
        df = pd.DataFrame(all_results)
        
        # Calculate per-category metrics
        df['is_idk'] = df['generated_answer'] == "I don't know"
        category_metrics = df.groupby("question_type").agg(
            accuracy=('correct', 'mean'),
            idk_ratio=('is_idk', 'mean')
        ).to_dict('index')
        
        # Reformat for JSON output
        category_accuracy = {k: v['accuracy'] for k, v in category_metrics.items()}
        category_idk_ratio = {k: v['idk_ratio'] for k, v in category_metrics.items()}
        
        if "person_known" in df.columns:
            known_accuracy = df[df["person_known"]]["correct"].mean()
            unknown_accuracy = df[~df["person_known"]]["correct"].mean()
        else:
            known_accuracy = unknown_accuracy = None
            
        if self.config.calculate_auprc and all_auprc_samples_by_type:
            for filter_type, samples in all_auprc_samples_by_type.items():
                samples_path = Path(self.config.output_path).with_name(f"{Path(self.config.output_path).stem}.{filter_type}.auprc_samples.json")
                with open(samples_path, 'w') as f:
                    json.dump([asdict(s) for s in samples], f)
                self.logger.info(f"AUPRC samples for filter '{filter_type}' saved to {samples_path}")
            
        results = {
            "accuracy": accuracy,
            "total_samples": total_samples,
            "total_correct": total_correct,
            "idk_ratio": idk_ratio,
            "total_idk": total_idk,
            "detail_result": category_accuracy,
            "detail_idk_ratio": category_idk_ratio,
            "known_person_accuracy": known_accuracy,
            "unknown_person_accuracy": unknown_accuracy,
            "config": asdict(self.config),
            "evaluation_time": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(self.config.output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
            
        if self.config.save_detailed_results:
            csv_path = Path(self.config.output_path).with_suffix(".csv")
            df.to_csv(csv_path, index=False)
            self.logger.info(f"Saved detailed results to {csv_path}")
            
        self.logger.info("Per-category accuracy:")
        for category, acc in category_accuracy.items():
            self.logger.info(f"  {category}: {acc:.4f}")
            
        return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate GPT-2 model on QA dataset")
    
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input QA file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save results")
    
    parser.add_argument("--first_n", type=int, default=-1, help="Number of samples to evaluate (-1 for all)")
    parser.add_argument("--processes_per_gpu", type=int, default=4, help="Number of processes per GPU")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 for deterministic)")
    
    parser.add_argument("--individuals_file", type=str, default="hallucinate_small/individuals.json", help="Path to individuals JSON file")
    
    parser.add_argument("--use_gradient_filter", action="store_true", help="Use gradient-based hallucination filter")
    parser.add_argument("--gradient_threshold", type=float, default=300.0, help="Gradient filter threshold")
    parser.add_argument("--use_loss_filter", action="store_true", help="Use loss-based hallucination filter")
    parser.add_argument("--loss_threshold", type=float, default=3.0, help="Loss filter threshold")
    
    parser.add_argument("--calculate_auprc", action="store_true", help="Calculate AUPRC for hallucination detection")
    parser.add_argument("--auprc_filter_types", type=str, nargs='+', help="List of filter types for AUPRC calculation (e.g., gradient loss)")
    
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    parser.add_argument("--save_detailed_results", action="store_true", help="Save detailed results to CSV")
    
    args = parser.parse_args()
    
    config = InferenceConfig(**vars(args))
    
    evaluator = GPT2Evaluator(config)
    results = evaluator.evaluate()
    
    print(f"\nEvaluation completed!")
    print(f"Overall accuracy: {results['accuracy']:.4f}")
    print(f"Results saved to: {args.output_path}")


if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()