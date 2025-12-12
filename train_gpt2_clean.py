import os
import sys
import json
import time
import uuid
import glob
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import GPT, GPTConfig
from dataloader import DistributedDataLoader


@dataclass
class TrainingConfig:
    """Configuration for training GPT-2 model."""
    # Data parameters
    input_folder: str
    sequence_length: int = 512
    val_tokens: int = 0
    
    # Model parameters
    model_size: str = "base"
    load_checkpoint: Optional[str] = None
    
    # Optimization parameters
    batch_size: int = 512  # Total batch size across all devices
    device_batch_size: int = 64  # Per device batch size
    learning_rate: float = 0.0036
    weight_decay: float = 0.0
    num_epochs: int = 1
    warmup_ratio: float = 0.0
    warmdown_ratio: float = 1.0
    
    # Training options
    bf16: bool = False
    ortho: bool = False
    mask_q: bool = False
    random_align: bool = False
    
    # Logging and checkpointing
    output_dir: str = "experiments"
    run_name: Optional[str] = None
    val_loss_every: int = 125
    save_every: int = 0
    log_gradient_stats: bool = True
    
    # Wandb integration
    use_wandb: bool = False
    wandb_project: str = "gpt2-finetune"
    
    def __post_init__(self):
        if self.run_name is None:
            self.run_name = f"gpt2_{self.model_size}_{time.strftime('%Y%m%d_%H%M%S')}"


class GPT2Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.setup_distributed()
        self.setup_directories()
        self.setup_logging()
        self.load_data()
        self.setup_model()
        self.setup_optimization()
        
    def setup_distributed(self):
        """Setup distributed training environment."""
        assert torch.cuda.is_available(), "CUDA is required for training"
        dist.init_process_group(backend="nccl")
        self.ddp_rank = int(os.environ["RANK"])
        self.ddp_local_rank = int(os.environ["LOCAL_RANK"])
        self.ddp_world_size = int(os.environ["WORLD_SIZE"])
        self.device = f"cuda:{self.ddp_local_rank}"
        torch.cuda.set_device(self.device)
        self.master_process = self.ddp_rank == 0
        
        self.logger_dist = logging.getLogger(f"trainer_rank_{self.ddp_rank}")
        self.logger_dist.info(f"Initialized process rank={self.ddp_rank}, device={self.device}")
        
    def setup_directories(self):
        """Create output directories."""
        self.exp_dir = Path(self.config.output_dir) / self.config.run_name
        if self.master_process:
            self.exp_dir.mkdir(parents=True, exist_ok=True)
            (self.exp_dir / "checkpoints").mkdir(exist_ok=True)
            (self.exp_dir / "logs").mkdir(exist_ok=True)
            
    def setup_logging(self):
        """Setup logging configuration."""
        if self.master_process:
            log_file = self.exp_dir / "logs" / "training.log"
            
            # Setup file handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            
            # Setup console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            # Setup logger
            self.logger = logging.getLogger("gpt2_trainer")
            self.logger.setLevel(logging.INFO)
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
            
            # Log configuration
            self.logger.info("Training Configuration:")
            self.logger.info(json.dumps(asdict(self.config), indent=2))
            
            # Save configuration
            with open(self.exp_dir / "config.json", 'w') as f:
                json.dump(asdict(self.config), f, indent=2)
                
            # Setup metrics logging
            self.metrics_file = self.exp_dir / "logs" / "metrics.jsonl"
            
            # Setup wandb if requested
            if self.config.use_wandb:
                try:
                    import wandb
                    wandb.init(
                        project=self.config.wandb_project,
                        name=self.config.run_name,
                        config=asdict(self.config)
                    )
                    self.wandb = wandb
                except ImportError:
                    self.logger.warning("Wandb not installed, disabling wandb logging")
                    self.wandb = None
                    self.config.use_wandb = False
            else:
                self.wandb = None
                
    def load_data(self):
        """Load training and validation data."""
        # Load metadata
        metadata_file = Path(self.config.input_folder) / "metadata.json"
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
            
        # Calculate training parameters
        self.do_val = self.config.val_tokens > 0
        B, T = self.config.device_batch_size, self.config.sequence_length
        
        # Validation steps
        if self.do_val:
            assert self.config.val_tokens % (B * T * self.ddp_world_size) == 0
            self.val_steps = self.config.val_tokens // (B * T * self.ddp_world_size)
        
        # Gradient accumulation steps
        assert self.config.batch_size % (B * self.ddp_world_size) == 0
        self.train_accumulation_steps = self.config.batch_size // (B * self.ddp_world_size)
        
        # Total iterations
        self.num_iterations = (
            self.config.num_epochs * self.metadata["train_tokens"] // 
            (T * self.config.batch_size)
        )
        self.warmup_iters = int(self.config.warmup_ratio * self.num_iterations)
        self.warmdown_iters = int(self.config.warmdown_ratio * self.num_iterations)
        
        if self.master_process:
            self.logger.info(f"Training iterations: {self.num_iterations}")
            self.logger.info(f"Warmup iterations: {self.warmup_iters}")
            self.logger.info(f"Warmdown iterations: {self.warmdown_iters}")
            self.logger.info(f"Tokens per iteration: {T * self.config.batch_size}")
            self.logger.info(f"Gradient accumulation steps: {self.train_accumulation_steps}")
            
        # Setup data loaders
        train_pattern = str(Path(self.config.input_folder) / "*_train_*.bin")
        self.train_loader = DistributedDataLoader(
            train_pattern, B, T, self.ddp_rank, self.ddp_world_size
        )
        
        if self.do_val:
            val_pattern = str(Path(self.config.input_folder) / "*_val_*.bin")
            self.val_loader = DistributedDataLoader(
                val_pattern, B, T, self.ddp_rank, self.ddp_world_size
            )
            
        if self.master_process:
            self.logger.info(
                f"Train data: {self.train_loader.ntok_total} tokens, "
                f"{len(self.train_loader.files)} files"
            )
            if self.do_val:
                self.logger.info(
                    f"Val data: {self.val_loader.ntok_total} tokens, "
                    f"{len(self.val_loader.files)} files"
                )
                
    def setup_model(self):
        """Initialize or load the model."""
        # Model configurations
        model_configs = {
            "nano": dict(n_layer=4, n_head=3, n_embd=192),
            "xxs": dict(n_layer=5, n_head=4, n_embd=256),
            "tiny": dict(n_layer=6, n_head=5, n_embd=320),
            "xs": dict(n_layer=7, n_head=6, n_embd=384),
            "small1": dict(n_layer=8, n_head=7, n_embd=448),
            "small": dict(n_layer=8, n_head=8, n_embd=512),
            "medium": dict(n_layer=9, n_head=9, n_embd=576),
            "large": dict(n_layer=10, n_head=10, n_embd=640),
            "xl": dict(n_layer=11, n_head=11, n_embd=704),
            "base": dict(n_layer=12, n_head=12, n_embd=768),
            "x0": dict(n_layer=16, n_head=16, n_embd=1024),
            "x1": dict(n_layer=20, n_head=16, n_embd=1024),
            "x2": dict(n_layer=24, n_head=20, n_embd=1440),
            "x3": dict(n_layer=32, n_head=25, n_embd=1600),
        }
        
        num_vocab = 50304
        
        if self.config.load_checkpoint:
            if self.master_process:
                self.logger.info(f"Loading checkpoint from {self.config.load_checkpoint}")
            self.model = GPT.from_pretrained(self.config.load_checkpoint, device=self.device)
        else:
            config = model_configs[self.config.model_size]
            self.model = GPT(GPTConfig(vocab_size=num_vocab, **config))
            
        self.model = self.model.cuda()
        self.model = DDP(self.model, device_ids=[self.ddp_local_rank])
        self.raw_model = self.model.module
        
        # Setup mixed precision context
        if self.config.bf16:
            self.ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:
            self.ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float32)
            
        if self.master_process:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.logger.info(f"Model size: {self.config.model_size}")
            self.logger.info(f"Total parameters: {total_params:,}")
            self.logger.info(f"Trainable parameters: {trainable_params:,}")
            
    def setup_optimization(self):
        """Setup optimizers and schedulers."""
        self.optimizer = torch.optim.AdamW(
            self.raw_model.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=self.config.weight_decay,
            fused=True,
            eps=1e-6,
        )
        
        self.optimizers = [self.optimizer]
        
        # Learning rate scheduler
        def get_lr(it):
            assert it <= self.num_iterations
            # Linear warmup
            if it < self.warmup_iters:
                return (it + 1) / self.warmup_iters
            # Constant lr
            elif it < self.num_iterations - self.warmdown_iters:
                return 1.0
            # Linear warmdown
            else:
                decay_ratio = 0.1 + 0.9 * (self.num_iterations - it) / self.warmdown_iters
                return decay_ratio
                
        self.schedulers = [
            torch.optim.lr_scheduler.LambdaLR(opt, get_lr) 
            for opt in self.optimizers
        ]
        
        # Initialize gradient tracking for orthogonal descent if enabled
        if self.config.ortho:
            self.grad_last = []
            
    def log_metrics(self, metrics: Dict[str, Any], step: int):
        """Log metrics to file and wandb."""
        if self.master_process:
            # Add timestamp and step
            metrics['step'] = step
            metrics['timestamp'] = time.time()
            
            # Log to file
            with open(self.metrics_file, 'a') as f:
                f.write(json.dumps(metrics) + '\n')
                
            # Log to wandb
            if self.wandb:
                self.wandb.log(metrics, step=step)
                
            # Log to console
            log_str = f"Step {step}/{self.num_iterations}"
            for key, value in metrics.items():
                if key not in ['step', 'timestamp']:
                    if isinstance(value, float):
                        log_str += f" | {key}: {value:.4f}"
                    else:
                        log_str += f" | {key}: {value}"
            self.logger.info(log_str)
            
    def evaluate(self, step: int):
        """Run validation evaluation."""
        if not self.do_val:
            return
            
        self.model.eval()
        self.val_loader.reset()
        val_loss = 0.0
        
        for _ in range(self.val_steps):
            x_val, y_val = self.val_loader.next_batch()
            with self.ctx:
                _, loss = self.model(
                    x_val, y_val, 
                    return_logits=False, 
                    mask_q=self.config.mask_q, 
                    random_align=self.config.random_align
                )
                val_loss += loss.detach()
                del loss
                
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss /= self.val_steps
        
        metrics = {'val_loss': val_loss.item()}
        self.log_metrics(metrics, step)
        
    def save_checkpoint(self, step: int, final: bool = False):
        """Save model checkpoint."""
        if self.master_process and final:
            checkpoint = {
                'step': step,
                'model': self.raw_model.state_dict(),
                'config': self.raw_model.config,
                'training_config': asdict(self.config),
                'optimizer': self.optimizer.state_dict(),
            }
            path = self.exp_dir / "checkpoints" / "final.pt"
            torch.save(checkpoint, path)
            self.logger.info(f"Saved final checkpoint to {path}")
            
    def train_step(self, step: int):
        """Execute one training step."""
        self.model.train()
        
        # Gradient accumulation
        for i in range(1, self.train_accumulation_steps + 1):
            # Get batch
            x, y = self.train_loader.next_batch()
            
            # Forward pass
            with self.ctx:
                _, loss = self.model(
                    x, y, 
                    return_logits=False, 
                    random_align=self.config.random_align, 
                    mask_q=self.config.mask_q
                )
                train_loss = loss.detach()
                
            # Backward pass
            if i < self.train_accumulation_steps:
                with self.model.no_sync():
                    loss.backward()
            else:
                loss.backward()
                
        # Gradient processing
        grad_stats = {}
        if self.config.log_gradient_stats:
            grad_all = []
            
        for i, (name, p) in enumerate(self.model.named_parameters()):
            if p.requires_grad and p.grad is not None:
                # Scale gradients
                p.grad /= self.train_accumulation_steps
                
                # Orthogonal gradient descent if enabled
                if self.config.ortho:
                    if len(self.grad_last) > i:
                        p.grad -= (
                            torch.dot(self.grad_last[i], p.grad) * 
                            self.grad_last[i] / 
                            torch.dot(self.grad_last[i], self.grad_last[i])
                        )
                        self.grad_last[i] = self.grad_last[i] * 0.9 + p.grad.clone() * 0.1
                    else:
                        self.grad_last.append(p.grad.clone())
                        
                # Collect gradient statistics
                if self.config.log_gradient_stats:
                    grad_all.append(p.grad.view(-1))
                    
        # Log gradient statistics
        if self.config.log_gradient_stats and grad_all:
            grad_all = torch.cat(grad_all)
            grad_stats = {
                'grad_norm': grad_all.norm().item(),
                'grad_mean': grad_all.mean().item(),
                'grad_std': grad_all.std().item(),
                'grad_nonzero_ratio': (grad_all.abs() > 1e-5).float().mean().item()
            }
            
        # Optimizer step
        for opt, sched in zip(self.optimizers, self.schedulers):
            opt.step()
            sched.step()
            
        # Zero gradients
        self.model.zero_grad(set_to_none=True)
        
        # Return metrics
        current_lr = self.schedulers[0].get_last_lr()[0]
        metrics = {
            'train_loss': train_loss.item(),
            'learning_rate': current_lr,
            **grad_stats
        }
        
        return metrics
        
    def train(self):
        """Main training loop."""
        if self.master_process:
            self.logger.info("Starting training...")
            
        # Initialize timers
        training_time_ms = 0
        torch.cuda.synchronize()
        t0 = time.time()
        
        # Reset data loader
        self.train_loader.reset()
        
        for step in range(self.num_iterations + 1):
            last_step = step == self.num_iterations
            
            # Reset timer after warmup
            if step == 10:
                training_time_ms = 0
                t0 = time.time()
                
            # Validation
            if self.do_val and (last_step or (self.config.val_loss_every > 0 and step % self.config.val_loss_every == 0)):
                torch.cuda.synchronize()
                training_time_ms += 1000 * (time.time() - t0)
                self.evaluate(step)
                torch.cuda.synchronize()
                t0 = time.time()
                
            # Save checkpoint
            if self.master_process and (last_step or (self.config.save_every > 0 and step % self.config.save_every == 0)):
                torch.cuda.synchronize()
                training_time_ms += 1000 * (time.time() - t0)
                self.save_checkpoint(step, final=last_step)
                torch.cuda.synchronize()
                t0 = time.time()
                
            # Exit after final evaluation/save
            if last_step:
                break
                
            # Training step
            metrics = self.train_step(step)
            
            # Add timing information
            if step > 10:
                timed_steps = step - 10 + 1
                metrics['time_ms'] = training_time_ms + 1000 * (time.time() - t0)
                metrics['ms_per_step'] = metrics['time_ms'] / timed_steps
                eta_minutes = metrics['ms_per_step'] * (self.num_iterations - step) / 1000 / 60
                metrics['eta_minutes'] = eta_minutes
                
            # Log metrics
            self.log_metrics(metrics, step + 1)
            
        # Final logging
        if self.master_process:
            peak_memory = torch.cuda.max_memory_allocated() // 1024 // 1024
            self.logger.info(f"Training completed!")
            self.logger.info(f"Peak memory consumption: {peak_memory} MiB")
            
            # Save final statistics
            final_stats = {
                'total_iterations': self.num_iterations,
                'total_time_ms': training_time_ms,
                'peak_memory_mb': peak_memory,
                'final_learning_rate': self.schedulers[0].get_last_lr()[0],
            }
            
            with open(self.exp_dir / "final_stats.json", 'w') as f:
                json.dump(final_stats, f, indent=2)
                
    def cleanup(self):
        """Clean up distributed training."""
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Train GPT-2 model")
    
    # Data parameters
    parser.add_argument("--input_folder", type=str, required=True, help="Input data folder")
    parser.add_argument("--sequence_length", type=int, default=512, help="Sequence length in tokens")
    parser.add_argument("--val_tokens", type=int, default=0, help="Number of validation tokens")
    
    # Model parameters
    parser.add_argument("--model_size", type=str, default="base",
                       choices=["nano", "xxs", "tiny", "xs", "small1", "small", "medium", "large", "xl", "base", "x0", "x1", "x2", "x3"],
                       help="Model size configuration")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Path to checkpoint to load")
    
    # Optimization parameters
    parser.add_argument("--batch_size", type=int, default=512, help="Total batch size across all devices")
    parser.add_argument("--device_batch_size", type=int, default=64, help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=0.0036, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--warmup_ratio", type=float, default=0.0, help="Warmup ratio")
    parser.add_argument("--warmdown_ratio", type=float, default=1.0, help="Warmdown ratio")
    
    # Training options
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 precision")
    parser.add_argument("--ortho", action="store_true", help="Use orthogonal gradient descent")
    parser.add_argument("--mask_q", action="store_true", help="Mask question tokens in loss")
    parser.add_argument("--random_align", action="store_true", help="Use random alignment loss")
    
    # Logging and checkpointing
    parser.add_argument("--output_dir", type=str, default="experiments", help="Output directory")
    parser.add_argument("--run_name", type=str, default=None, help="Run name")
    parser.add_argument("--val_loss_every", type=int, default=125, help="Validation frequency")
    parser.add_argument("--save_every", type=int, default=0, help="Checkpoint save frequency")
    parser.add_argument("--log_gradient_stats", action="store_true", help="Log gradient statistics")
    
    # Wandb integration
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="gpt2-finetune", help="Wandb project name")
    
    args = parser.parse_args()
    
    # Create configuration
    config = TrainingConfig(**vars(args))
    
    # Create trainer and run training
    trainer = GPT2Trainer(config)
    try:
        trainer.train()
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
