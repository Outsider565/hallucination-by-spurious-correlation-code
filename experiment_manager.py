import os
import json
import shutil
import datetime
import subprocess
import sys
from pathlib import Path
import argparse
import logging
from typing import Dict, Any, List, Optional
import hashlib
import numpy as np

# Get the base directory of the project
BASE_DIR = Path(__file__).parent.resolve()

class ExperimentManager:
    def __init__(self, config_path: str, resume_dir: Optional[str] = None):
        """Initialize experiment manager with a configuration file."""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.experiment_name = self.config['experiment_name']
        self.base_dir = Path(self.config.get('base_dir', 'experiments'))
        
        if resume_dir:
            self.exp_dir = Path(resume_dir)
            self.timestamp = self.exp_dir.name.replace(f"{self.experiment_name}_", "")
            print(f"Resuming from specified directory: {self.exp_dir}")
            # Re-setup logger to point to the resumed directory's log file
            self._setup_logging()
        else:
            self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.exp_dir = self.base_dir / f"{self.experiment_name}_{self.timestamp}"
            # Create experiment directory structure
            self._create_directories()
            # Setup logging for a new experiment
            self._setup_logging()
            # Copy configuration file to experiment directory
            shutil.copy2(config_path, self.exp_dir / 'config.json')

            
    def _create_directories(self):
        """Create experiment directory structure."""
        directories = ['data', 'models', 'logs', 'results', 'scripts']
        for directory in directories:
            (self.exp_dir / directory).mkdir(parents=True, exist_ok=True)
            
    def _setup_logging(self):
        """Setup logging for the experiment."""
        log_file = self.exp_dir / 'logs' / 'experiment.log'
        
        # This removes all handlers associated with the root logger.
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='a'), # Append mode for resumed runs
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def log_experiment_info(self):
        """Log experiment configuration and environment information."""
        self.logger.info(f"Experiment: {self.experiment_name}")
        self.logger.info(f"Timestamp: {self.timestamp}")
        self.logger.info(f"Directory: {self.exp_dir}")
        self.logger.info("Configuration:")
        self.logger.info(json.dumps(self.config, indent=2))
        
        # Log system information
        self.logger.info("\nSystem Information:")
        try:
            # Get GPU information
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], 
                                 capture_output=True, text=True)
            if result.returncode == 0:
                self.logger.info(f"GPU Info:\n{result.stdout}")
        except:
            pass
            
    def generate_data(self):
        """Generate data using the configured parameters, or link to existing data."""
        data_config = self.config['data_generation']
        current_prob = data_config.get('family_city_probability', 0.9)

        # Check if data for this probability already exists in another run
        for existing_run in self.base_dir.iterdir():
            if existing_run.is_dir() and existing_run != self.exp_dir:
                config_path = existing_run / 'config.json'
                data_stats_path = existing_run / 'data' / 'statistics.json'
                if config_path.exists() and data_stats_path.exists():
                    try:
                        with open(config_path, 'r') as f:
                            existing_config = json.load(f)
                        existing_prob = existing_config.get('data_generation', {}).get('family_city_probability')
                        if existing_prob == current_prob:
                            self.logger.info(f"Found existing data for probability {current_prob} in {existing_run}")
                            # Symlink the data directory
                            source_data_dir = existing_run / 'data'
                            target_data_dir = self.exp_dir / 'data'
                            
                            # Remove the empty data dir and create a symlink
                            if target_data_dir.is_dir():
                                target_data_dir.rmdir()
                            os.symlink(source_data_dir.resolve(), target_data_dir, target_is_directory=True)
                            self.logger.info(f"Successfully linked data from {source_data_dir} to {target_data_dir}")
                            return # Skip data generation
                    except Exception as e:
                        self.logger.warning(f"Could not check existing run {existing_run} due to error: {e}")

        # If no existing data was found, generate it
        self.logger.info("No existing data found for this configuration. Starting data generation...")
        
        cmd = [
            sys.executable, str(BASE_DIR / 'generate_bios_correlation_clean.py'),
            '--K', str(data_config['num_individuals']),
            '--ratio', str(data_config['mix_ratio']),
            '--probability', str(current_prob),
            '--output_dir', str(self.exp_dir / 'data'),
            '--data_dir', str(BASE_DIR / data_config.get('source_data_dir', 'data'))
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            self.logger.error(f"Data generation failed with exit code {result.returncode}")
            self.logger.error(f"Stderr:\n{result.stderr}")
            raise RuntimeError("Data generation failed")
            
        self.logger.info("Data generation completed successfully")
        self._log_data_statistics()
        
    def _log_data_statistics(self):
        """Log statistics about generated data."""
        data_dir = self.exp_dir / 'data'
        
        stats = {}
        for file_path in data_dir.glob('*.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                stats[file_path.name] = {
                    'num_lines': len(lines),
                    'file_size_mb': file_path.stat().st_size / (1024 * 1024),
                    'md5': self._calculate_md5(file_path)
                }
                
        self.logger.info("Data Statistics:")
        self.logger.info(json.dumps(stats, indent=2))
        
        # Save statistics to file
        with open(self.exp_dir / 'data' / 'statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)
            
    def _calculate_md5(self, file_path):
        """Calculate MD5 hash of a file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
        
    def train_model(self, stage: str = 'pretrain'):
        """Train model with configured parameters."""
        train_config = self.config['training'][stage]
        
        # Check if model checkpoint already exists
        run_name = f"{self.experiment_name}_{stage}"
        final_checkpoint_path = self.exp_dir / 'models' / run_name / 'checkpoints' / 'final.pt'
        if final_checkpoint_path.exists():
            self.logger.info(f"{stage} model checkpoint already exists, skipping training.")
            return

        self.logger.info(f"Starting {stage} training...")
        
        # Build training command
        cmd = [
            'torchrun', '--standalone',
            '--nproc_per_node', str(train_config.get('num_gpus', 1)),
            str(BASE_DIR / 'train_gpt2_clean.py'),
            '--input_folder', str(self.exp_dir / 'data' / train_config['data_folder']),
            '--output_dir', str(self.exp_dir / 'models'),
            '--run_name', f"{self.experiment_name}_{stage}",
            '--batch_size', str(train_config['batch_size']),
            '--device_batch_size', str(train_config['device_batch_size']),
            '--sequence_length', str(train_config['sequence_length']),
            '--learning_rate', str(train_config['learning_rate']),
            '--num_epochs', str(train_config['num_epochs']),
            '--warmup_ratio', str(train_config.get('warmup_ratio', 0.05)),
            '--warmdown_ratio', str(train_config.get('warmdown_ratio', 0.1)),
            '--weight_decay', str(train_config.get('weight_decay', 0.0)),
            '--save_every', str(train_config.get('save_every', 1000)),
            '--val_loss_every', str(train_config.get('val_loss_every', 1000)),
            '--model_size', train_config.get('model_size', 'base'),
        ]
        
        # Add optional parameters
        if train_config.get('use_bf16', False):
            cmd.append('--bf16')
            
        if 'load_checkpoint' in train_config:
            checkpoint_path = self.exp_dir / 'models' / train_config['load_checkpoint']
            cmd.extend(['--load_checkpoint', str(checkpoint_path)])
            
        if train_config.get('val_tokens', 0) > 0:
            cmd.extend(['--val_tokens', str(train_config['val_tokens'])])
            
        # Set environment variables
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = train_config.get('cuda_devices', '0')
        
        self.logger.info(f"Running: {' '.join(cmd)}")
        self.logger.info(f"CUDA_VISIBLE_DEVICES: {env['CUDA_VISIBLE_DEVICES']}")
        
        # Run training
        stderr_log_path = self.exp_dir / "logs" / f"{stage}_stderr.log"
        with open(stderr_log_path, 'w') as stderr_log:
            result = subprocess.run(cmd, env=env, stderr=stderr_log)
        
        if result.returncode != 0:
            self.logger.error(f"Training failed with exit code {result.returncode}")
            raise RuntimeError(f"{stage} training failed")
            
        self.logger.info(f"{stage} training completed successfully")
        
    def run_inference(self, eval_name: str):
        """Run model inference and save results and samples."""
        eval_config = self.config['evaluation'][eval_name]
        output_path = self.exp_dir / 'results' / f"{eval_name}.json"
        
        self.logger.info(f"Starting inference: {eval_name}")
        
        cmd = [
            sys.executable, str(BASE_DIR / 'inference_SFT_clean.py'),
            '--model_path', str(self.exp_dir / 'models' / eval_config['model_path']),
            '--input_path', str(self.exp_dir / 'data' / eval_config['test_data']),
            '--output_path', str(output_path),
            '--individuals_file', str(self.exp_dir / 'data' / 'individuals.json'),
            '--processes_per_gpu', str(eval_config.get('processes_per_gpu', 4))
        ]
        if 'first_n' in eval_config:
            cmd.extend(['--first_n', str(eval_config['first_n'])])
        if eval_config.get('calculate_auprc', False) and self.config.get('filter_types_to_test'):
            cmd.append('--calculate_auprc')
            cmd.extend(['--auprc_filter_types', *self.config['filter_types_to_test']])

        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = eval_config.get('cuda_devices', '0')
        self.logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, env=env)
        
        if result.returncode != 0:
            self.logger.error(f"Inference for {eval_name} failed with exit code {result.returncode}")
            raise RuntimeError(f"Inference {eval_name} failed")
        self.logger.info(f"Inference {eval_name} completed successfully")

    def run_analysis(self, eval_name: str):
        """Run analysis based on evaluation type."""
        eval_config = self.config['evaluation'][eval_name]
        
        if eval_config.get('calculate_auprc', False):
            self._run_auprc_analysis(eval_name, eval_config)
        
        if eval_config.get('calculate_category_accuracy', False):
            self._run_category_accuracy_analysis(eval_name, eval_config)

    def _run_auprc_analysis(self, eval_name: str, eval_config: Dict):
        """Run AUPRC analysis on saved samples."""
        # This method remains unchanged
        pass

    def _run_category_accuracy_analysis(self, eval_name: str, eval_config: Dict):
        """Run category accuracy analysis."""
        # This method remains unchanged
        pass

    def run_full_experiment(self):
        """Run the complete experiment pipeline."""
        self.logger.info("Starting full experiment pipeline...")
        
        try:
            # 1. Log experiment information
            self.log_experiment_info()
            
            # 2. Generate data
            if self.config.get('generate_data', True):
                self.generate_data()
            
            # 3. Run training stages
            if self.config.get('training'):
                for stage in self.config['training']:
                    self.train_model(stage)
                
            # 4. Run inference and analysis
            if self.config.get('evaluation'):
                for eval_name in self.config['evaluation']:
                    self.run_inference(eval_name)
                for eval_name in self.config['evaluation']:
                    self.run_analysis(eval_name)
                
            self.logger.info("Experiment completed successfully!")
            
            # Create summary
            self._create_summary()
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {str(e)}")
            raise
            
    def _create_summary(self):
        """Create a summary of the experiment."""
        summary = {
            'experiment_name': self.experiment_name,
            'timestamp': self.timestamp,
            'config': self.config,
            'results': {}
        }
        
        # Collect all results
        for result_file in (self.exp_dir / 'results').glob('*.json'):
            with open(result_file, 'r') as f:
                try:
                    summary['results'][result_file.stem] = json.load(f)
                except json.JSONDecodeError:
                    self.logger.warning(f"Could not decode summary result file: {result_file}")
                
        # Save summary
        with open(self.exp_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
            
        self.logger.info("Experiment summary saved to summary.json")


def main():
    parser = argparse.ArgumentParser(description='Run experiments with proper management')
    parser.add_argument('config', type=str, help='Path to experiment configuration file')
    parser.add_argument('--stage', type=str, choices=['data', 'train', 'eval', 'all'],
                       default='all', help='Which stage to run')
    parser.add_argument('--resume-dir', type=str, help='Path to an existing experiment directory to resume')
    args = parser.parse_args()
    
    manager = ExperimentManager(args.config, resume_dir=args.resume_dir)
    
    # Run requested stage
    if args.stage == 'all':
        manager.run_full_experiment()
    elif args.stage == 'data':
        manager.log_experiment_info()
        manager.generate_data()
    elif args.stage == 'train':
        manager.log_experiment_info()
        if manager.config.get('training'):
            for stage in manager.config['training']:
                manager.train_model(stage)
    elif args.stage == 'eval':
        manager.log_experiment_info()
        if manager.config.get('evaluation'):
            for eval_name in manager.config['evaluation']:
                manager.run_inference(eval_name)
            for eval_name in manager.config['evaluation']:
                manager.run_analysis(eval_name)


if __name__ == '__main__':
    main()