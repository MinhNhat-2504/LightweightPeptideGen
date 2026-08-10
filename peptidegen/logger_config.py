"""
Enhanced logging configuration for LightweightPeptideGen training monitoring.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import torch
import json
from typing import Optional, Dict, Any


class TrainingLogger:
    """
    Enhanced logger for monitoring GAN training progress.
    Logs to both console and file with detailed metrics.
    """
    
    def __init__(
        self,
        log_dir: str = "./logs",
        experiment_name: Optional[str] = None,
        log_level: int = logging.INFO,
        log_gpu_memory: bool = True,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate experiment name
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name
        
        self.log_gpu_memory = log_gpu_memory and torch.cuda.is_available()
        
        # Setup logger
        self.logger = logging.getLogger(f"training_{experiment_name}")
        self.logger.setLevel(log_level)
        self.logger.handlers = []  # Clear existing handlers
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_format = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler
        log_file = self.log_dir / f"train_{experiment_name}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
        
        # Metrics log file (JSON format for easy parsing)
        self.metrics_file = self.log_dir / f"metrics_{experiment_name}.jsonl"
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.epoch_start_time = None
        self.batch_times = []
        
        self.info(f"Logger initialized. Log file: {log_file}")
        
    def info(self, msg: str):
        self.logger.info(msg)
        
    def debug(self, msg: str):
        self.logger.debug(msg)
        
    def warning(self, msg: str):
        self.logger.warning(msg)
        
    def error(self, msg: str):
        self.logger.error(msg)
        
    def get_gpu_memory_info(self) -> Dict[str, float]:
        """Get current GPU memory usage."""
        if not self.log_gpu_memory:
            return {}
            
        try:
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
            
            return {
                'gpu_allocated_gb': round(allocated, 2),
                'gpu_reserved_gb': round(reserved, 2),
                'gpu_max_allocated_gb': round(max_allocated, 2),
            }
        except Exception:
            return {}
            
    def log_config(self, config: Dict[str, Any]):
        """Log training configuration."""
        self.info("=" * 60)
        self.info("TRAINING CONFIGURATION")
        self.info("=" * 60)
        
        def log_dict(d, prefix=""):
            for k, v in d.items():
                if isinstance(v, dict):
                    self.info(f"{prefix}{k}:")
                    log_dict(v, prefix + "  ")
                else:
                    self.info(f"{prefix}{k}: {v}")
                    
        log_dict(config)
        self.info("=" * 60)
        
        # Save config to file
        config_file = self.log_dir / f"config_{self.experiment_name}.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
            
    def log_model_info(
        self,
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        structure_evaluator: Optional[torch.nn.Module] = None,
    ):
        """Log model architecture information."""
        self.info("=" * 60)
        self.info("MODEL INFORMATION")
        self.info("=" * 60)
        
        g_params = sum(p.numel() for p in generator.parameters())
        g_trainable = sum(p.numel() for p in generator.parameters() if p.requires_grad)
        self.info(f"Generator: {g_params:,} total params, {g_trainable:,} trainable")
        
        d_params = sum(p.numel() for p in discriminator.parameters())
        d_trainable = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
        self.info(f"Discriminator: {d_params:,} total params, {d_trainable:,} trainable")
        
        if structure_evaluator is not None:
            se_params = sum(p.numel() for p in structure_evaluator.parameters())
            se_trainable = sum(p.numel() for p in structure_evaluator.parameters() if p.requires_grad)
            self.info(f"Structure Evaluator: {se_params:,} total params, {se_trainable:,} trainable")
            
        # GPU info
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            self.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
            
        self.info("=" * 60)
        
    def log_epoch_start(self, epoch: int, total_epochs: int):
        """Log epoch start."""
        self.epoch = epoch
        self.epoch_start_time = datetime.now()
        self.batch_times = []
        
        self.info(f"\n{'='*60}")
        self.info(f"EPOCH {epoch + 1}/{total_epochs}")
        self.info(f"{'='*60}")
        
        gpu_info = self.get_gpu_memory_info()
        if gpu_info:
            self.info(f"GPU Memory: {gpu_info['gpu_allocated_gb']:.2f}GB allocated, "
                     f"{gpu_info['gpu_reserved_gb']:.2f}GB reserved")
            
    def log_batch(
        self,
        batch_idx: int,
        total_batches: int,
        losses: Dict[str, float],
        batch_time: float,
        log_every: int = 100,
    ):
        """Log batch metrics."""
        self.global_step += 1
        self.batch_times.append(batch_time)
        
        if batch_idx % log_every == 0 or batch_idx == total_batches - 1:
            avg_time = sum(self.batch_times[-log_every:]) / len(self.batch_times[-log_every:])
            eta = avg_time * (total_batches - batch_idx)
            
            msg = f"Batch {batch_idx + 1}/{total_batches} | "
            msg += f"G: {losses.get('g_loss', 0):.4f} | "
            msg += f"D: {losses.get('d_loss', 0):.4f} | "
            msg += f"D_real: {losses.get('d_real', 0):.4f} | "
            msg += f"D_fake: {losses.get('d_fake', 0):.4f} | "
            
            if 'stability_score' in losses:
                msg += f"Stab: {losses['stability_score']:.4f} | "
                
            msg += f"Time: {avg_time:.3f}s/batch | "
            msg += f"ETA: {eta/60:.1f}min"
            
            self.info(msg)
            
            # Log to metrics file
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'epoch': self.epoch,
                'batch': batch_idx,
                'global_step': self.global_step,
                **losses,
                'batch_time': batch_time,
                **self.get_gpu_memory_info(),
            }
            self._write_metrics(metrics)
            
    def log_epoch_end(
        self,
        train_losses: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None,
    ):
        """Log epoch end summary."""
        elapsed = (datetime.now() - self.epoch_start_time).total_seconds()
        
        self.info("-" * 60)
        self.info(f"EPOCH {self.epoch + 1} SUMMARY (took {elapsed/60:.1f} min)")
        self.info("-" * 60)
        
        # Training metrics
        self.info("Training Losses:")
        for k, v in train_losses.items():
            self.info(f"  {k}: {v:.4f}")
            
        # Validation metrics
        if val_metrics:
            self.info("Validation Metrics:")
            for k, v in val_metrics.items():
                self.info(f"  {k}: {v:.4f}")
                
        # GPU memory
        gpu_info = self.get_gpu_memory_info()
        if gpu_info:
            self.info(f"Peak GPU Memory: {gpu_info['gpu_max_allocated_gb']:.2f}GB")
            
        # GAN health indicators
        self._log_gan_health(train_losses)
        
        self.info("-" * 60)
        
    def _log_gan_health(self, losses: Dict[str, float]):
        """Log GAN training health indicators."""
        g_loss = losses.get('g_loss', 0)
        d_loss = losses.get('d_loss', 0)
        d_real = losses.get('d_real', 0)
        d_fake = losses.get('d_fake', 0)
        
        self.info("GAN Health Check:")
        
        # Check for mode collapse
        if d_loss < 0.01:
            self.warning("  ⚠️  D_loss very low - possible mode collapse!")
        elif d_loss > 2.0:
            self.warning("  ⚠️  D_loss too high - discriminator struggling!")
        else:
            self.info("  ✓ D_loss in healthy range")
            
        # Check G/D balance
        if g_loss < -10:
            self.warning("  ⚠️  G_loss very negative - generator might be collapsing!")
        elif g_loss > 10:
            self.warning("  ⚠️  G_loss too high - generator struggling!")
        else:
            self.info("  ✓ G_loss in healthy range")
            
        # Check discriminator outputs
        if abs(d_real - d_fake) < 0.1:
            self.info("  ✓ D cannot distinguish real/fake well (good for generator)")
        elif d_real > 0.9 and d_fake < 0.1:
            self.warning("  ⚠️  D too strong - generator needs more training")
            
    def log_checkpoint(self, path: str, is_best: bool = False):
        """Log checkpoint save."""
        prefix = "🏆 BEST " if is_best else ""
        self.info(f"{prefix}Checkpoint saved: {path}")
        
    def log_early_stopping(self, epoch: int, patience: int):
        """Log early stopping."""
        self.warning(f"Early stopping triggered at epoch {epoch} (patience: {patience})")
        
    def log_training_complete(self, total_epochs: int, total_time: float):
        """Log training completion."""
        self.info("=" * 60)
        self.info("🎉 TRAINING COMPLETE!")
        self.info(f"Total epochs: {total_epochs}")
        self.info(f"Total time: {total_time/3600:.2f} hours")
        self.info(f"Metrics saved to: {self.metrics_file}")
        self.info("=" * 60)
        
    def _write_metrics(self, metrics: Dict[str, Any]):
        """Write metrics to JSONL file."""
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')


def setup_training_logger(
    log_dir: str = "./logs",
    experiment_name: Optional[str] = None,
) -> TrainingLogger:
    """
    Setup training logger.
    
    Args:
        log_dir: Directory for log files
        experiment_name: Name for this experiment
        
    Returns:
        TrainingLogger instance
    """
    return TrainingLogger(
        log_dir=log_dir,
        experiment_name=experiment_name,
    )


# Quick status check function
def check_training_status(log_dir: str = "./logs") -> Dict[str, Any]:
    """
    Check training status from latest metrics file.
    
    Args:
        log_dir: Directory containing log files
        
    Returns:
        Dictionary with latest training status
    """
    log_path = Path(log_dir)
    metrics_files = list(log_path.glob("metrics_*.jsonl"))
    
    if not metrics_files:
        return {"status": "No training logs found"}
        
    # Get latest file
    latest_file = max(metrics_files, key=lambda x: x.stat().st_mtime)
    
    # Read last few lines
    with open(latest_file, 'r') as f:
        lines = f.readlines()
        
    if not lines:
        return {"status": "Empty metrics file"}
        
    # Parse last entry
    last_entry = json.loads(lines[-1])
    
    return {
        "status": "Training in progress" if len(lines) > 0 else "Unknown",
        "latest_metrics": last_entry,
        "total_steps": len(lines),
        "log_file": str(latest_file),
    }
