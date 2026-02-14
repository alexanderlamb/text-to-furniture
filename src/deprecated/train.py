"""
Training script for the furniture generation transformer.

Usage:
    python src/train.py --data training_data --epochs 100 --batch-size 32
"""
import sys
import os
import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from model import (
    FurnitureTransformer, 
    FurnitureModelConfig, 
    FurnitureLoss,
    ComponentTokenizer,
    MATERIAL_VOCAB,
)


# =============================================================================
# Dataset
# =============================================================================

class FurnitureDataset(Dataset):
    """
    Dataset for furniture designs.
    
    Loads designs from JSON files and converts them to model format.
    """
    
    def __init__(self, data_dir: str, max_components: int = 20):
        self.data_dir = Path(data_dir)
        self.max_components = max_components
        
        # Load all designs
        self.designs = []
        self._load_designs()
        
        print(f"Loaded {len(self.designs)} designs from {data_dir}")
    
    def _load_designs(self):
        """Load all design JSON files."""
        designs_dir = self.data_dir / "designs"
        
        if not designs_dir.exists():
            raise ValueError(f"Designs directory not found: {designs_dir}")
        
        for design_dir in sorted(designs_dir.iterdir()):
            if design_dir.is_dir():
                json_path = design_dir / "design.json"
                if json_path.exists():
                    with open(json_path) as f:
                        data = json.load(f)
                    
                    # Only include designs with components
                    if data.get("components"):
                        self.designs.append(data)
    
    def __len__(self) -> int:
        return len(self.designs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training example.
        
        Returns:
            Dict with:
            - continuous: (seq_len, 9) normalized continuous features
            - material_ids: (seq_len,) material indices
            - eos_target: (seq_len,) binary EOS targets
            - mask: (seq_len,) valid position mask
            - length: actual number of components
        """
        design = self.designs[idx]
        components = design["components"]
        
        # Truncate if needed
        components = components[:self.max_components]
        num_components = len(components)
        
        # Initialize tensors
        continuous = torch.zeros(self.max_components, 9)
        material_ids = torch.zeros(self.max_components, dtype=torch.long)
        eos_target = torch.zeros(self.max_components)
        mask = torch.zeros(self.max_components)
        
        # Fill in component data
        for i, comp in enumerate(components):
            cont, mat_idx = ComponentTokenizer.normalize_component(comp)
            continuous[i] = cont
            material_ids[i] = mat_idx
            mask[i] = 1.0
        
        # EOS at last valid position
        if num_components > 0:
            eos_target[num_components - 1] = 1.0
        
        return {
            "continuous": continuous,
            "material_ids": material_ids,
            "eos_target": eos_target,
            "mask": mask,
            "length": num_components,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate batch of examples."""
    return {
        "continuous": torch.stack([b["continuous"] for b in batch]),
        "material_ids": torch.stack([b["material_ids"] for b in batch]),
        "eos_target": torch.stack([b["eos_target"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "lengths": torch.tensor([b["length"] for b in batch]),
    }


# =============================================================================
# Training Loop
# =============================================================================

class Trainer:
    """Training manager for furniture transformer."""
    
    def __init__(
        self,
        model: FurnitureTransformer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: FurnitureModelConfig,
        output_dir: str,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.output_dir = Path(output_dir)
        self.device = device
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss function
        self.loss_fn = FurnitureLoss()
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Scheduler
        self.scheduler = None
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # History
        self.train_history = []
        self.val_history = []
    
    def train_epoch(self) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        
        total_loss = 0
        total_continuous = 0
        total_material = 0
        total_eos = 0
        num_batches = 0
        
        for batch in self.train_loader:
            # Move to device
            continuous = batch["continuous"].to(self.device)
            material_ids = batch["material_ids"].to(self.device)
            eos_target = batch["eos_target"].to(self.device)
            mask = batch["mask"].to(self.device)
            
            # Forward pass
            cont_pred, mat_logits, eos_logits = self.model(continuous, material_ids)
            
            # Compute loss
            loss, loss_dict = self.loss_fn(
                cont_pred, continuous,
                mat_logits, material_ids,
                eos_logits, eos_target,
                mask=mask,
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            # Track losses
            total_loss += loss_dict["total"]
            total_continuous += loss_dict["continuous"]
            total_material += loss_dict["material"]
            total_eos += loss_dict["eos"]
            num_batches += 1
            self.global_step += 1
        
        return {
            "loss": total_loss / num_batches,
            "continuous": total_continuous / num_batches,
            "material": total_material / num_batches,
            "eos": total_eos / num_batches,
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0
        total_continuous = 0
        total_material = 0
        total_eos = 0
        num_batches = 0
        
        for batch in self.val_loader:
            continuous = batch["continuous"].to(self.device)
            material_ids = batch["material_ids"].to(self.device)
            eos_target = batch["eos_target"].to(self.device)
            mask = batch["mask"].to(self.device)
            
            cont_pred, mat_logits, eos_logits = self.model(continuous, material_ids)
            
            loss, loss_dict = self.loss_fn(
                cont_pred, continuous,
                mat_logits, material_ids,
                eos_logits, eos_target,
                mask=mask,
            )
            
            total_loss += loss_dict["total"]
            total_continuous += loss_dict["continuous"]
            total_material += loss_dict["material"]
            total_eos += loss_dict["eos"]
            num_batches += 1
        
        return {
            "loss": total_loss / num_batches,
            "continuous": total_continuous / num_batches,
            "material": total_material / num_batches,
            "eos": total_eos / num_batches,
        }
    
    def save_checkpoint(self, name: str = "checkpoint.pt"):
        """Save model checkpoint."""
        path = self.output_dir / name
        torch.save({
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "best_val_loss": self.best_val_loss,
        }, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        print(f"Loaded checkpoint from {path}")
    
    def train(self, num_epochs: int, save_interval: int = 10):
        """Run full training."""
        print(f"\nStarting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("=" * 60)
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch()
            self.train_history.append(train_metrics)
            
            # Validate
            val_metrics = self.validate()
            if val_metrics:
                self.val_history.append(val_metrics)
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            epoch_time = time.time() - epoch_start
            
            # Print progress
            val_str = f" | Val: {val_metrics['loss']:.4f}" if val_metrics else ""
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train: {train_metrics['loss']:.4f} (cont {train_metrics['continuous']:.4f}, "
                  f"mat {train_metrics['material']:.4f}, eos {train_metrics['eos']:.4f})"
                  f"{val_str} | Time: {epoch_time:.1f}s")
            
            # Save best model
            if val_metrics and val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.save_checkpoint("best_model.pt")
            
            # Periodic save
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(f"checkpoint_epoch{epoch+1}.pt")
        
        # Final save
        self.save_checkpoint("final_model.pt")
        
        # Save training history
        history_path = self.output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump({
                "train": self.train_history,
                "val": self.val_history,
            }, f, indent=2)
        
        print("=" * 60)
        print(f"Training complete! Best val loss: {self.best_val_loss:.4f}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train furniture transformer")
    parser.add_argument("--data", type=str, required=True, help="Training data directory")
    parser.add_argument("--output", type=str, default="models/furniture_transformer",
                        help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--d-model", type=int, default=256, help="Model hidden dimension")
    parser.add_argument("--n-layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--n-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 60)
    print("FURNITURE TRANSFORMER TRAINING")
    print("=" * 60)
    print(f"Data:       {args.data}")
    print(f"Output:     {args.output}")
    print(f"Device:     {device}")
    print(f"Epochs:     {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Model:      d_model={args.d_model}, layers={args.n_layers}, heads={args.n_heads}")
    print("=" * 60)
    
    # Load dataset
    full_dataset = FurnitureDataset(args.data)
    
    # Split into train/val
    n_val = int(len(full_dataset) * args.val_split)
    n_train = len(full_dataset) - n_val
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    ) if n_val > 0 else None
    
    # Create model
    config = FurnitureModelConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        learning_rate=args.lr,
    )
    
    model = FurnitureTransformer(config)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        output_dir=args.output,
        device=device,
    )
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train(num_epochs=args.epochs)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
