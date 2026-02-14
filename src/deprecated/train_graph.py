"""
Training script for the graph-based furniture transformer.

Usage:
    python src/train_graph.py --data training_data_v2 --epochs 200 --batch-size 32
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
import numpy as np

from model_graph import (
    GraphFurnitureTransformer,
    GraphModelConfig,
    GraphFurnitureLoss,
    MATERIAL_VOCAB,
    FACE_VOCAB,
)


# =============================================================================
# Dataset
# =============================================================================

class GraphFurnitureDataset(Dataset):
    """
    Dataset for graph-based furniture generation.
    
    Each example contains:
    - Component properties (dimensions, rotation, material)
    - Connection graph (parent indices, faces, offsets)
    """
    
    def __init__(self, data_dir: str, max_components: int = 20, max_connections: int = 30):
        self.data_dir = Path(data_dir)
        self.max_components = max_components
        self.max_connections = max_connections
        
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
                    
                    # Only include designs with components and connections
                    if data.get("components") and data.get("connections"):
                        self.designs.append(data)
    
    def _normalize_continuous(self, comp: Dict) -> torch.Tensor:
        """Normalize component dimensions and rotation to [0, 1]."""
        dims = comp["dimensions"]
        rot = comp["rotation"]
        
        # Normalize dimensions (assume max 1000mm for w/h, 50mm for thickness)
        w = (dims["width"] - 10) / 1000.0
        h = (dims["height"] - 10) / 1000.0
        t = (dims["thickness"] - 1) / 50.0
        
        # Normalize rotation (already in [-0.5, 0.5], shift to [0, 1])
        rx = rot["rx"] + 0.5
        ry = rot["ry"] + 0.5
        rz = rot["rz"] + 0.5
        
        return torch.tensor([w, h, t, rx, ry, rz], dtype=torch.float32).clamp(0, 1)
    
    def _get_material_idx(self, material: str) -> int:
        """Convert material name to index."""
        # Handle material name variations
        material_map = {
            "plywood_baltic_birch": "plywood_baltic_birch",
            "baltic_birch_plywood": "plywood_baltic_birch",
            "mdf": "mdf",
            "mild_steel": "mild_steel",
            "aluminum_5052": "aluminum_5052",
            "stainless_304": "stainless_304",
            "stainless_steel_304": "stainless_304",
            "acrylic_clear": "acrylic_clear",
            "hdpe": "hdpe",
            "polycarbonate": "polycarbonate",
            "hardboard": "hardboard",
            "acrylic_black": "acrylic_black",
            "neoprene": "neoprene",
        }
        
        mapped = material_map.get(material, "plywood_baltic_birch")
        return MATERIAL_VOCAB.get(mapped, MATERIAL_VOCAB["plywood_baltic_birch"])
    
    def _get_face_idx(self, anchor: List[float]) -> int:
        """Infer face from anchor position."""
        # Anchor is [u, v, w] in [0, 1] space
        # w=0 means bottom face (z-), w=1 means top (z+)
        # Similar for u (x) and v (y)
        u, v, w = anchor
        
        if w < 0.1:
            return FACE_VOCAB["z-"]
        elif w > 0.9:
            return FACE_VOCAB["z+"]
        elif u < 0.1:
            return FACE_VOCAB["x-"]
        elif u > 0.9:
            return FACE_VOCAB["x+"]
        elif v < 0.1:
            return FACE_VOCAB["y-"]
        elif v > 0.9:
            return FACE_VOCAB["y+"]
        else:
            return FACE_VOCAB["z+"]  # Default to top
    
    def __len__(self) -> int:
        return len(self.designs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training example."""
        design = self.designs[idx]
        components = design["components"][:self.max_components]
        connections = design["connections"][:self.max_connections]
        
        num_components = len(components)
        num_connections = len(connections)
        
        # Initialize tensors
        continuous = torch.zeros(self.max_components, 6)
        material_ids = torch.zeros(self.max_components, dtype=torch.long)
        conn_pairs = torch.zeros(self.max_connections, 2, dtype=torch.long) - 1
        parent_ids = torch.zeros(self.max_components, dtype=torch.long)
        face_ids = torch.zeros(self.max_components, dtype=torch.long)
        offsets = torch.zeros(self.max_components, 2)
        eos_target = torch.zeros(self.max_components)
        mask = torch.zeros(self.max_components)
        
        # Build parent lookup from connections
        parent_map = {}  # child_idx -> (parent_idx, anchor_from, anchor_to)
        for conn in connections:
            child = conn["to"]
            parent = conn["from"]
            anchor_from = conn.get("anchor_from", [0.5, 0.5, 0.0])
            anchor_to = conn.get("anchor_to", [0.5, 0.5, 1.0])
            parent_map[child] = (parent, anchor_from, anchor_to)
        
        # Fill component data
        for i, comp in enumerate(components):
            continuous[i] = self._normalize_continuous(comp)
            material_ids[i] = self._get_material_idx(comp.get("material", "plywood"))
            mask[i] = 1.0
            
            # Get parent info
            if i in parent_map:
                parent_idx, anchor_from, anchor_to = parent_map[i]
                parent_ids[i] = parent_idx
                face_ids[i] = self._get_face_idx(anchor_to)
                offsets[i, 0] = anchor_to[0]  # u
                offsets[i, 1] = anchor_to[1]  # v
            else:
                # First component or no parent
                parent_ids[i] = 0
                face_ids[i] = FACE_VOCAB["z+"]
                offsets[i] = torch.tensor([0.5, 0.5])
        
        # Fill connection pairs
        for i, conn in enumerate(connections):
            if i < self.max_connections:
                conn_pairs[i, 0] = conn["from"]
                conn_pairs[i, 1] = conn["to"]
        
        # EOS at last component
        if num_components > 0:
            eos_target[num_components - 1] = 1.0
        
        return {
            "continuous": continuous,
            "material_ids": material_ids,
            "connections": conn_pairs,
            "parent_ids": parent_ids,
            "face_ids": face_ids,
            "offsets": offsets,
            "eos_target": eos_target,
            "mask": mask,
            "length": num_components,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate batch."""
    return {
        "continuous": torch.stack([b["continuous"] for b in batch]),
        "material_ids": torch.stack([b["material_ids"] for b in batch]),
        "connections": torch.stack([b["connections"] for b in batch]),
        "parent_ids": torch.stack([b["parent_ids"] for b in batch]),
        "face_ids": torch.stack([b["face_ids"] for b in batch]),
        "offsets": torch.stack([b["offsets"] for b in batch]),
        "eos_target": torch.stack([b["eos_target"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "lengths": torch.tensor([b["length"] for b in batch]),
    }


# =============================================================================
# Trainer
# =============================================================================

class GraphTrainer:
    """Training manager for graph furniture model."""
    
    def __init__(
        self,
        model: GraphFurnitureTransformer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: GraphModelConfig,
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
        
        self.loss_fn = GraphFurnitureLoss()
        
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        self.train_history = []
        self.val_history = []
    
    def train_epoch(self) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        
        total_losses = {
            "total": 0, "continuous": 0, "material": 0,
            "parent": 0, "face": 0, "offset": 0, "eos": 0
        }
        num_batches = 0
        
        for batch in self.train_loader:
            # Move to device
            continuous = batch["continuous"].to(self.device)
            material_ids = batch["material_ids"].to(self.device)
            connections = batch["connections"].to(self.device)
            parent_ids = batch["parent_ids"].to(self.device)
            face_ids = batch["face_ids"].to(self.device)
            offsets = batch["offsets"].to(self.device)
            eos_target = batch["eos_target"].to(self.device)
            mask = batch["mask"].to(self.device)
            
            # Forward
            outputs = self.model(
                continuous, material_ids, connections,
                parent_ids, face_ids, offsets, mask
            )
            
            # Compute loss
            targets = {
                "continuous": continuous,
                "material_ids": material_ids,
                "parent_ids": parent_ids,
                "face_ids": face_ids,
                "offsets": offsets,
                "eos_target": eos_target,
            }
            
            loss, loss_dict = self.loss_fn(outputs, targets, mask)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Track
            for k, v in loss_dict.items():
                total_losses[k] += v
            num_batches += 1
            self.global_step += 1
        
        return {k: v / num_batches for k, v in total_losses.items()}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        total_losses = {
            "total": 0, "continuous": 0, "material": 0,
            "parent": 0, "face": 0, "offset": 0, "eos": 0
        }
        num_batches = 0
        
        for batch in self.val_loader:
            continuous = batch["continuous"].to(self.device)
            material_ids = batch["material_ids"].to(self.device)
            connections = batch["connections"].to(self.device)
            parent_ids = batch["parent_ids"].to(self.device)
            face_ids = batch["face_ids"].to(self.device)
            offsets = batch["offsets"].to(self.device)
            eos_target = batch["eos_target"].to(self.device)
            mask = batch["mask"].to(self.device)
            
            outputs = self.model(
                continuous, material_ids, connections,
                parent_ids, face_ids, offsets, mask
            )
            
            targets = {
                "continuous": continuous,
                "material_ids": material_ids,
                "parent_ids": parent_ids,
                "face_ids": face_ids,
                "offsets": offsets,
                "eos_target": eos_target,
            }
            
            _, loss_dict = self.loss_fn(outputs, targets, mask)
            
            for k, v in loss_dict.items():
                total_losses[k] += v
            num_batches += 1
        
        return {k: v / num_batches for k, v in total_losses.items()}
    
    def save_checkpoint(self, name: str = "checkpoint.pt"):
        """Save checkpoint."""
        path = self.output_dir / name
        torch.save({
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "best_val_loss": self.best_val_loss,
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        print(f"Loaded checkpoint from {path} (epoch {self.epoch})")
    
    def train(self, num_epochs: int, save_interval: int = 10):
        """Run full training."""
        print(f"\nStarting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("=" * 70)
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            epoch_start = time.time()
            
            train_metrics = self.train_epoch()
            self.train_history.append(train_metrics)
            
            val_metrics = self.validate()
            if val_metrics:
                self.val_history.append(val_metrics)
            
            epoch_time = time.time() - epoch_start
            
            # Print progress
            val_str = f" | Val: {val_metrics['total']:.4f}" if val_metrics else ""
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train: {train_metrics['total']:.4f} "
                  f"(cont {train_metrics['continuous']:.3f}, "
                  f"mat {train_metrics['material']:.3f}, "
                  f"parent {train_metrics['parent']:.3f})"
                  f"{val_str} | {epoch_time:.1f}s")
            
            # Save best
            if val_metrics and val_metrics["total"] < self.best_val_loss:
                self.best_val_loss = val_metrics["total"]
                self.save_checkpoint("best_model.pt")
            
            # Periodic save
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(f"checkpoint_epoch{epoch+1}.pt")
        
        self.save_checkpoint("final_model.pt")
        
        # Save history
        with open(self.output_dir / "training_history.json", 'w') as f:
            json.dump({"train": self.train_history, "val": self.val_history}, f, indent=2)
        
        print("=" * 70)
        print(f"Training complete! Best val loss: {self.best_val_loss:.4f}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train graph furniture model")
    parser.add_argument("--data", type=str, required=True, help="Training data directory")
    parser.add_argument("--output", type=str, default="models/graph_transformer",
                        help="Output directory")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--d-model", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--n-layers", type=int, default=6, help="Transformer layers")
    parser.add_argument("--n-graph-layers", type=int, default=3, help="Graph attention layers")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 70)
    print("GRAPH FURNITURE TRANSFORMER TRAINING")
    print("=" * 70)
    print(f"Data:       {args.data}")
    print(f"Output:     {args.output}")
    print(f"Device:     {device}")
    print(f"Model:      d_model={args.d_model}, layers={args.n_layers}, graph_layers={args.n_graph_layers}")
    print("=" * 70)
    
    # Load data
    full_dataset = GraphFurnitureDataset(args.data)
    
    n_val = int(len(full_dataset) * args.val_split)
    n_train = len(full_dataset) - n_val
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
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
    config = GraphModelConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_graph_layers=args.n_graph_layers,
        learning_rate=args.lr,
    )
    
    model = GraphFurnitureTransformer(config)
    
    # Create trainer
    trainer = GraphTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        output_dir=args.output,
        device=device,
    )
    
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    trainer.train(num_epochs=args.epochs)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
