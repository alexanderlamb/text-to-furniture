"""
Transformer model for autoregressive furniture generation.

The model generates furniture as a sequence of components, where each component
is represented as a token containing:
- Position (x, y, z) - normalized to [0, 1]
- Dimensions (width, height, thickness) - normalized  
- Rotation (rx, ry, rz) - already in [-0.5, 0.5]
- Material (categorical)

Architecture:
- Input: sequence of component tokens + optional text embedding
- Output: next component prediction (continuous + categorical)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class FurnitureModelConfig:
    """Model configuration."""
    # Component representation
    num_materials: int = 15  # Number of material types
    max_components: int = 20  # Maximum components per design
    
    # Model architecture
    d_model: int = 256  # Hidden dimension
    n_heads: int = 8  # Attention heads
    n_layers: int = 6  # Transformer layers
    d_ff: int = 1024  # Feed-forward dimension
    dropout: float = 0.1
    
    # Input dimensions
    continuous_dim: int = 9  # x,y,z + w,h,t + rx,ry,rz
    
    # Text conditioning (optional)
    text_embed_dim: int = 512  # Dimension of text embeddings
    use_text: bool = False
    
    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 0.01


# Material vocabulary
MATERIAL_VOCAB = {
    "<pad>": 0,
    "<sos>": 1,  # Start of sequence
    "<eos>": 2,  # End of sequence
    "baltic_birch_plywood": 3,
    "mdf": 4,
    "mild_steel": 5,
    "aluminum_5052": 6,
    "stainless_steel_304": 7,
    "acrylic_clear": 8,
    "hdpe": 9,
    "polycarbonate": 10,
    "brass_260": 11,
    "copper_110": 12,
    "g10_fiberglass": 13,
    "delrin": 14,
}

VOCAB_TO_MATERIAL = {v: k for k, v in MATERIAL_VOCAB.items()}


# =============================================================================
# Positional Encoding
# =============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[:, :x.size(1)]


# =============================================================================
# Component Tokenizer
# =============================================================================

class ComponentTokenizer:
    """
    Convert furniture components to/from model tokens.
    
    Normalization ranges (based on typical furniture dimensions in mm):
    - Position: [0, 2000] -> [0, 1]
    - Dimensions: [10, 1000] -> [0, 1]
    - Rotation: [-0.5, 0.5] -> [0, 1]
    """
    
    POS_MAX = 2000.0
    DIM_MIN = 10.0
    DIM_MAX = 1000.0
    ROT_MIN = -0.5
    ROT_MAX = 0.5
    
    @classmethod
    def normalize_component(cls, component: Dict) -> Tuple[torch.Tensor, int]:
        """
        Convert component dict to normalized tensor + material index.
        
        Returns:
            continuous: Tensor of shape (9,) with normalized values
            material_idx: Integer material index
        """
        dims = component["dimensions"]
        pos = component["position"]
        rot = component["rotation"]
        
        # Normalize position
        x = pos["x"] / cls.POS_MAX
        y = pos["y"] / cls.POS_MAX
        z = pos["z"] / cls.POS_MAX
        
        # Normalize dimensions
        w = (dims["width"] - cls.DIM_MIN) / (cls.DIM_MAX - cls.DIM_MIN)
        h = (dims["height"] - cls.DIM_MIN) / (cls.DIM_MAX - cls.DIM_MIN)
        t = (dims["thickness"] - cls.DIM_MIN) / (cls.DIM_MAX - cls.DIM_MIN)
        
        # Normalize rotation
        rx = (rot["rx"] - cls.ROT_MIN) / (cls.ROT_MAX - cls.ROT_MIN)
        ry = (rot["ry"] - cls.ROT_MIN) / (cls.ROT_MAX - cls.ROT_MIN)
        rz = (rot["rz"] - cls.ROT_MIN) / (cls.ROT_MAX - cls.ROT_MIN)
        
        # Clamp to [0, 1]
        continuous = torch.tensor([x, y, z, w, h, t, rx, ry, rz], dtype=torch.float32)
        continuous = continuous.clamp(0, 1)
        
        # Material
        material = component.get("material", "baltic_birch_plywood")
        material_idx = MATERIAL_VOCAB.get(material, MATERIAL_VOCAB["baltic_birch_plywood"])
        
        return continuous, material_idx
    
    @classmethod
    def denormalize_component(cls, continuous: torch.Tensor, material_idx: int) -> Dict:
        """
        Convert normalized tensor + material index back to component dict.
        """
        continuous = continuous.clamp(0, 1)
        
        x, y, z, w, h, t, rx, ry, rz = continuous.tolist()
        
        return {
            "position": {
                "x": x * cls.POS_MAX,
                "y": y * cls.POS_MAX,
                "z": z * cls.POS_MAX,
            },
            "dimensions": {
                "width": w * (cls.DIM_MAX - cls.DIM_MIN) + cls.DIM_MIN,
                "height": h * (cls.DIM_MAX - cls.DIM_MIN) + cls.DIM_MIN,
                "thickness": t * (cls.DIM_MAX - cls.DIM_MIN) + cls.DIM_MIN,
            },
            "rotation": {
                "rx": rx * (cls.ROT_MAX - cls.ROT_MIN) + cls.ROT_MIN,
                "ry": ry * (cls.ROT_MAX - cls.ROT_MIN) + cls.ROT_MIN,
                "rz": rz * (cls.ROT_MAX - cls.ROT_MIN) + cls.ROT_MIN,
            },
            "material": VOCAB_TO_MATERIAL.get(material_idx, "baltic_birch_plywood"),
        }


# =============================================================================
# Model Components
# =============================================================================

class ComponentEmbedding(nn.Module):
    """
    Embed a furniture component into the model's hidden space.
    
    Combines:
    - Continuous features (position, dimensions, rotation) via MLP
    - Material embedding (learned)
    """
    
    def __init__(self, config: FurnitureModelConfig):
        super().__init__()
        self.config = config
        
        # Continuous feature projection
        self.continuous_proj = nn.Sequential(
            nn.Linear(config.continuous_dim, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        
        # Material embedding
        self.material_embed = nn.Embedding(
            len(MATERIAL_VOCAB), 
            config.d_model,
            padding_idx=MATERIAL_VOCAB["<pad>"]
        )
        
        # Combine continuous + material
        self.combine = nn.Linear(config.d_model * 2, config.d_model)
        
        # Layer norm
        self.norm = nn.LayerNorm(config.d_model)
    
    def forward(self, continuous: torch.Tensor, material_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            continuous: (batch, seq_len, 9) normalized continuous features
            material_ids: (batch, seq_len) material indices
        
        Returns:
            embeddings: (batch, seq_len, d_model)
        """
        cont_embed = self.continuous_proj(continuous)
        mat_embed = self.material_embed(material_ids)
        
        combined = torch.cat([cont_embed, mat_embed], dim=-1)
        embeddings = self.combine(combined)
        
        return self.norm(embeddings)


class ComponentHead(nn.Module):
    """
    Output head that predicts the next component.
    
    Outputs:
    - Continuous features (position, dimensions, rotation) - regression
    - Material - classification
    - End-of-sequence probability
    """
    
    def __init__(self, config: FurnitureModelConfig):
        super().__init__()
        self.config = config
        
        # Continuous output (regression)
        self.continuous_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.continuous_dim),
            nn.Sigmoid(),  # Output in [0, 1]
        )
        
        # Material output (classification)
        self.material_head = nn.Linear(config.d_model, len(MATERIAL_VOCAB))
        
        # EOS probability
        self.eos_head = nn.Linear(config.d_model, 1)
    
    def forward(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden: (batch, seq_len, d_model)
        
        Returns:
            continuous: (batch, seq_len, 9) predicted continuous features
            material_logits: (batch, seq_len, num_materials) material logits
            eos_logits: (batch, seq_len, 1) end-of-sequence logits
        """
        continuous = self.continuous_head(hidden)
        material_logits = self.material_head(hidden)
        eos_logits = self.eos_head(hidden)
        
        return continuous, material_logits, eos_logits


# =============================================================================
# Main Transformer Model
# =============================================================================

class FurnitureTransformer(nn.Module):
    """
    Autoregressive transformer for furniture generation.
    
    Given a sequence of components [c1, c2, ..., cn], predicts the next
    component cn+1. Uses causal masking so each position can only attend
    to previous positions.
    """
    
    def __init__(self, config: FurnitureModelConfig):
        super().__init__()
        self.config = config
        
        # Component embedding
        self.embedding = ComponentEmbedding(config)
        
        # Positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(config.d_model, config.max_components + 1)
        
        # Special token embeddings
        self.sos_embed = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)
        
        # Text conditioning (optional)
        if config.use_text:
            self.text_proj = nn.Linear(config.text_embed_dim, config.d_model)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=config.n_layers)
        
        # Output head
        self.output_head = ComponentHead(config)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.02)
    
    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(
        self,
        continuous: torch.Tensor,
        material_ids: torch.Tensor,
        text_embed: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            continuous: (batch, seq_len, 9) normalized continuous features
            material_ids: (batch, seq_len) material indices
            text_embed: (batch, text_embed_dim) optional text embedding
        
        Returns:
            continuous_pred: (batch, seq_len, 9) predicted continuous features
            material_logits: (batch, seq_len, num_materials)
            eos_logits: (batch, seq_len, 1)
        """
        batch_size, seq_len, _ = continuous.shape
        device = continuous.device
        
        # Embed components
        embeddings = self.embedding(continuous, material_ids)
        
        # Prepend SOS token
        sos = self.sos_embed.expand(batch_size, -1, -1)
        embeddings = torch.cat([sos, embeddings[:, :-1]], dim=1)  # Shift right
        
        # Add positional encoding
        embeddings = self.pos_encoding(embeddings)
        
        # Prepare memory (for cross-attention)
        if self.config.use_text and text_embed is not None:
            memory = self.text_proj(text_embed).unsqueeze(1)  # (batch, 1, d_model)
        else:
            # Use dummy memory if no text
            memory = torch.zeros(batch_size, 1, self.config.d_model, device=device)
        
        # Generate causal mask
        causal_mask = self._generate_causal_mask(seq_len, device)
        
        # Run transformer
        hidden = self.transformer(
            embeddings,
            memory,
            tgt_mask=causal_mask,
        )
        
        # Predict next component
        continuous_pred, material_logits, eos_logits = self.output_head(hidden)
        
        return continuous_pred, material_logits, eos_logits
    
    @torch.no_grad()
    def generate(
        self,
        max_components: int = 15,
        temperature: float = 1.0,
        text_embed: Optional[torch.Tensor] = None,
        device: torch.device = None,
    ) -> List[Dict]:
        """
        Generate a furniture design autoregressively.
        
        Args:
            max_components: Maximum number of components to generate
            temperature: Sampling temperature (higher = more random)
            text_embed: Optional text embedding for conditioning
            device: Device to generate on
        
        Returns:
            List of component dictionaries
        """
        if device is None:
            device = next(self.parameters()).device
        
        self.eval()
        
        # Start with empty sequence
        components = []
        
        # Initialize with zeros (will be replaced by SOS in forward)
        continuous = torch.zeros(1, 1, self.config.continuous_dim, device=device)
        material_ids = torch.full((1, 1), MATERIAL_VOCAB["<sos>"], dtype=torch.long, device=device)
        
        for step in range(max_components):
            # Forward pass
            cont_pred, mat_logits, eos_logits = self.forward(continuous, material_ids, text_embed)
            
            # Get prediction for last position
            cont_next = cont_pred[:, -1]  # (1, 9)
            mat_logits_next = mat_logits[:, -1]  # (1, num_materials)
            eos_next = torch.sigmoid(eos_logits[:, -1, 0])  # (1,)
            
            # Check for EOS
            if eos_next.item() > 0.5 and step > 2:  # At least 3 components
                break
            
            # Sample material
            mat_probs = F.softmax(mat_logits_next / temperature, dim=-1)
            # Mask special tokens
            mat_probs[:, :3] = 0
            mat_probs = mat_probs / mat_probs.sum(dim=-1, keepdim=True)
            mat_next = torch.multinomial(mat_probs, 1)  # (1, 1)
            
            # Add noise to continuous for diversity
            if temperature > 0:
                noise = torch.randn_like(cont_next) * 0.02 * temperature
                cont_next = (cont_next + noise).clamp(0, 1)
            
            # Convert to component dict
            component = ComponentTokenizer.denormalize_component(
                cont_next.squeeze(0),
                mat_next.item()
            )
            components.append(component)
            
            # Append to sequence for next iteration
            continuous = torch.cat([continuous, cont_next.unsqueeze(1)], dim=1)
            material_ids = torch.cat([material_ids, mat_next], dim=1)
        
        return components


# =============================================================================
# Loss Function
# =============================================================================

class FurnitureLoss(nn.Module):
    """
    Combined loss for furniture generation.
    
    - MSE loss for continuous features (position, dimensions, rotation)
    - Cross-entropy for material prediction
    - Binary cross-entropy for EOS prediction
    """
    
    def __init__(self, continuous_weight: float = 1.0, material_weight: float = 0.5, eos_weight: float = 0.3):
        super().__init__()
        self.continuous_weight = continuous_weight
        self.material_weight = material_weight
        self.eos_weight = eos_weight
        
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss(ignore_index=MATERIAL_VOCAB["<pad>"])
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(
        self,
        continuous_pred: torch.Tensor,
        continuous_target: torch.Tensor,
        material_logits: torch.Tensor,
        material_target: torch.Tensor,
        eos_logits: torch.Tensor,
        eos_target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.
        
        Args:
            continuous_pred: (batch, seq_len, 9)
            continuous_target: (batch, seq_len, 9)
            material_logits: (batch, seq_len, num_materials)
            material_target: (batch, seq_len)
            eos_logits: (batch, seq_len, 1)
            eos_target: (batch, seq_len)
            mask: (batch, seq_len) optional mask for padding
        
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual losses
        """
        if mask is not None:
            # Apply mask to continuous loss
            mask_expanded = mask.unsqueeze(-1)
            continuous_loss = (self.mse(continuous_pred, continuous_target) * mask_expanded).sum() / mask_expanded.sum()
        else:
            continuous_loss = self.mse(continuous_pred, continuous_target)
        
        # Material loss
        material_loss = self.ce(
            material_logits.reshape(-1, material_logits.size(-1)),
            material_target.reshape(-1)
        )
        
        # EOS loss
        eos_loss = self.bce(eos_logits.squeeze(-1), eos_target.float())
        
        # Combined
        total_loss = (
            self.continuous_weight * continuous_loss +
            self.material_weight * material_loss +
            self.eos_weight * eos_loss
        )
        
        return total_loss, {
            "continuous": continuous_loss.item(),
            "material": material_loss.item(),
            "eos": eos_loss.item(),
            "total": total_loss.item(),
        }


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    print("Testing FurnitureTransformer...")
    
    config = FurnitureModelConfig(
        d_model=128,
        n_heads=4,
        n_layers=3,
    )
    
    model = FurnitureTransformer(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 4
    seq_len = 5
    
    continuous = torch.rand(batch_size, seq_len, 9)
    material_ids = torch.randint(3, 15, (batch_size, seq_len))
    
    cont_pred, mat_logits, eos_logits = model(continuous, material_ids)
    
    print(f"Input: continuous {continuous.shape}, materials {material_ids.shape}")
    print(f"Output: continuous {cont_pred.shape}, materials {mat_logits.shape}, eos {eos_logits.shape}")
    
    # Test generation
    print("\nTesting generation...")
    components = model.generate(max_components=5, temperature=1.0)
    print(f"Generated {len(components)} components:")
    for i, comp in enumerate(components):
        print(f"  {i+1}. Material: {comp['material']}, "
              f"Dims: ({comp['dimensions']['width']:.0f}, {comp['dimensions']['height']:.0f}, {comp['dimensions']['thickness']:.0f})")
    
    # Test loss
    print("\nTesting loss...")
    loss_fn = FurnitureLoss()
    eos_target = torch.zeros(batch_size, seq_len)
    eos_target[:, -1] = 1  # Last position is EOS
    
    loss, loss_dict = loss_fn(cont_pred, continuous, mat_logits, material_ids, eos_logits, eos_target)
    print(f"Loss: {loss_dict}")
    
    print("\nAll tests passed!")
