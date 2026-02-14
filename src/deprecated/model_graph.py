"""
Graph-based transformer for furniture generation with explicit connections.

Key insight: Instead of predicting absolute positions, we predict:
1. Component properties (dimensions, material)
2. Which existing component to attach to (parent index)
3. Attachment parameters (face, offset)

This makes the model learn the STRUCTURE of furniture, not just geometry.
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
class GraphModelConfig:
    """Model configuration."""
    # Component representation  
    num_materials: int = 15
    max_components: int = 20
    
    # Model architecture
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1
    
    # Graph message passing
    n_graph_layers: int = 3
    
    # Component features
    # Dimensions (w, h, t) + rotation (rx, ry, rz) = 6
    continuous_dim: int = 6
    
    # Attachment prediction
    num_faces: int = 6  # 6 faces of a box
    
    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 0.01


# Material vocabulary (same as before)
MATERIAL_VOCAB = {
    "<pad>": 0,
    "<sos>": 1,
    "<eos>": 2,
    "plywood_baltic_birch": 3,
    "mdf": 4,
    "mild_steel": 5,
    "aluminum_5052": 6,
    "stainless_304": 7,
    "acrylic_clear": 8,
    "hdpe": 9,
    "polycarbonate": 10,
    "hardboard": 11,
    "acrylic_black": 12,
    "neoprene": 13,
}

VOCAB_TO_MATERIAL = {v: k for k, v in MATERIAL_VOCAB.items()}

# Face vocabulary for attachment
FACE_VOCAB = {
    "x-": 0,  # Left
    "x+": 1,  # Right
    "y-": 2,  # Front
    "y+": 3,  # Back
    "z-": 4,  # Bottom
    "z+": 5,  # Top
}


# =============================================================================
# Graph Neural Network Layers
# =============================================================================

class GraphAttention(nn.Module):
    """
    Graph attention layer for message passing between connected components.
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self, 
        x: torch.Tensor,  # (batch, num_nodes, d_model)
        adj: torch.Tensor,  # (batch, num_nodes, num_nodes) adjacency matrix
    ) -> torch.Tensor:
        batch_size, num_nodes, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, num_nodes, self.n_heads, self.d_head)
        k = self.k_proj(x).view(batch_size, num_nodes, self.n_heads, self.d_head)
        v = self.v_proj(x).view(batch_size, num_nodes, self.n_heads, self.d_head)
        
        # Attention scores
        # (batch, heads, nodes, nodes)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        # Mask with adjacency (only attend to connected nodes + self)
        # adj: (batch, nodes, nodes) -> (batch, 1, nodes, nodes)
        adj_mask = adj.unsqueeze(1)
        # Add self-connections
        eye = torch.eye(num_nodes, device=x.device).unsqueeze(0).unsqueeze(0)
        adj_mask = adj_mask + eye
        adj_mask = (adj_mask > 0).float()
        
        # Apply mask
        scores = scores.masked_fill(adj_mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, num_nodes, self.d_model)
        out = self.out_proj(out)
        
        # Residual + norm
        return self.norm(x + out)


class GraphTransformerLayer(nn.Module):
    """Combined graph attention + feedforward."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.graph_attn = GraphAttention(d_model, n_heads, dropout)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = self.graph_attn(x, adj)
        x = self.norm(x + self.ff(x))
        return x


# =============================================================================
# Component Embedding & Prediction Heads
# =============================================================================

class ComponentEmbedding(nn.Module):
    """Embed component properties into hidden space."""
    
    def __init__(self, config: GraphModelConfig):
        super().__init__()
        self.config = config
        
        # Continuous features (dims + rotation)
        self.continuous_proj = nn.Sequential(
            nn.Linear(config.continuous_dim, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        
        # Material embedding
        self.material_embed = nn.Embedding(
            len(MATERIAL_VOCAB),
            config.d_model,
            padding_idx=0
        )
        
        # Combine
        self.combine = nn.Linear(config.d_model * 2, config.d_model)
        self.norm = nn.LayerNorm(config.d_model)
    
    def forward(
        self,
        continuous: torch.Tensor,  # (batch, seq, 6)
        material_ids: torch.Tensor,  # (batch, seq)
    ) -> torch.Tensor:
        cont_emb = self.continuous_proj(continuous)
        mat_emb = self.material_embed(material_ids)
        combined = torch.cat([cont_emb, mat_emb], dim=-1)
        return self.norm(self.combine(combined))


class ComponentHead(nn.Module):
    """
    Predict next component properties.
    
    Outputs:
    - Dimensions (w, h, t) - continuous
    - Rotation (rx, ry, rz) - continuous  
    - Material - categorical
    - Parent index - which component to attach to
    - Attachment face - which face of parent
    - Attachment offset (u, v) - where on that face
    - EOS - whether to stop
    """
    
    def __init__(self, config: GraphModelConfig):
        super().__init__()
        self.config = config
        
        # Continuous outputs (dims + rotation)
        self.continuous_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.continuous_dim),
            nn.Sigmoid(),
        )
        
        # Material prediction
        self.material_head = nn.Linear(config.d_model, len(MATERIAL_VOCAB))
        
        # Parent index prediction (pointer network style)
        self.parent_query = nn.Linear(config.d_model, config.d_model)
        self.parent_key = nn.Linear(config.d_model, config.d_model)
        
        # Attachment face
        self.face_head = nn.Linear(config.d_model, config.num_faces)
        
        # Attachment offset (u, v in [0,1])
        self.offset_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, 2),
            nn.Sigmoid(),
        )
        
        # EOS
        self.eos_head = nn.Linear(config.d_model, 1)
    
    def forward(
        self,
        hidden: torch.Tensor,  # (batch, d_model) - representation for next component
        graph_hidden: torch.Tensor,  # (batch, num_existing, d_model) - existing components
        mask: torch.Tensor,  # (batch, num_existing) - which positions are valid
    ) -> Dict[str, torch.Tensor]:
        
        # Continuous properties
        continuous = self.continuous_head(hidden)
        
        # Material
        material_logits = self.material_head(hidden)
        
        # Parent selection (attention over existing components)
        query = self.parent_query(hidden).unsqueeze(1)  # (batch, 1, d_model)
        keys = self.parent_key(graph_hidden)  # (batch, num_existing, d_model)
        
        parent_scores = torch.matmul(query, keys.transpose(-2, -1)).squeeze(1)  # (batch, num_existing)
        parent_scores = parent_scores / math.sqrt(self.config.d_model)
        
        # Mask invalid positions
        parent_scores = parent_scores.masked_fill(mask == 0, float('-inf'))
        parent_logits = parent_scores
        
        # Attachment face
        face_logits = self.face_head(hidden)
        
        # Attachment offset
        offset = self.offset_head(hidden)
        
        # EOS
        eos_logits = self.eos_head(hidden)
        
        return {
            "continuous": continuous,
            "material_logits": material_logits,
            "parent_logits": parent_logits,
            "face_logits": face_logits,
            "offset": offset,
            "eos_logits": eos_logits,
        }


# =============================================================================
# Main Model
# =============================================================================

class GraphFurnitureTransformer(nn.Module):
    """
    Graph-based autoregressive model for furniture generation.
    
    Generation process:
    1. Start with a single "base" component
    2. For each new component:
       a. Run graph attention over existing components
       b. Predict new component properties
       c. Predict which existing component to attach to
       d. Predict attachment face and offset
       e. Add new component to graph
    3. Stop when EOS is predicted
    """
    
    def __init__(self, config: GraphModelConfig):
        super().__init__()
        self.config = config
        
        # Component embedding
        self.embedding = ComponentEmbedding(config)
        
        # Position encoding (for sequence order)
        self.pos_embed = nn.Embedding(config.max_components, config.d_model)
        
        # Graph transformer layers
        self.graph_layers = nn.ModuleList([
            GraphTransformerLayer(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_graph_layers)
        ])
        
        # Autoregressive transformer for predicting next component
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
        )
        self.ar_decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.n_layers)
        
        # Output head
        self.output_head = ComponentHead(config)
        
        # Learnable start token
        self.start_token = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.02)
    
    def _build_adjacency(
        self,
        connections: torch.Tensor,  # (batch, max_conn, 2) - pairs of (from, to)
        num_nodes: int,
    ) -> torch.Tensor:
        """Build adjacency matrix from connection list."""
        batch_size = connections.shape[0]
        adj = torch.zeros(batch_size, num_nodes, num_nodes, device=connections.device)
        
        for b in range(batch_size):
            for c in range(connections.shape[1]):
                i, j = connections[b, c, 0].item(), connections[b, c, 1].item()
                if i >= 0 and j >= 0 and i < num_nodes and j < num_nodes:
                    adj[b, i, j] = 1
                    adj[b, j, i] = 1  # Undirected
        
        return adj
    
    def encode_graph(
        self,
        continuous: torch.Tensor,  # (batch, num_nodes, 6)
        material_ids: torch.Tensor,  # (batch, num_nodes)
        connections: torch.Tensor,  # (batch, max_conn, 2)
        mask: torch.Tensor,  # (batch, num_nodes)
    ) -> torch.Tensor:
        """Encode existing furniture graph."""
        batch_size, num_nodes, _ = continuous.shape
        
        # Embed components
        x = self.embedding(continuous, material_ids)
        
        # Add position embeddings
        positions = torch.arange(num_nodes, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.pos_embed(positions)
        
        # Build adjacency
        adj = self._build_adjacency(connections, num_nodes)
        
        # Apply mask to adjacency
        mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)
        adj = adj * mask_2d
        
        # Graph attention layers
        for layer in self.graph_layers:
            x = layer(x, adj)
        
        return x
    
    def forward(
        self,
        continuous: torch.Tensor,  # (batch, seq_len, 6)
        material_ids: torch.Tensor,  # (batch, seq_len)
        connections: torch.Tensor,  # (batch, max_conn, 2)
        parent_ids: torch.Tensor,  # (batch, seq_len) - which component each attaches to
        face_ids: torch.Tensor,  # (batch, seq_len) - attachment face
        offsets: torch.Tensor,  # (batch, seq_len, 2) - attachment offset
        mask: torch.Tensor,  # (batch, seq_len)
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass.
        
        For each position i, predict component i given components 0..i-1.
        """
        batch_size, seq_len, _ = continuous.shape
        device = continuous.device
        
        # Encode the full graph (teacher forcing - use ground truth)
        graph_hidden = self.encode_graph(continuous, material_ids, connections, mask)
        
        # For autoregressive prediction, we need to predict each position
        # given only the previous positions
        
        # Prepare shifted inputs for AR decoding
        # Start token + components[:-1]
        start = self.start_token.expand(batch_size, -1, -1)
        shifted_hidden = torch.cat([start, graph_hidden[:, :-1]], dim=1)
        
        # Causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        causal_mask = causal_mask.masked_fill(causal_mask == 1, float('-inf'))
        
        # AR decode
        # Use graph_hidden as memory (cross-attention to see structure)
        ar_out = self.ar_decoder(
            shifted_hidden,
            graph_hidden,
            tgt_mask=causal_mask,
        )
        
        # Predict at each position
        all_outputs = {
            "continuous": [],
            "material_logits": [],
            "parent_logits": [],
            "face_logits": [],
            "offset": [],
            "eos_logits": [],
        }
        
        for i in range(seq_len):
            # Get hidden state for position i
            hidden_i = ar_out[:, i]
            
            # Get graph context up to position i
            # (for parent selection, can only point to 0..i-1)
            if i == 0:
                # First component has no parent
                graph_context = torch.zeros(batch_size, 1, self.config.d_model, device=device)
                context_mask = torch.zeros(batch_size, 1, device=device)
            else:
                graph_context = graph_hidden[:, :i]
                context_mask = mask[:, :i]
            
            # Predict
            outputs = self.output_head(hidden_i, graph_context, context_mask)
            
            for k, v in outputs.items():
                all_outputs[k].append(v)
        
        # Stack outputs
        result = {}
        for k, v_list in all_outputs.items():
            if k == "parent_logits":
                # Parent logits have variable size, need padding
                max_len = max(v.shape[-1] for v in v_list)
                padded = []
                for v in v_list:
                    if v.shape[-1] < max_len:
                        pad = torch.full((batch_size, max_len - v.shape[-1]), float('-inf'), device=device)
                        v = torch.cat([v, pad], dim=-1)
                    padded.append(v)
                result[k] = torch.stack(padded, dim=1)
            else:
                result[k] = torch.stack(v_list, dim=1)
        
        return result
    
    @torch.no_grad()
    def generate(
        self,
        max_components: int = 15,
        temperature: float = 1.0,
        device: torch.device = None,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Generate a furniture design autoregressively.
        
        Returns:
            components: List of component dicts
            connections: List of connection dicts
        """
        if device is None:
            device = next(self.parameters()).device
        
        self.eval()
        
        components = []
        connections = []
        
        # Tensors to track state
        continuous_list = []
        material_list = []
        
        for step in range(max_components):
            # Build current state tensors
            if step == 0:
                # Empty graph - use placeholder
                curr_continuous = torch.zeros(1, 1, self.config.continuous_dim, device=device)
                curr_materials = torch.zeros(1, 1, dtype=torch.long, device=device)
                curr_connections = torch.zeros(1, 1, 2, dtype=torch.long, device=device) - 1
                curr_mask = torch.zeros(1, 1, device=device)
            else:
                # Stack existing components: list of (6,) tensors -> (1, step, 6)
                curr_continuous = torch.stack(continuous_list, dim=0).unsqueeze(0)  # (1, step, 6)
                curr_materials = torch.tensor([material_list], dtype=torch.long, device=device)  # (1, step)
                
                # Build connection tensor
                conn_pairs = [(c["from"], c["to"]) for c in connections]
                if conn_pairs:
                    curr_connections = torch.tensor([conn_pairs], dtype=torch.long, device=device)
                else:
                    curr_connections = torch.zeros(1, 1, 2, dtype=torch.long, device=device) - 1
                
                curr_mask = torch.ones(1, step, device=device)
            
            # Encode existing graph
            if step > 0:
                graph_hidden = self.encode_graph(
                    curr_continuous, curr_materials, curr_connections, curr_mask
                )
                # Use last layer output + mean pool as context
                context = graph_hidden.mean(dim=1)
            else:
                context = self.start_token.squeeze(0).squeeze(0)
            
            # Predict next component
            hidden = context.unsqueeze(0) if context.dim() == 1 else context
            
            if step > 0:
                graph_context = graph_hidden
                context_mask = curr_mask
            else:
                graph_context = torch.zeros(1, 1, self.config.d_model, device=device)
                context_mask = torch.zeros(1, 1, device=device)
            
            outputs = self.output_head(hidden.squeeze(1) if hidden.dim() == 3 else hidden, 
                                       graph_context, context_mask)
            
            # Check EOS
            eos_prob = torch.sigmoid(outputs["eos_logits"]).item()
            if eos_prob > 0.5 and step >= 3:
                break
            
            # Sample continuous (add noise for diversity)
            cont = outputs["continuous"]
            if temperature > 0:
                cont = cont + torch.randn_like(cont) * 0.02 * temperature
            cont = cont.clamp(0, 1).squeeze(0)
            
            # Sample material
            mat_probs = F.softmax(outputs["material_logits"] / temperature, dim=-1)
            mat_probs[:, :3] = 0  # Mask special tokens
            mat_probs = mat_probs / mat_probs.sum(dim=-1, keepdim=True)
            mat_idx = torch.multinomial(mat_probs.squeeze(0), 1).item()
            
            # Sample parent (if not first)
            if step > 0:
                parent_probs = F.softmax(outputs["parent_logits"] / temperature, dim=-1)
                parent_idx = torch.multinomial(parent_probs.squeeze(0), 1).item()
            else:
                parent_idx = -1
            
            # Sample face
            face_probs = F.softmax(outputs["face_logits"] / temperature, dim=-1)
            face_idx = torch.multinomial(face_probs.squeeze(0), 1).item()
            
            # Get offset
            offset = outputs["offset"].squeeze(0).tolist()
            
            # Store component
            component = {
                "dimensions": {
                    "width": cont[0].item() * 1000 + 10,  # Denormalize
                    "height": cont[1].item() * 1000 + 10,
                    "thickness": cont[2].item() * 50 + 1,
                },
                "rotation": {
                    "rx": (cont[3].item() - 0.5),
                    "ry": (cont[4].item() - 0.5),
                    "rz": (cont[5].item() - 0.5),
                },
                "material": VOCAB_TO_MATERIAL.get(mat_idx, "plywood_baltic_birch"),
            }
            components.append(component)
            
            # Store connection
            if parent_idx >= 0:
                connections.append({
                    "from": parent_idx,
                    "to": step,
                    "face": list(FACE_VOCAB.keys())[face_idx],
                    "offset": offset,
                })
            
            # Update state
            continuous_list.append(cont)
            material_list.append(mat_idx)
        
        return components, connections


# =============================================================================
# Loss Function
# =============================================================================

class GraphFurnitureLoss(nn.Module):
    """Loss for graph-based furniture generation."""
    
    def __init__(
        self,
        continuous_weight: float = 1.0,
        material_weight: float = 0.5,
        parent_weight: float = 1.0,
        face_weight: float = 0.5,
        offset_weight: float = 0.3,
        eos_weight: float = 0.3,
    ):
        super().__init__()
        self.weights = {
            "continuous": continuous_weight,
            "material": material_weight,
            "parent": parent_weight,
            "face": face_weight,
            "offset": offset_weight,
            "eos": eos_weight,
        }
        
        self.mse = nn.MSELoss(reduction='none')
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss.
        
        predictions: Dict from model forward
        targets: Dict with same keys as targets
        mask: (batch, seq_len) valid positions
        """
        losses = {}
        
        # Continuous loss
        cont_loss = self.mse(predictions["continuous"], targets["continuous"])
        cont_loss = (cont_loss.mean(dim=-1) * mask).sum() / mask.sum()
        losses["continuous"] = cont_loss
        
        # Material loss
        mat_loss = self.ce(
            predictions["material_logits"].reshape(-1, predictions["material_logits"].shape[-1]),
            targets["material_ids"].reshape(-1)
        )
        mat_loss = (mat_loss.view(mask.shape) * mask).sum() / mask.sum()
        losses["material"] = mat_loss
        
        # Parent loss (skip first position which has no parent)
        parent_mask = mask.clone()
        parent_mask[:, 0] = 0
        
        if parent_mask.sum() > 0:
            # Clamp parent targets to valid range
            max_parent = predictions["parent_logits"].shape[-1]
            if max_parent > 0:
                parent_targets = targets["parent_ids"].clamp(0, max_parent - 1)
                
                parent_loss = self.ce(
                    predictions["parent_logits"].reshape(-1, max_parent),
                    parent_targets.reshape(-1)
                )
                # Handle NaN/Inf in loss
                parent_loss = torch.where(
                    torch.isfinite(parent_loss),
                    parent_loss,
                    torch.zeros_like(parent_loss)
                )
                parent_loss = (parent_loss.view(mask.shape) * parent_mask).sum() / (parent_mask.sum() + 1e-8)
            else:
                parent_loss = torch.tensor(0.0, device=mask.device)
        else:
            parent_loss = torch.tensor(0.0, device=mask.device)
        losses["parent"] = parent_loss
        
        # Face loss
        face_loss = self.ce(
            predictions["face_logits"].reshape(-1, predictions["face_logits"].shape[-1]),
            targets["face_ids"].reshape(-1)
        )
        face_loss = (face_loss.view(mask.shape) * mask).sum() / mask.sum()
        losses["face"] = face_loss
        
        # Offset loss
        offset_loss = self.mse(predictions["offset"], targets["offsets"])
        offset_loss = (offset_loss.mean(dim=-1) * mask).sum() / mask.sum()
        losses["offset"] = offset_loss
        
        # EOS loss
        eos_loss = self.bce(predictions["eos_logits"].squeeze(-1), targets["eos_target"])
        eos_loss = (eos_loss * mask).sum() / mask.sum()
        losses["eos"] = eos_loss
        
        # Total
        total = sum(self.weights[k] * v for k, v in losses.items())
        
        loss_dict = {k: v.item() for k, v in losses.items()}
        loss_dict["total"] = total.item()
        
        return total, loss_dict


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing GraphFurnitureTransformer...")
    
    config = GraphModelConfig(d_model=128, n_heads=4, n_layers=3, n_graph_layers=2)
    model = GraphFurnitureTransformer(config)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward
    batch_size = 2
    seq_len = 5
    
    continuous = torch.rand(batch_size, seq_len, 6)
    material_ids = torch.randint(3, 12, (batch_size, seq_len))
    connections = torch.tensor([
        [[0, 1], [1, 2], [2, 3], [3, 4], [-1, -1]],
        [[0, 1], [0, 2], [1, 3], [2, 4], [-1, -1]],
    ])
    parent_ids = torch.tensor([[0, 0, 1, 2, 3], [0, 0, 0, 1, 2]])
    face_ids = torch.randint(0, 6, (batch_size, seq_len))
    offsets = torch.rand(batch_size, seq_len, 2)
    mask = torch.ones(batch_size, seq_len)
    
    outputs = model(continuous, material_ids, connections, parent_ids, face_ids, offsets, mask)
    
    print(f"Continuous: {outputs['continuous'].shape}")
    print(f"Material: {outputs['material_logits'].shape}")
    print(f"Parent: {outputs['parent_logits'].shape}")
    print(f"Face: {outputs['face_logits'].shape}")
    print(f"Offset: {outputs['offset'].shape}")
    print(f"EOS: {outputs['eos_logits'].shape}")
    
    # Test generation
    print("\nTesting generation...")
    components, connections = model.generate(max_components=5, temperature=1.0)
    print(f"Generated {len(components)} components, {len(connections)} connections")
    
    for i, comp in enumerate(components):
        print(f"  {i}: {comp['material']}, dims={comp['dimensions']['width']:.0f}x{comp['dimensions']['height']:.0f}")
    for conn in connections:
        print(f"  Connection: {conn['from']} -> {conn['to']} on face {conn['face']}")
    
    print("\nAll tests passed!")
