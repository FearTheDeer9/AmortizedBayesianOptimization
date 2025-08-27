"""
Complete script to generate thesis-quality architecture diagrams
Combines model implementation with diagram generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
import os

# ============= MODEL IMPLEMENTATION =============

class NodeFeatureEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, num_layers=8, num_heads=8, dropout_rate=0.1, widening_factor=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers * 2
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList()
        for i in range(self.num_layers):
            block = nn.ModuleDict({
                'ln_attn': nn.LayerNorm(hidden_dim),
                'mha': nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout_rate, batch_first=True),
                'ln_ffn': nn.LayerNorm(hidden_dim),
                'ffn': nn.Sequential(
                    nn.Linear(hidden_dim, widening_factor * hidden_dim),
                    nn.ReLU(),
                    nn.Linear(widening_factor * hidden_dim, hidden_dim),
                ),
                'dropout': nn.Dropout(dropout_rate)
            })
            self.blocks.append(block)
        self.final_ln = nn.LayerNorm(hidden_dim)
    
    def forward(self, data):
        if data.dim() == 3:
            data = data.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        batch_size, N, d, channels = data.shape
        z = self.input_projection(data)
        for i, block in enumerate(self.blocks):
            if i % 2 == 0:
                z_reshaped = z.transpose(1, 2).reshape(batch_size * d, N, self.hidden_dim)
            else:
                z_reshaped = z.reshape(batch_size * N, d, self.hidden_dim)
            z_norm = block['ln_attn'](z_reshaped)
            z_attn, _ = block['mha'](z_norm, z_norm, z_norm)
            z_reshaped = z_reshaped + block['dropout'](z_attn)
            z_norm = block['ln_ffn'](z_reshaped)
            z_ffn = block['ffn'](z_norm)
            z_reshaped = z_reshaped + block['dropout'](z_ffn)
            if i % 2 == 0:
                z = z_reshaped.reshape(batch_size, d, N, self.hidden_dim).transpose(1, 2)
            else:
                z = z_reshaped.reshape(batch_size, N, d, self.hidden_dim)
            z = z.transpose(1, 2)
        z = self.final_ln(z)
        if self.num_layers % 2 == 0:
            embeddings = z.max(dim=1)[0]
        else:
            embeddings = z.max(dim=2)[0]
        if squeeze_output:
            embeddings = embeddings.squeeze(0)
        return embeddings

class ParentAttentionLayer(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.score_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, target_embedding, node_embeddings, target_idx):
        if node_embeddings.dim() == 2:
            node_embeddings = node_embeddings.unsqueeze(0)
            target_embedding = target_embedding.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        batch_size, n_vars, hidden_dim = node_embeddings.shape
        target_expanded = target_embedding.unsqueeze(1).expand(-1, n_vars, -1)
        combined = torch.cat([target_expanded, node_embeddings], dim=-1)
        scores = self.score_network(combined).squeeze(-1)
        if squeeze_output:
            scores = scores.squeeze(0)
        return scores

class ContinuousParentSetPredictionModel(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=4, num_heads=8, dropout=0.1):
        super().__init__()
        self.encoder = NodeFeatureEncoder(input_dim=3, hidden_dim=hidden_dim, num_layers=num_layers, 
                                         num_heads=num_heads, dropout_rate=dropout)
        self.parent_attention = ParentAttentionLayer(hidden_dim=hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, data, target_variable):
        node_embeddings = self.encoder(data)
        if self.training:
            node_embeddings = self.dropout(node_embeddings)
        target_embedding = node_embeddings[target_variable]
        parent_logits = self.parent_attention(target_embedding, node_embeddings, target_variable)
        masked_logits = parent_logits.clone()
        masked_logits[target_variable] = -1e9
        parent_probs = torch.sigmoid(masked_logits)
        return {
            'node_embeddings': node_embeddings,
            'target_embedding': target_embedding,
            'attention_logits': parent_logits,
            'parent_probabilities': parent_probs
        }

# ============= DIAGRAM GENERATION =============

def generate_all_diagrams():
    print("\nGenerating thesis-quality architecture diagrams...")
    
    # Create model
    model = ContinuousParentSetPredictionModel(hidden_dim=128, num_layers=4, num_heads=8, dropout=0.1)
    model.eval()
    
    # Create dummy input
    N, d = 32, 10
    input_data = torch.randn(N, d, 3)
    target_var = 5
    
    os.makedirs('thesis_diagrams', exist_ok=True)
    
    # 1. Generate with torchview (best quality)
    try:
        from torchview import draw_graph
        print("✓ Generating torchview diagram...")
        model_graph = draw_graph(
            model, 
            input_data=(input_data, target_var),
            expand_nested=True,
            depth=3,
            roll=True,
            show_shapes=True,
            save_graph=True,
            filename='thesis_diagrams/architecture',
            directory='thesis_diagrams'
        )
        print("  Saved: thesis_diagrams/architecture.png")
    except Exception as e:
        print(f"  Skipped torchview: {e}")
    
    # 2. Generate custom Graphviz diagram
    try:
        from graphviz import Digraph
        print("✓ Generating custom Graphviz diagram...")
        
        dot = Digraph(comment='Model Architecture', engine='dot')
        dot.attr(rankdir='TB', bgcolor='white', fontname='Arial')
        dot.attr('node', fontname='Arial', fontsize='10')
        
        # Professional color scheme
        dot.node('A', 'Input Data\\n[N, d, 3]', shape='box3d', style='filled', fillcolor='#2E86AB')
        dot.node('B', 'Linear Projection\\n3 → 128', shape='box', style='filled,rounded', fillcolor='#A23B72')
        dot.node('C', 'Alternating Attention\\n16 Blocks', shape='box', style='filled,rounded', fillcolor='#F18F01')
        dot.node('D', 'Max Pool', shape='box', style='filled,rounded', fillcolor='#C73E1D')
        dot.node('E', 'Node Embeddings\\n[d, 128]', shape='box', style='filled,rounded', fillcolor='#6C969D')
        dot.node('F', 'Target Selection\\n[128]', shape='box', style='filled,rounded', fillcolor='#5C946E')
        dot.node('G', 'Parent Attention\\nScore Network', shape='box', style='filled,rounded', fillcolor='#80A1C1')
        dot.node('H', 'Sigmoid\\n[d]', shape='box', style='filled,rounded', fillcolor='#8D6A9F')
        
        dot.edges(['AB', 'BC', 'CD', 'DE', 'EF', 'EG', 'FG', 'GH'])
        
        dot.render('thesis_diagrams/custom_diagram', format='png', cleanup=True)
        print("  Saved: thesis_diagrams/custom_diagram.png")
    except Exception as e:
        print(f"  Skipped Graphviz: {e}")
    
    # 3. Generate model summary
    try:
        import torchinfo
        print("✓ Generating model summary...")
        summary = torchinfo.summary(model, input_data=[(input_data.shape, torch.float32), target_var], verbose=0)
        with open('thesis_diagrams/model_summary.txt', 'w') as f:
            f.write(str(summary))
        print("  Saved: thesis_diagrams/model_summary.txt")
        print(f"\n  Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"  Skipped summary: {e}")

if __name__ == "__main__":
    generate_all_diagrams()
    print("\n✅ Done! Check the 'thesis_diagrams' folder for your diagrams.")
