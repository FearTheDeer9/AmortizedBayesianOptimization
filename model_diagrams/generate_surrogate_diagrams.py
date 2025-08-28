"""
Minimal, clean architecture diagrams - visual focus, minimal text
"""

from graphviz import Digraph
import os


def generate_minimal_main_architecture(save_path='thesis_diagrams/'):
    """Generate minimal main architecture diagram"""
    os.makedirs(save_path, exist_ok=True)
    
    dot = Digraph(comment='Model Architecture', engine='dot')
    dot.attr(rankdir='TB', bgcolor='white', splines='ortho', nodesep='0.6', ranksep='0.8')
    dot.attr('node', fontname='Helvetica', fontsize='10', penwidth='1', height='0.5')
    dot.attr('edge', penwidth='1', color='#666666')
    
    # Minimal color scheme - subtle and professional
    input_color = '#E8F4FD'
    encoder_color = '#FFF8F0'  
    attention_color = '#F8F0FF'
    output_color = '#F0FFF0'
    
    # Input
    dot.node('input', 'Input Data', 
             shape='box', style='rounded,filled', fillcolor=input_color, width='2')
    
    # Linear Projection
    dot.node('proj', 'Linear Projection', 
             shape='box', style='filled', fillcolor=encoder_color, width='2')
    
    # Create subgraph for alternating attention blocks
    with dot.subgraph(name='cluster_0') as c:
        c.attr(style='dashed', color='gray', label='×8', fontsize='14', labelloc='r')
        
        # Layer 1: Attention over N
        c.node('attn_n', 'Attention (N)', 
               shape='box', style='filled', fillcolor='#E0F0FF', width='1.8')
        
        # Layer 2: Attention over d  
        c.node('attn_d', 'Attention (d)', 
               shape='box', style='filled', fillcolor='#FFE0F0', width='1.8')
        
        c.edge('attn_n', 'attn_d')
    
    # Max Pool
    dot.node('pool', 'Max Pool', 
             shape='box', style='filled', fillcolor=encoder_color, width='2')
    
    # Node Embeddings
    dot.node('embeddings', 'Node Embeddings',
             shape='box', style='rounded,filled', fillcolor='#FFF5E0', width='2')
    
    # Parent Attention
    dot.node('parent_att', 'Parent Attention',
             shape='box', style='rounded,filled', fillcolor=attention_color, width='2')
    
    # Sigmoid
    dot.node('sigmoid', 'Sigmoid',
             shape='box', style='filled', fillcolor='#F0F0F0', width='1.5')
    
    # Output
    dot.node('output', 'Parent Probabilities',
             shape='box', style='rounded,filled', fillcolor=output_color, width='2')
    
    # Connections
    dot.edge('input', 'proj')
    dot.edge('proj', 'attn_n')
    dot.edge('attn_d', 'pool')
    dot.edge('pool', 'embeddings')
    dot.edge('embeddings', 'parent_att')
    dot.edge('parent_att', 'sigmoid')
    dot.edge('sigmoid', 'output')
    
    # Render
    dot.render(os.path.join(save_path, 'architecture_minimal'), format='png', cleanup=True)
    dot.render(os.path.join(save_path, 'architecture_minimal'), format='pdf', cleanup=True)
    print(f"✓ Saved minimal architecture to {save_path}architecture_minimal.png/pdf")


def generate_minimal_attention_block(save_path='thesis_diagrams/'):
    """Generate minimal attention block detail"""
    os.makedirs(save_path, exist_ok=True)
    
    dot = Digraph(comment='Attention Block', engine='dot')
    dot.attr(rankdir='TB', bgcolor='white', splines='ortho', nodesep='0.5')
    dot.attr('node', fontname='Helvetica', fontsize='10', shape='box', style='filled', penwidth='1')
    dot.attr('edge', penwidth='1', color='#666666')
    
    # Components
    dot.node('input', 'Input', fillcolor='#E8F4FD', width='1.5')
    dot.node('ln1', 'LayerNorm', fillcolor='#FFF8E1', width='1.5')
    dot.node('mha', 'Multi-Head\nAttention', fillcolor='#FCE4EC', width='1.5')
    dot.node('add1', '+', shape='circle', width='0.3', fillcolor='#F5F5F5')
    dot.node('ln2', 'LayerNorm', fillcolor='#FFF8E1', width='1.5')
    dot.node('ffn', 'FFN', fillcolor='#E8F5E9', width='1.5')
    dot.node('add2', '+', shape='circle', width='0.3', fillcolor='#F5F5F5')
    dot.node('output', 'Output', fillcolor='#E8F4FD', width='1.5')
    
    # Main flow
    dot.edge('input', 'ln1')
    dot.edge('ln1', 'mha')
    dot.edge('mha', 'add1')
    dot.edge('add1', 'ln2')
    dot.edge('ln2', 'ffn')
    dot.edge('ffn', 'add2')
    dot.edge('add2', 'output')
    
    # Residual connections
    dot.edge('input', 'add1', style='dashed')
    dot.edge('add1', 'add2', style='dashed')
    
    # Render
    dot.render(os.path.join(save_path, 'attention_block_minimal'), format='png', cleanup=True)
    dot.render(os.path.join(save_path, 'attention_block_minimal'), format='pdf', cleanup=True)
    print(f"✓ Saved minimal attention block to {save_path}attention_block_minimal.png/pdf")


def generate_minimal_score_network(save_path='thesis_diagrams/'):
    """Generate minimal score network"""
    os.makedirs(save_path, exist_ok=True)
    
    dot = Digraph(comment='Score Network', engine='dot')
    dot.attr(rankdir='TB', bgcolor='white', splines='ortho')
    dot.attr('node', fontname='Helvetica', fontsize='10', shape='box', style='filled', penwidth='1')
    dot.attr('edge', penwidth='1', color='#666666')
    
    # Inputs
    dot.node('target', 'Target', fillcolor='#E8F5E9', width='1.2')
    dot.node('candidate', 'Candidate', fillcolor='#E3F2FD', width='1.2')
    
    # Network components
    dot.node('concat', 'Concat', fillcolor='#FFF3E0', width='1.2')
    dot.node('mlp', 'MLP', fillcolor='#F3E5F5', width='1.2', height='0.8')
    dot.node('score', 'Score', fillcolor='#C8E6C9', width='1.2')
    
    # Connections
    dot.edge('target', 'concat')
    dot.edge('candidate', 'concat')
    dot.edge('concat', 'mlp')
    dot.edge('mlp', 'score')
    
    # Render
    dot.render(os.path.join(save_path, 'score_network_minimal'), format='png', cleanup=True)
    dot.render(os.path.join(save_path, 'score_network_minimal'), format='pdf', cleanup=True)
    print(f"✓ Saved minimal score network to {save_path}score_network_minimal.png/pdf")


def generate_enhanced_architecture(save_path='thesis_diagrams/'):
    """Generate a more visually enhanced version similar to the reference"""
    os.makedirs(save_path, exist_ok=True)
    
    dot = Digraph(comment='Model Architecture', engine='dot')
    dot.attr(rankdir='TB', bgcolor='white', splines='ortho', nodesep='0.8', ranksep='1.0')
    dot.attr('node', fontname='Helvetica-Bold', fontsize='11', penwidth='1.5')
    dot.attr('edge', penwidth='1.5', color='#333333', arrowsize='0.8')
    
    # Define a cleaner style
    box_style = 'filled,rounded'
    
    # Input
    dot.node('input', 'INPUT', 
             shape='box', style=box_style, fillcolor='#4A90E2', fontcolor='white', width='2.5', height='0.6')
    
    # Encoder section with subgraph
    with dot.subgraph(name='cluster_encoder') as c:
        c.attr(label='ENCODER', style='rounded,filled', fillcolor='#F5F5F5', fontsize='12', labeljust='l')
        
        c.node('proj', 'Linear Projection', 
               shape='box', style='filled', fillcolor='white', width='2.2')
        
        # Alternating attention with multiplier
        with c.subgraph(name='cluster_alt') as s:
            s.attr(label='', style='dashed', color='#999999')
            s.node('block1', 'Attention N', 
                   shape='box', style='filled', fillcolor='#B8D4F0', width='1.8')
            s.node('block2', 'Attention d', 
                   shape='box', style='filled', fillcolor='#F0B8D4', width='1.8')
            s.edge('block1', 'block2')
        
        c.node('mult', '× 8', shape='plaintext', fontsize='16', fontcolor='#666666')
        
        c.node('pool', 'Max Pool', 
               shape='box', style='filled', fillcolor='white', width='2.2')
        
        # Internal connections
        c.edge('proj', 'block1')
        c.edge('block2', 'pool')
    
    # Embeddings
    dot.node('emb', 'EMBEDDINGS', 
             shape='box', style=box_style, fillcolor='#F39C12', fontcolor='white', width='2.5', height='0.6')
    
    # Attention
    dot.node('att', 'ATTENTION', 
             shape='box', style=box_style, fillcolor='#9B59B6', fontcolor='white', width='2.5', height='0.6')
    
    # Output
    dot.node('out', 'OUTPUT', 
             shape='box', style=box_style, fillcolor='#27AE60', fontcolor='white', width='2.5', height='0.6')
    
    # Main connections
    dot.edge('input', 'proj')
    dot.edge('pool', 'emb')
    dot.edge('emb', 'att')
    dot.edge('att', 'out')
    
    # Position the multiplier
    dot.body.append('{rank=same; block2; mult;}')
    
    # Render
    dot.render(os.path.join(save_path, 'architecture_enhanced'), format='png', cleanup=True)
    dot.render(os.path.join(save_path, 'architecture_enhanced'), format='pdf', cleanup=True)
    print(f"✓ Saved enhanced architecture to {save_path}architecture_enhanced.png/pdf")


def main():
    print("\n" + "="*60)
    print("Generating Minimal Thesis Architecture Diagrams")
    print("="*60)
    
    os.makedirs('thesis_diagrams', exist_ok=True)
    
    print("\n1. Minimal Main Architecture...")
    generate_minimal_main_architecture()
    
    print("\n2. Minimal Attention Block...")
    generate_minimal_attention_block()
    
    print("\n3. Minimal Score Network...")
    generate_minimal_score_network()
    
    print("\n4. Enhanced Architecture Version...")
    generate_enhanced_architecture()
    
    print("\n" + "="*60)
    print("SUCCESS! Minimal diagrams generated")
    print("="*60)
    print("\nGenerated files in 'thesis_diagrams/':")
    print("  • architecture_minimal.png - Clean minimal architecture")
    print("  • architecture_enhanced.png - Enhanced visual style")
    print("  • attention_block_minimal.png - Simple attention block")
    print("  • score_network_minimal.png - Simple score network")


if __name__ == "__main__":
    import subprocess
    subprocess.run(['pip', 'install', 'graphviz'], capture_output=True)
    main()