"""
Minimal architecture diagrams for Quantile Policy model
"""

from graphviz import Digraph
import os


def generate_quantile_policy_main(save_path='thesis_diagrams/'):
    """Generate main architecture diagram for quantile policy"""
    os.makedirs(save_path, exist_ok=True)
    
    dot = Digraph(comment='Quantile Policy Architecture', engine='dot')
    dot.attr(rankdir='TB', bgcolor='white', splines='ortho', nodesep='0.6', ranksep='0.8')
    dot.attr('node', fontname='Helvetica', fontsize='10', penwidth='1', height='0.5')
    dot.attr('edge', penwidth='1', color='#666666')
    
    # Color scheme
    input_color = '#E8F4FD'
    processing_color = '#FFF8F0'
    attention_color = '#F8F0FF'
    output_color = '#F0FFF0'
    
    # Input
    dot.node('input', 'Input Tensor\n[T, n_vars, 4]', 
             shape='box', style='rounded,filled', fillcolor=input_color, width='2')
    
    # Input Projection
    dot.node('proj', 'Input Projection', 
             shape='box', style='filled', fillcolor=processing_color, width='2')
    
    # Alternating attention blocks with ×2 notation
    with dot.subgraph(name='cluster_0') as c:
        c.attr(style='dashed', color='gray', label='×2', fontsize='14', labelloc='r')
        
        # Sample attention (over T)
        c.node('attn_t', 'Sample Attention (T)', 
               shape='box', style='filled', fillcolor='#E0F0FF', width='1.8')
        
        # Variable attention (over n_vars)
        c.node('attn_v', 'Variable Attention (d)', 
               shape='box', style='filled', fillcolor='#FFE0F0', width='1.8')
        
        c.edge('attn_t', 'attn_v')
    
    # Attention Pooling
    dot.node('pool', 'Attention Pooling', 
             shape='box', style='filled', fillcolor=processing_color, width='2')
    
    # Layer Norm
    dot.node('norm', 'Layer Norm',
             shape='box', style='filled', fillcolor='#FFF5E0', width='1.5')
    
    # Quantile Head
    dot.node('quantile', 'Quantile Head',
             shape='box', style='rounded,filled', fillcolor=attention_color, width='2')
    
    # Output
    dot.node('output', 'Quantile Scores\n[n_vars, k]',
             shape='box', style='rounded,filled', fillcolor=output_color, width='2')
    
    # Connections
    dot.edge('input', 'proj')
    dot.edge('proj', 'attn_t')
    dot.edge('attn_v', 'pool')
    dot.edge('pool', 'norm')
    dot.edge('norm', 'quantile')
    dot.edge('quantile', 'output')
    
    # Render
    dot.render(os.path.join(save_path, 'quantile_policy_main'), format='png', cleanup=True)
    dot.render(os.path.join(save_path, 'quantile_policy_main'), format='pdf', cleanup=True)
    print(f"✓ Saved quantile policy main architecture to {save_path}quantile_policy_main.png/pdf")


def generate_sample_attention_detail(save_path='thesis_diagrams/'):
    """Generate detail of sample attention mechanism"""
    os.makedirs(save_path, exist_ok=True)
    
    dot = Digraph(comment='Sample Attention', engine='dot')
    dot.attr(rankdir='TB', bgcolor='white', splines='ortho', nodesep='0.5')
    dot.attr('node', fontname='Helvetica', fontsize='10', shape='box', style='filled', penwidth='1')
    dot.attr('edge', penwidth='1', color='#666666')
    
    # Input
    dot.node('input', 'Variable Samples\n[T, hidden]', fillcolor='#E8F4FD', width='1.8')
    
    # Components
    dot.node('ln1', 'LayerNorm', fillcolor='#FFF8E1', width='1.5')
    dot.node('mha', 'Multi-Head\nAttention', fillcolor='#E0F0FF', width='1.5')
    dot.node('add1', '+', shape='circle', width='0.3', fillcolor='#F5F5F5')
    dot.node('ln2', 'LayerNorm', fillcolor='#FFF8E1', width='1.5')
    dot.node('ffn', 'FFN', fillcolor='#E8F5E9', width='1.5')
    dot.node('add2', '+', shape='circle', width='0.3', fillcolor='#F5F5F5')
    dot.node('output', 'Output\n[T, hidden]', fillcolor='#E8F4FD', width='1.8')
    
    # Note about vmap removed per request
    
    # Flow
    dot.edge('input', 'ln1')
    dot.edge('ln1', 'mha')
    dot.edge('mha', 'add1')
    dot.edge('add1', 'ln2')
    dot.edge('ln2', 'ffn')
    dot.edge('ffn', 'add2')
    dot.edge('add2', 'output')
    
    # Residuals
    dot.edge('input', 'add1', style='dashed')
    dot.edge('add1', 'add2', style='dashed')
    
    # Render
    dot.render(os.path.join(save_path, 'sample_attention_detail'), format='png', cleanup=True)
    dot.render(os.path.join(save_path, 'sample_attention_detail'), format='pdf', cleanup=True)
    print(f"✓ Saved sample attention detail to {save_path}sample_attention_detail.png/pdf")


def generate_attention_pooling_detail(save_path='thesis_diagrams/'):
    """Generate detail of attention pooling mechanism"""
    os.makedirs(save_path, exist_ok=True)
    
    dot = Digraph(comment='Attention Pooling', engine='dot')
    dot.attr(rankdir='TB', bgcolor='white', splines='ortho')
    dot.attr('node', fontname='Helvetica', fontsize='10', shape='box', style='filled', penwidth='1')
    dot.attr('edge', penwidth='1', color='#666666')
    
    # Components
    dot.node('input', 'Features\n[T, n_vars, hidden]', fillcolor='#E8F4FD', width='2')
    dot.node('query', 'Learned Query\n[n_vars, hidden]', fillcolor='#FFE0F0', width='1.8')
    
    dot.node('scores', 'Attention Scores\n[T, n_vars]', fillcolor='#FFF3E0', width='1.8')
    dot.node('softmax', 'Softmax\n(over T)', fillcolor='#F0F0F0', width='1.5')
    dot.node('weights', 'Weights\n[T, n_vars]', fillcolor='#FFF3E0', width='1.5')
    
    dot.node('weighted', 'Weighted Sum', fillcolor='#E8F5E9', width='1.8')
    dot.node('output', 'Pooled Features\n[n_vars, hidden]', fillcolor='#F0FFF0', width='2')
    
    # Connections
    dot.edge('input', 'scores')
    dot.edge('query', 'scores')
    dot.edge('scores', 'softmax')
    dot.edge('softmax', 'weights')
    dot.edge('weights', 'weighted')
    dot.edge('input', 'weighted')
    dot.edge('weighted', 'output')
    
    # Render
    dot.render(os.path.join(save_path, 'attention_pooling_detail'), format='png', cleanup=True)
    dot.render(os.path.join(save_path, 'attention_pooling_detail'), format='pdf', cleanup=True)
    print(f"✓ Saved attention pooling detail to {save_path}attention_pooling_detail.png/pdf")


def generate_quantile_head_detail(save_path='thesis_diagrams/'):
    """Generate detail of quantile head"""
    os.makedirs(save_path, exist_ok=True)
    
    dot = Digraph(comment='Quantile Head', engine='dot')
    dot.attr(rankdir='TB', bgcolor='white', splines='ortho')
    dot.attr('node', fontname='Helvetica', fontsize='10', shape='box', style='filled', penwidth='1')
    dot.attr('edge', penwidth='1', color='#666666')
    
    # Input
    dot.node('input', 'Variable Features\n[n_vars, hidden]', fillcolor='#E8F4FD', width='2')
    
    # MLP
    dot.node('linear1', 'Linear\nhidden → hidden/2', fillcolor='#F3E5F5', width='1.8')
    dot.node('gelu', 'GELU', fillcolor='#FFEBEE', width='1.2')
    dot.node('linear2', 'Linear\nhidden/2 → k', fillcolor='#F3E5F5', width='1.8')
    
    # Output with quantiles
    dot.node('output', 'Quantile Scores\n[n_vars, k]\n(k quantiles)', fillcolor='#C8E6C9', width='2')
    
    # Connections
    dot.edge('input', 'linear1')
    dot.edge('linear1', 'gelu')
    dot.edge('gelu', 'linear2')
    dot.edge('linear2', 'output')
    
    # Render
    dot.render(os.path.join(save_path, 'quantile_head_detail'), format='png', cleanup=True)
    dot.render(os.path.join(save_path, 'quantile_head_detail'), format='pdf', cleanup=True)
    print(f"✓ Saved quantile head detail to {save_path}quantile_head_detail.png/pdf")


def generate_enhanced_quantile_architecture(save_path='thesis_diagrams/'):
    """Generate enhanced version with better visual style"""
    os.makedirs(save_path, exist_ok=True)
    
    dot = Digraph(comment='Quantile Policy', engine='dot')
    dot.attr(rankdir='TB', bgcolor='white', splines='ortho', nodesep='0.8', ranksep='1.0')
    dot.attr('node', fontname='Helvetica-Bold', fontsize='11', penwidth='1.5')
    dot.attr('edge', penwidth='1.5', color='#333333', arrowsize='0.8')
    
    box_style = 'filled,rounded'
    
    # Input
    dot.node('input', 'INPUT\n[T, n_vars, 4]', 
             shape='box', style=box_style, fillcolor='#4A90E2', fontcolor='white', width='2.5', height='0.7')
    
    # Processing section
    with dot.subgraph(name='cluster_process') as c:
        c.attr(label='PROCESSING', style='rounded,filled', fillcolor='#F5F5F5', fontsize='12', labeljust='l')
        
        c.node('proj', 'Projection', 
               shape='box', style='filled', fillcolor='white', width='2')
        
        # Alternating attention
        with c.subgraph(name='cluster_alt') as s:
            s.attr(label='', style='dashed', color='#999999')
            s.node('att_t', 'Attention T', 
                   shape='box', style='filled', fillcolor='#B8D4F0', width='1.6')
            s.node('att_v', 'Attention V', 
                   shape='box', style='filled', fillcolor='#F0B8D4', width='1.6')
            s.edge('att_t', 'att_v')
        
        c.node('mult', '× 2', shape='plaintext', fontsize='16', fontcolor='#666666')
        
        c.node('pool', 'Attention Pool', 
               shape='box', style='filled', fillcolor='white', width='2')
        
        c.edge('proj', 'att_t')
        c.edge('att_v', 'pool')
    
    # Quantile head
    dot.node('quantile', 'QUANTILE HEAD', 
             shape='box', style=box_style, fillcolor='#9B59B6', fontcolor='white', width='2.5', height='0.6')
    
    # Output
    dot.node('output', 'SCORES\n[n_vars × k]', 
             shape='box', style=box_style, fillcolor='#27AE60', fontcolor='white', width='2.5', height='0.7')
    
    # Main flow
    dot.edge('input', 'proj')
    dot.edge('pool', 'quantile')
    dot.edge('quantile', 'output')
    
    # Position multiplier
    dot.body.append('{rank=same; att_v; mult;}')
    
    # Render
    dot.render(os.path.join(save_path, 'quantile_policy_enhanced'), format='png', cleanup=True)
    dot.render(os.path.join(save_path, 'quantile_policy_enhanced'), format='pdf', cleanup=True)
    print(f"✓ Saved enhanced quantile policy to {save_path}quantile_policy_enhanced.png/pdf")


def main():
    print("\n" + "="*60)
    print("Generating Quantile Policy Architecture Diagrams")
    print("="*60)
    
    os.makedirs('thesis_diagrams', exist_ok=True)
    
    print("\n1. Main Architecture...")
    generate_quantile_policy_main()
    
    print("\n2. Sample Attention Detail...")
    generate_sample_attention_detail()
    
    print("\n3. Attention Pooling Detail...")
    generate_attention_pooling_detail()
    
    print("\n4. Quantile Head Detail...")
    generate_quantile_head_detail()
    
    print("\n5. Enhanced Version...")
    generate_enhanced_quantile_architecture()
    
    print("\n" + "="*60)
    print("SUCCESS! Quantile Policy diagrams generated")
    print("="*60)
    print("\nGenerated files in 'thesis_diagrams/':")
    print("  • quantile_policy_main.png - Main architecture")
    print("  • quantile_policy_enhanced.png - Enhanced visual style")
    print("  • sample_attention_detail.png - Sample attention mechanism")
    print("  • attention_pooling_detail.png - Attention pooling mechanism")
    print("  • quantile_head_detail.png - Quantile output head")


if __name__ == "__main__":
    import subprocess
    subprocess.run(['pip', 'install', 'graphviz'], capture_output=True)
    main()