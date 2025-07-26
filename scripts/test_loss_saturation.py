#!/usr/bin/env python3
"""
Test loss saturation and gradient vanishing.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

def analyze_tanh_clipping():
    """Analyze the tanh clipping function and its gradients."""
    
    # Test different raw loss values
    raw_losses = np.linspace(0, 100, 1000)
    
    # Current clipping: 10 * tanh(x/10)
    clipped_10 = 10.0 * np.tanh(raw_losses / 10.0)
    grad_10 = 1 - np.tanh(raw_losses / 10.0)**2  # Derivative of tanh
    
    # Alternative 1: Larger scale (50)
    clipped_50 = 50.0 * np.tanh(raw_losses / 50.0)
    grad_50 = 1 - np.tanh(raw_losses / 50.0)**2
    
    # Alternative 2: Log scaling
    clipped_log = np.log1p(raw_losses)
    grad_log = 1 / (1 + raw_losses)
    
    # Alternative 3: Sqrt scaling
    clipped_sqrt = np.sqrt(raw_losses)
    grad_sqrt = 0.5 / np.sqrt(raw_losses + 1e-8)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Loss clipping functions
    ax = axes[0, 0]
    ax.plot(raw_losses, raw_losses, 'k--', label='No clipping', alpha=0.5)
    ax.plot(raw_losses, clipped_10, 'r-', label='10*tanh(x/10)', linewidth=2)
    ax.plot(raw_losses, clipped_50, 'g-', label='50*tanh(x/50)', linewidth=2)
    ax.plot(raw_losses, clipped_log, 'b-', label='log(1+x)', linewidth=2)
    ax.plot(raw_losses, clipped_sqrt, 'm-', label='sqrt(x)', linewidth=2)
    ax.set_xlabel('Raw Loss')
    ax.set_ylabel('Clipped Loss')
    ax.set_title('Loss Clipping Functions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 20)
    
    # Plot 2: Gradients
    ax = axes[0, 1]
    ax.plot(raw_losses, grad_10, 'r-', label='tanh(x/10)', linewidth=2)
    ax.plot(raw_losses, grad_50, 'g-', label='tanh(x/50)', linewidth=2)
    ax.plot(raw_losses, grad_log, 'b-', label='log(1+x)', linewidth=2)
    ax.plot(raw_losses, grad_sqrt, 'm-', label='sqrt(x)', linewidth=2)
    ax.axhline(y=0.1, color='k', linestyle='--', alpha=0.5, label='Gradient=0.1')
    ax.set_xlabel('Raw Loss')
    ax.set_ylabel('Gradient')
    ax.set_title('Gradient of Clipping Functions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 1)
    
    # Plot 3: Zoom on typical loss range
    ax = axes[1, 0]
    typical_range = raw_losses[raw_losses <= 30]
    ax.plot(typical_range, 10.0 * np.tanh(typical_range / 10.0), 'r-', label='Current: 10*tanh(x/10)', linewidth=2)
    ax.plot(typical_range, 50.0 * np.tanh(typical_range / 50.0), 'g-', label='Proposed: 50*tanh(x/50)', linewidth=2)
    ax.plot(typical_range, np.log1p(typical_range), 'b-', label='log(1+x)', linewidth=2)
    ax.axvline(x=22, color='k', linestyle='--', alpha=0.5, label='Typical loss (~22)')
    ax.set_xlabel('Raw Loss')
    ax.set_ylabel('Clipped Loss')
    ax.set_title('Clipping in Typical Loss Range')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Analysis table
    ax = axes[1, 1]
    ax.axis('off')
    
    # Calculate values at typical loss = 22
    loss = 22.0
    table_data = []
    table_data.append(['Method', 'Clipped Loss', 'Gradient', 'Keeps Learning?'])
    
    # Current method
    clipped = 10.0 * np.tanh(loss / 10.0)
    grad = 1 - np.tanh(loss / 10.0)**2
    table_data.append(['10*tanh(x/10)', f'{clipped:.2f}', f'{grad:.4f}', '❌' if grad < 0.1 else '✓'])
    
    # Alternative methods
    clipped = 50.0 * np.tanh(loss / 50.0)
    grad = 1 - np.tanh(loss / 50.0)**2
    table_data.append(['50*tanh(x/50)', f'{clipped:.2f}', f'{grad:.4f}', '❌' if grad < 0.1 else '✓'])
    
    clipped = np.log1p(loss)
    grad = 1 / (1 + loss)
    table_data.append(['log(1+x)', f'{clipped:.2f}', f'{grad:.4f}', '❌' if grad < 0.1 else '✓'])
    
    clipped = np.sqrt(loss)
    grad = 0.5 / np.sqrt(loss)
    table_data.append(['sqrt(x)', f'{clipped:.2f}', f'{grad:.4f}', '❌' if grad < 0.1 else '✓'])
    
    # Create table
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax.set_title(f'Analysis at Raw Loss = {loss}')
    
    plt.tight_layout()
    plt.savefig('loss_clipping_analysis.png', dpi=150, bbox_inches='tight')
    # plt.show()  # Don't show to avoid timeout
    
    # Print analysis
    print("\n=== Loss Clipping Analysis ===")
    print(f"At typical raw loss of 22.0:")
    print(f"  Current (10*tanh(x/10)): clipped={10.0 * np.tanh(22.0/10.0):.2f}, gradient={1 - np.tanh(22.0/10.0)**2:.4f}")
    print(f"  Better (50*tanh(x/50)): clipped={50.0 * np.tanh(22.0/50.0):.2f}, gradient={1 - np.tanh(22.0/50.0)**2:.4f}")
    print(f"  Log scaling: clipped={np.log1p(22.0):.2f}, gradient={1/(1+22.0):.4f}")
    print(f"  Sqrt scaling: clipped={np.sqrt(22.0):.2f}, gradient={0.5/np.sqrt(22.0):.4f}")
    
    print("\n=== Recommendations ===")
    print("1. Current clipping (10*tanh(x/10)) causes gradient vanishing at typical losses")
    print("2. Better options:")
    print("   - 50*tanh(x/50): Maintains gradients better while still bounding loss")
    print("   - log(1+x): Smooth growth, always has gradient")
    print("   - sqrt(x): Moderate growth, good gradients")
    print("3. Consider using gradient clipping instead of loss clipping")

if __name__ == "__main__":
    analyze_tanh_clipping()