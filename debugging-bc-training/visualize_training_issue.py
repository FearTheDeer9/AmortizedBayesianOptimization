#!/usr/bin/env python3
"""
Visualize the actual training issue with variable indexing.
"""

def show_training_issue():
    """Show why mixing different SCM sizes causes problems."""
    
    print("="*80)
    print("THE ACTUAL TRAINING ISSUE")
    print("="*80)
    
    # Example 1: 5-variable SCM
    scm_5var = {
        'variables': ['X0', 'X1', 'X2', 'X3', 'Y'],  # After sorting
        'intervention': 'X3',
        'shape': '[N, 5, 5]'  # Input tensor shape
    }
    
    # Example 2: 12-variable SCM  
    scm_12var = {
        'variables': ['X0', 'X1', 'X10', 'X11', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'Y'],  # After alphabetical sorting!
        'intervention': 'X3',
        'shape': '[N, 13, 5]'  # Input tensor shape
    }
    
    print("\nTraining Example 1 (5-var SCM):")
    print(f"  Input shape: {scm_5var['shape']}")
    print(f"  Variables: {scm_5var['variables']}")
    print(f"  Target: {scm_5var['intervention']}")
    print(f"  → VariableMapper maps 'X3' to index: {scm_5var['variables'].index('X3')}")
    print(f"  → Model learns: For 5-var input, output index 3 for X3")
    
    print("\nTraining Example 2 (12-var SCM):")
    print(f"  Input shape: {scm_12var['shape']}")
    print(f"  Variables: {scm_12var['variables'][:6]}... (showing first 6)")
    print(f"  Target: {scm_12var['intervention']}")
    print(f"  → VariableMapper maps 'X3' to index: {scm_12var['variables'].index('X3')}")
    print(f"  → Model learns: For 13-var input, output index 5 for X3")
    
    print("\n" + "="*60)
    print("THE PROBLEM")
    print("="*60)
    
    print("""
The model receives inputs of DIFFERENT SIZES:
- 5-var SCM:  Input [N, 5, 5]  → Should output 3 for X3
- 12-var SCM: Input [N, 13, 5] → Should output 5 for X3

The model CAN distinguish based on input dimensions, BUT:

1. The policy network might be using padding/truncation to handle variable dimensions
2. OR it's trained separately on different SCM sizes
3. OR there's a bug where we're not handling variable dimensions correctly

Let me check how the policy actually handles different input sizes...
""")

if __name__ == "__main__":
    show_training_issue()