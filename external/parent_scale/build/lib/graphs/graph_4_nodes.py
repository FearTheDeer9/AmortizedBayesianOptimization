"""
Graph4Nodes class for PARENT_SCALE
"""

class Graph4Nodes:
    """Base class for 4-node graphs used in PARENT_SCALE."""
    
    def __init__(self):
        self.nodes = ['X1', 'X2', 'X3', 'Y']
        self.target = 'Y'
        
    def get_variables(self):
        """Get all variable names."""
        return self.nodes
        
    def get_target(self):
        """Get target variable."""
        return self.target