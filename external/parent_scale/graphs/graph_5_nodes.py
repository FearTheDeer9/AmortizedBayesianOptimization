"""
Graph5Nodes class for PARENT_SCALE
"""

class Graph5Nodes:
    """Base class for 5-node graphs used in PARENT_SCALE."""
    
    def __init__(self):
        self.nodes = ['X1', 'X2', 'X3', 'X4', 'Y']
        self.target = 'Y'
        
    def get_variables(self):
        """Get all variable names."""
        return self.nodes
        
    def get_target(self):
        """Get target variable."""
        return self.target