"""
ErdosRenyiGraph class for PARENT_SCALE
"""

class ErdosRenyiGraph:
    """Erdos-Renyi random graph for PARENT_SCALE."""
    
    def __init__(self, n_nodes=5):
        self.nodes = [f'X{i}' for i in range(1, n_nodes)] + ['Y']
        self.target = 'Y'
        
    def get_variables(self):
        """Get all variable names."""
        return self.nodes
        
    def get_target(self):
        """Get target variable."""
        return self.target