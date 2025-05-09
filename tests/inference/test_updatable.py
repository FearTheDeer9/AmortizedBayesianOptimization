import unittest
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

# This will fail until the interface is implemented
try:
    from causal_meta.inference.interfaces import Updatable, Data
    INTERFACE_IMPORTED = True
except (ImportError, AttributeError):
    INTERFACE_IMPORTED = False
    # Mock definitions for testing
    class Data(Dict[str, Any]):
        pass
    
    class Updatable(ABC):
        @abstractmethod
        def update(self, data: Data) -> bool:
            pass
        
        @abstractmethod
        def reset(self) -> None:
            pass


class TestUpdatableInterface(unittest.TestCase):
    """Test suite for the Updatable interface."""
    
    def test_interface_exists(self):
        """Test that the interface exists in the module."""
        self.assertTrue(INTERFACE_IMPORTED, "Updatable interface not found in module")
    
    def test_interface_contract(self):
        """Test that the interface contract is enforced."""
        
        # Cannot instantiate abstract class
        with self.assertRaises(TypeError):
            model = Updatable()
    
    def test_concrete_implementation(self):
        """Test that a concrete implementation can be instantiated."""
        
        class SimpleUpdatableModel(Updatable):
            def __init__(self):
                self.data = None
                self.updates_count = 0
                self.initial_state = {'param': 0.0}
                self.current_state = self.initial_state.copy()
            
            def update(self, data: Data) -> bool:
                # Just store the data and count updates
                self.data = data
                self.updates_count += 1
                self.current_state['param'] += 0.1
                return True
            
            def reset(self) -> None:
                # Reset to initial state
                self.data = None
                self.updates_count = 0
                self.current_state = self.initial_state.copy()
        
        # Should be able to instantiate concrete implementation
        model = SimpleUpdatableModel()
        self.assertIsInstance(model, Updatable)
        
        # Test update method
        mock_data = {"observations": np.random.randn(100, 3)}
        success = model.update(mock_data)
        self.assertTrue(success)
        self.assertEqual(model.updates_count, 1)
        self.assertEqual(model.current_state['param'], 0.1)
        
        # Test multiple updates
        model.update(mock_data)
        self.assertEqual(model.updates_count, 2)
        self.assertEqual(model.current_state['param'], 0.2)
        
        # Test reset method
        model.reset()
        self.assertEqual(model.updates_count, 0)
        self.assertEqual(model.current_state['param'], 0.0)
        self.assertIsNone(model.data)

    def test_error_handling(self):
        """Test error handling for invalid data."""
        
        class DataValidatingModel(Updatable):
            def __init__(self):
                self.reset()
            
            def update(self, data: Data) -> bool:
                if not isinstance(data, dict):
                    raise TypeError("Data must be a dictionary")
                
                if 'observations' not in data:
                    raise ValueError("Data must contain 'observations' key")
                
                self.data = data
                self.updated = True
                return True
            
            def reset(self) -> None:
                self.data = None
                self.updated = False
        
        model = DataValidatingModel()
        
        # Test invalid data types
        with self.assertRaises(TypeError):
            model.update("not a dictionary")
        
        # Test missing required keys
        with self.assertRaises(ValueError):
            model.update({"wrong_key": "value"})
        
        # Test successful update
        mock_data = {"observations": np.random.randn(100, 3)}
        success = model.update(mock_data)
        self.assertTrue(success)
        self.assertTrue(model.updated)
        
        # Test reset functionality
        model.reset()
        self.assertFalse(model.updated)
        self.assertIsNone(model.data)


if __name__ == '__main__':
    unittest.main() 