"""
BC Checkpoint Loader Cell

Insert this code into a new cell after Cell 4 (Training Configuration) 
to skip training and load from existing checkpoints.

This allows you to:
1. Skip Cells 5-6 (Surrogate and Acquisition Training)
2. Jump directly to Cell 7 (Model Loading & Validation)
3. Continue with the rest of the notebook normally
"""

# ===== INSERT THIS CODE INTO A NEW JUPYTER CELL =====

print("🔄 Loading BC Models from Existing Checkpoints...")

from pathlib import Path
import pickle
import gzip

# Find latest checkpoints
checkpoint_base_dir = project_root / "checkpoints/behavioral_cloning/dev"

# Find latest surrogate checkpoint
surrogate_dir = checkpoint_base_dir / "surrogate"
surrogate_checkpoints = list(surrogate_dir.glob("surrogate_bc_development_*.pkl"))
if not surrogate_checkpoints:
    raise RuntimeError("❌ No surrogate checkpoints found!")
latest_surrogate = max(surrogate_checkpoints, key=lambda p: p.stat().st_mtime)

# Find latest acquisition checkpoint
acquisition_dir = checkpoint_base_dir / "acquisition"
acquisition_checkpoints = list(acquisition_dir.glob("bc_demo_acquisition_*.pkl"))
if not acquisition_checkpoints:
    # Try alternative pattern
    acquisition_checkpoints = list(acquisition_dir.glob("*acquisition*.pkl"))
    
if not acquisition_checkpoints:
    raise RuntimeError("❌ No acquisition checkpoints found!")
latest_acquisition = max(acquisition_checkpoints, key=lambda p: p.stat().st_mtime)

print(f"✅ Found checkpoints:")
print(f"  Surrogate: {latest_surrogate.name}")
print(f"  Acquisition: {latest_acquisition.name}")

# Create mock trainer classes to satisfy notebook requirements
class MockCheckpointManager:
    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
    
    def get_latest_checkpoint(self):
        """Return checkpoint info in expected format."""
        return type('CheckpointInfo', (), {'path': str(self.checkpoint_path)})()

class MockTrainer:
    def __init__(self, checkpoint_path):
        self.checkpoint_manager = MockCheckpointManager(checkpoint_path)

# Create mock results that the notebook expects from training
surrogate_results = {
    'trainer': MockTrainer(latest_surrogate),
    'training_results': None,
    'training_time': 0.0,
    'training_metrics': [],
    'validation_metrics': [],
    'final_loss': None,
    'processed_dataset': processed_data  # From Cell 3
}

acquisition_results = {
    'trainer': MockTrainer(latest_acquisition),
    'training_results': None,  
    'training_time': 0.0,
    'final_accuracy': None,
    'training_metrics': []
}

print("\n💾 Mock training results created from checkpoints")
print("✅ You can now skip to Cell 7 (Model Loading & Validation)")

# Optionally load and inspect checkpoint contents
def inspect_checkpoint(checkpoint_path):
    """Load and display checkpoint contents."""
    with open(checkpoint_path, 'rb') as f:
        magic = f.read(2)
        f.seek(0)
        
        if magic == b'\x1f\x8b':  # gzip
            with gzip.open(f, 'rb') as gz_f:
                data = pickle.load(gz_f)
        else:
            data = pickle.load(f)
    
    return data

# Quick inspection
print("\n🔍 Checkpoint Contents:")
surrogate_data = inspect_checkpoint(latest_surrogate)
print(f"Surrogate checkpoint keys: {list(surrogate_data.keys())}")

acquisition_data = inspect_checkpoint(latest_acquisition)  
print(f"Acquisition checkpoint keys: {list(acquisition_data.keys())}")

print("\n✅ Ready to continue with notebook from Cell 7!")

# ===== END OF LOADER CELL =====