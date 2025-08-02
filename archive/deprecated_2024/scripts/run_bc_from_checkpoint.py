#!/usr/bin/env python3
"""
Run BC Development Workflow from Existing Checkpoints

This script allows running the bc_development_workflow.ipynb notebook
starting from existing checkpoints, skipping the training steps.

Usage:
    python run_bc_from_checkpoint.py --surrogate <path> --acquisition <path>
    
    or to use latest checkpoints:
    
    python run_bc_from_checkpoint.py --use-latest
"""

import sys
import os
from pathlib import Path
import argparse
import pickle
import gzip
import logging
from typing import Dict, Optional, Tuple

# Setup paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_latest_checkpoints() -> Tuple[Optional[Path], Optional[Path]]:
    """Find the latest BC checkpoints in the standard directory."""
    checkpoint_base = project_root / "checkpoints/behavioral_cloning/dev"
    
    # Find latest surrogate checkpoint
    surrogate_dir = checkpoint_base / "surrogate"
    surrogate_checkpoints = list(surrogate_dir.glob("surrogate_bc_development_*.pkl"))
    latest_surrogate = max(surrogate_checkpoints, key=lambda p: p.stat().st_mtime) if surrogate_checkpoints else None
    
    # Find latest acquisition checkpoint
    acquisition_dir = checkpoint_base / "acquisition"
    acquisition_checkpoints = list(acquisition_dir.glob("bc_demo_acquisition_*.pkl"))
    latest_acquisition = max(acquisition_checkpoints, key=lambda p: p.stat().st_mtime) if acquisition_checkpoints else None
    
    return latest_surrogate, latest_acquisition


def load_checkpoint(checkpoint_path: Path) -> Dict:
    """Load a checkpoint file (handles gzipped and regular files)."""
    with open(checkpoint_path, 'rb') as f:
        magic = f.read(2)
        f.seek(0)
        
        if magic == b'\x1f\x8b':  # gzip magic number
            with gzip.open(f, 'rb') as gz_f:
                return pickle.load(gz_f)
        else:
            return pickle.load(f)


def verify_checkpoint(checkpoint_path: Path, checkpoint_type: str) -> bool:
    """Verify checkpoint has required fields."""
    try:
        data = load_checkpoint(checkpoint_path)
        
        if checkpoint_type == 'surrogate':
            required_fields = ['model_params', 'training_state', 'config']
        elif checkpoint_type == 'acquisition':
            required_fields = ['policy_params', 'training_state', 'config']
        else:
            return False
            
        for field in required_fields:
            if field not in data:
                logger.error(f"Missing required field '{field}' in {checkpoint_type} checkpoint")
                return False
                
        logger.info(f"✅ {checkpoint_type.capitalize()} checkpoint verified: {checkpoint_path.name}")
        return True
        
    except Exception as e:
        logger.error(f"Error loading {checkpoint_type} checkpoint: {e}")
        return False


def setup_notebook_environment(surrogate_checkpoint: Path, acquisition_checkpoint: Path):
    """Set up the notebook environment to run from checkpoints."""
    
    # Import necessary modules
    import jax
    import jax.numpy as jnp
    import pyrsistent as pyr
    from omegaconf import OmegaConf
    
    # Import BC-specific modules
    from src.causal_bayes_opt.training.bc_data_pipeline import process_all_demonstrations
    from scripts.core.acbo_comparison.method_registry import MethodRegistry
    from scripts.core.acbo_comparison.bc_method_wrappers import (
        create_bc_surrogate_random_method,
        create_bc_acquisition_learning_method,
        create_bc_trained_both_method
    )
    from scripts.core.acbo_comparison.baseline_methods import (
        create_random_baseline_method,
        create_oracle_baseline_method,
        create_learning_baseline_method
    )
    
    print("\n🚀 BC Development Environment (Checkpoint Mode)")
    print(f"📁 Project Root: {project_root}")
    print(f"📦 Surrogate Checkpoint: {surrogate_checkpoint.name}")
    print(f"📦 Acquisition Checkpoint: {acquisition_checkpoint.name}")
    
    # Create mock training results to satisfy notebook requirements
    class MockTrainer:
        def __init__(self, checkpoint_path):
            self.checkpoint_path = checkpoint_path
            
        class CheckpointManager:
            def __init__(self, checkpoint_path):
                self.checkpoint_path = checkpoint_path
                
            def get_latest_checkpoint(self):
                return type('CheckpointInfo', (), {'path': self.checkpoint_path})()
    
    # Create mock results that notebook expects
    surrogate_results = {
        'trainer': MockTrainer(str(surrogate_checkpoint)),
        'training_results': None,
        'training_time': 0.0,
        'training_metrics': [],
        'validation_metrics': [],
        'final_loss': None
    }
    surrogate_results['trainer'].checkpoint_manager = MockTrainer.CheckpointManager(str(surrogate_checkpoint))
    
    acquisition_results = {
        'trainer': MockTrainer(str(acquisition_checkpoint)),
        'training_results': None,
        'training_time': 0.0,
        'final_accuracy': None,
        'training_metrics': []
    }
    acquisition_results['trainer'].checkpoint_manager = MockTrainer.CheckpointManager(str(acquisition_checkpoint))
    
    # Create method registry and register methods
    print("\n🔬 Setting up ACBO Method Registry...")
    method_registry = MethodRegistry()
    
    # Register baseline methods
    method_registry.register_method(create_random_baseline_method())
    method_registry.register_method(create_oracle_baseline_method())
    method_registry.register_method(create_learning_baseline_method())
    
    # Register BC methods with checkpoints
    method_registry.register_method(create_bc_surrogate_random_method(str(surrogate_checkpoint)))
    method_registry.register_method(create_bc_acquisition_learning_method(str(acquisition_checkpoint)))
    method_registry.register_method(create_bc_trained_both_method(str(surrogate_checkpoint), str(acquisition_checkpoint)))
    
    all_methods = method_registry.list_available_methods()
    print(f"✅ Registered {len(all_methods)} methods: {all_methods}")
    
    # Store in globals to simulate notebook environment
    globals_dict = {
        'project_root': project_root,
        'surrogate_results': surrogate_results,
        'acquisition_results': acquisition_results,
        'bc_integration_results': {
            'method_registry': method_registry,
            'registered_methods': all_methods,
            'baseline_methods': ['random_baseline', 'oracle_baseline', 'learning_baseline'],
            'bc_methods': ['bc_surrogate_random', 'bc_acquisition_learning', 'bc_trained_both'],
            'surrogate_checkpoint': str(surrogate_checkpoint),
            'acquisition_checkpoint': str(acquisition_checkpoint)
        }
    }
    
    return globals_dict


def run_notebook_cells(globals_dict: Dict):
    """Run key notebook cells starting from checkpoint loading."""
    
    print("\n📋 Running Notebook Workflow from Checkpoints...")
    
    # The notebook can now be run starting from Cell 7 (Model Loading & Validation)
    # or Cell 8 (ACBO Integration Setup) since we've pre-populated the required globals
    
    print("\n✅ Environment ready for running notebook cells:")
    print("  - Skip Cells 1-6 (Environment Setup through Training)")
    print("  - Start from Cell 7 (Model Loading & Validation) or")
    print("  - Start from Cell 8 (ACBO Integration Setup)")
    print("  - Continue with remaining cells for evaluation")
    
    print("\n💡 To run in Jupyter:")
    print("  1. Open bc_development_workflow.ipynb")
    print("  2. Run this setup code in the first cell:")
    print(f"""
import sys
sys.path.insert(0, '{project_root}')
from scripts.run_bc_from_checkpoint import setup_notebook_environment

# Set up environment with checkpoints
globals_dict = setup_notebook_environment(
    Path('{globals_dict['bc_integration_results']['surrogate_checkpoint']}'),
    Path('{globals_dict['bc_integration_results']['acquisition_checkpoint']}')
)

# Update notebook globals
for key, value in globals_dict.items():
    globals()[key] = value
    """)
    print("  3. Skip to Cell 7 or 8 and continue from there")


def main():
    parser = argparse.ArgumentParser(description='Run BC workflow from existing checkpoints')
    parser.add_argument('--surrogate', type=str, help='Path to surrogate checkpoint')
    parser.add_argument('--acquisition', type=str, help='Path to acquisition checkpoint')
    parser.add_argument('--use-latest', action='store_true', help='Use latest checkpoints found')
    
    args = parser.parse_args()
    
    # Determine which checkpoints to use
    if args.use_latest:
        print("🔍 Finding latest checkpoints...")
        surrogate_checkpoint, acquisition_checkpoint = find_latest_checkpoints()
        
        if not surrogate_checkpoint:
            logger.error("❌ No surrogate checkpoints found")
            return 1
        if not acquisition_checkpoint:
            logger.error("❌ No acquisition checkpoints found")
            return 1
            
        print(f"Found surrogate: {surrogate_checkpoint.name}")
        print(f"Found acquisition: {acquisition_checkpoint.name}")
        
    else:
        if not args.surrogate or not args.acquisition:
            parser.error("Must specify both --surrogate and --acquisition or use --use-latest")
            
        surrogate_checkpoint = Path(args.surrogate)
        acquisition_checkpoint = Path(args.acquisition)
        
        if not surrogate_checkpoint.exists():
            logger.error(f"❌ Surrogate checkpoint not found: {surrogate_checkpoint}")
            return 1
        if not acquisition_checkpoint.exists():
            logger.error(f"❌ Acquisition checkpoint not found: {acquisition_checkpoint}")
            return 1
    
    # Verify checkpoints
    if not verify_checkpoint(surrogate_checkpoint, 'surrogate'):
        return 1
    if not verify_checkpoint(acquisition_checkpoint, 'acquisition'):
        return 1
    
    # Set up environment
    globals_dict = setup_notebook_environment(surrogate_checkpoint, acquisition_checkpoint)
    
    # Run notebook workflow
    run_notebook_cells(globals_dict)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())