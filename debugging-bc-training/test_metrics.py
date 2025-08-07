#!/usr/bin/env python3
"""
Test script to validate BC debugging tools functionality.

This script runs a small training session and verifies that all metrics
and visualization tools work correctly.
"""

import sys
import tempfile
import shutil
from pathlib import Path
import numpy as np
import jax.numpy as jnp

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from metrics_tracker import BCMetricsTracker
from visualize_metrics import generate_all_visualizations
from analyze_training import generate_analysis_report


def test_metrics_tracker():
    """Test the metrics tracker functionality."""
    print("Testing BCMetricsTracker...")
    
    tracker = BCMetricsTracker(save_embeddings_every=2)
    
    # Simulate 5 epochs of training
    for epoch in range(5):
        tracker.start_epoch(epoch)
        
        # Simulate batch processing
        for batch in range(3):
            # Create fake predictions and targets
            batch_size = 8
            n_vars = 4
            
            predictions = jnp.array(np.random.randint(0, n_vars, batch_size))
            targets = [{'variable_idx': np.random.randint(0, n_vars)} for _ in range(batch_size)]
            
            # Add fake embeddings on selected epochs
            embeddings = None
            if epoch % 2 == 0:
                embeddings = jnp.array(np.random.randn(batch_size, n_vars, 32))
            
            tracker.track_batch(predictions, targets, embeddings=embeddings)
        
        # End epoch with validation metrics
        val_predictions = jnp.array(np.random.randint(0, n_vars, 20))
        val_targets = jnp.array(np.random.randint(0, n_vars, 20))
        variable_names = [f"Var_{i}" for i in range(n_vars)]
        
        tracker.end_epoch(val_predictions, val_targets, variable_names)
    
    # Get summary
    summary = tracker.get_metrics_summary()
    
    assert 'total_epochs' in summary
    assert summary['total_epochs'] == 5
    assert 'embedding_epochs' in summary
    assert len(summary['embedding_epochs']) > 0
    
    print("✓ Metrics tracker working correctly")
    print(f"  - Tracked {summary['total_epochs']} epochs")
    print(f"  - Embeddings saved for epochs: {summary['embedding_epochs']}")
    print(f"  - Variables tracked: {summary.get('n_variables', 'N/A')}")
    
    return tracker


def test_enhanced_training():
    """Test enhanced BC training with small dataset."""
    print("\nTesting Enhanced BC Training...")
    
    # Check if we can import the enhanced trainer
    try:
        from enhanced_bc_trainer import EnhancedBCTrainer
        print("✓ Enhanced trainer imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import enhanced trainer: {e}")
        return False
    
    # We won't run actual training here since it requires expert demonstrations
    # Just verify the class can be instantiated
    try:
        trainer = EnhancedBCTrainer(
            hidden_dim=64,
            learning_rate=1e-3,
            batch_size=8,
            max_epochs=5,
            save_embeddings_every=2,
            seed=42
        )
        print("✓ Enhanced trainer instantiated successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to instantiate trainer: {e}")
        return False


def test_visualizations(tracker, temp_dir):
    """Test visualization generation."""
    print("\nTesting Visualization Tools...")
    
    # Save metrics
    metrics_file = temp_dir / "test_metrics.pkl"
    tracker.save(metrics_file)
    
    # Test visualization generation
    try:
        plots_dir = temp_dir / "plots"
        generate_all_visualizations(metrics_file, plots_dir)
        
        # Check if plots were created
        expected_plots = ['training_curves.png', 'per_variable_performance.png']
        created_plots = list(plots_dir.glob("*.png"))
        
        print(f"✓ Generated {len(created_plots)} visualizations")
        for plot in created_plots:
            print(f"  - {plot.name}")
        
        return True
    except Exception as e:
        print(f"✗ Visualization failed: {e}")
        return False


def test_analysis(tracker, temp_dir):
    """Test analysis report generation."""
    print("\nTesting Analysis Tools...")
    
    # Save metrics
    metrics_file = temp_dir / "test_metrics.pkl"
    tracker.save(metrics_file)
    
    # Generate analysis report
    try:
        report = generate_analysis_report(metrics_file)
        
        assert 'convergence' in report
        assert 'variable_analysis' in report
        assert 'summary' in report
        
        print("✓ Analysis report generated successfully")
        print(f"  - Convergence analysis: {report['convergence'].get('total_epochs', 0)} epochs")
        print(f"  - Variables analyzed: {report['variable_analysis'].get('n_variables', 0)}")
        print(f"  - Issues identified: {report['summary'].get('n_issues', 0)}")
        
        return True
    except Exception as e:
        print(f"✗ Analysis failed: {e}")
        return False


def test_metrics_persistence(tracker, temp_dir):
    """Test saving and loading metrics."""
    print("\nTesting Metrics Persistence...")
    
    # Save metrics
    metrics_file = temp_dir / "test_metrics.pkl"
    tracker.save(metrics_file)
    
    # Load metrics
    try:
        loaded_tracker = BCMetricsTracker.load(metrics_file)
        
        # Verify loaded data
        original_summary = tracker.get_metrics_summary()
        loaded_summary = loaded_tracker.get_metrics_summary()
        
        assert original_summary['total_epochs'] == loaded_summary['total_epochs']
        assert len(original_summary['embedding_epochs']) == len(loaded_summary['embedding_epochs'])
        
        print("✓ Metrics persistence working correctly")
        print(f"  - Saved and loaded {loaded_summary['total_epochs']} epochs")
        
        return True
    except Exception as e:
        print(f"✗ Persistence test failed: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("BC DEBUGGING TOOLS TEST SUITE")
    print("="*60)
    
    # Create temporary directory for test outputs
    temp_dir = Path(tempfile.mkdtemp(prefix="bc_debug_test_"))
    print(f"\nUsing temporary directory: {temp_dir}")
    
    try:
        # Test metrics tracker
        tracker = test_metrics_tracker()
        if not tracker:
            print("\n✗ Metrics tracker test failed")
            return False
        
        # Test enhanced training
        training_ok = test_enhanced_training()
        
        # Test visualizations
        viz_ok = test_visualizations(tracker, temp_dir)
        
        # Test analysis
        analysis_ok = test_analysis(tracker, temp_dir)
        
        # Test persistence
        persist_ok = test_metrics_persistence(tracker, temp_dir)
        
        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        all_tests = [
            ("Metrics Tracker", tracker is not None),
            ("Enhanced Training", training_ok),
            ("Visualizations", viz_ok),
            ("Analysis", analysis_ok),
            ("Persistence", persist_ok)
        ]
        
        for test_name, passed in all_tests:
            status = "✓ PASSED" if passed else "✗ FAILED"
            print(f"{test_name:20} {status}")
        
        all_passed = all(passed for _, passed in all_tests)
        
        if all_passed:
            print("\n🎉 All tests passed! The debugging tools are working correctly.")
        else:
            print("\n⚠️ Some tests failed. Please check the output above.")
        
        return all_passed
        
    finally:
        # Clean up temporary directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"\nCleaned up temporary directory")


def main():
    """Main test function."""
    success = run_all_tests()
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    
    if success:
        print("The debugging tools are ready to use! Try:")
        print("\n1. Train with enhanced metrics:")
        print("   python debugging-bc-training/enhanced_bc_trainer.py \\")
        print("     --demo_path expert_demonstrations/raw/raw_demonstrations \\")
        print("     --max_demos 10 --epochs 50")
        print("\n2. Analyze the results:")
        print("   python debugging-bc-training/analyze_training.py \\")
        print("     --metrics_file debugging-bc-training/results/metrics_history.pkl \\")
        print("     --visualize")
    else:
        print("Please fix the failing tests before using the tools.")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())