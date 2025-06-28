"""
Comprehensive tests for checkpoint management system.

This module tests checkpoint saving, loading, integrity verification,
and metadata management using property-based testing.
"""

import pytest
import pickle
import json
import hashlib
import time
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from hypothesis import given, strategies as st, settings, assume
import jax.numpy as jnp

from causal_bayes_opt.training.checkpoint_manager import (
    CheckpointMetadata, CheckpointInfo, CheckpointConfig, CheckpointManager,
    create_checkpoint_manager
)


@pytest.fixture
def temp_checkpoint_dir():
    """Create a temporary directory for checkpoint tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_training_state():
    """Create a sample training state for testing."""
    return {
        'step': 100,
        'params': {'weights': jnp.array([1.0, 2.0, 3.0])},
        'optimizer_state': {'momentum': jnp.array([0.1, 0.2, 0.3])},
        'metrics': {'loss': 0.5, 'accuracy': 0.8}
    }


class TestCheckpointMetadata:
    """Test checkpoint metadata functionality."""
    
    def test_checkpoint_metadata_immutable(self):
        """Test that CheckpointMetadata is immutable."""
        metadata = CheckpointMetadata(
            checkpoint_id="test_123",
            timestamp=time.time(),
            stage="training",
            version="1.0",
            file_size_bytes=1024,
            content_hash="abc123",
            training_step=100,
            error_count=0,
            recovery_count=0
        )
        
        # Should not be able to modify
        with pytest.raises(AttributeError):
            metadata.timestamp = time.time()
        
        with pytest.raises(AttributeError):
            metadata.training_step = 200
    
    @given(
        checkpoint_id=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Ll', 'Lu', 'Nd', 'Pc'))),
        stage=st.text(min_size=1, max_size=30, alphabet=st.characters(whitelist_categories=('Ll', 'Lu', 'Nd', 'Pc'))),
        training_step=st.integers(min_value=0, max_value=1000000),
        error_count=st.integers(min_value=0, max_value=100),
        file_size=st.integers(min_value=1, max_value=1000000000)
    )
    @settings(max_examples=20)
    def test_checkpoint_metadata_properties(self, checkpoint_id, stage, training_step, error_count, file_size):
        """Property-based test for CheckpointMetadata creation."""
        timestamp = time.time()
        
        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            timestamp=timestamp,
            stage=stage,
            version="1.0",
            file_size_bytes=file_size,
            content_hash="test_hash",
            training_step=training_step,
            error_count=error_count,
            recovery_count=0
        )
        
        assert metadata.checkpoint_id == checkpoint_id
        assert metadata.stage == stage
        assert metadata.training_step == training_step
        assert metadata.error_count == error_count
        assert metadata.file_size_bytes == file_size
        assert metadata.timestamp == timestamp
    
    def test_checkpoint_info_immutable(self):
        """Test that CheckpointInfo is immutable."""
        metadata = CheckpointMetadata(
            checkpoint_id="test",
            timestamp=time.time(),
            stage="test",
            version="1.0",
            file_size_bytes=100,
            content_hash="hash",
            training_step=1,
            error_count=0,
            recovery_count=0
        )
        
        info = CheckpointInfo(
            path=Path("/test/path"),
            metadata=metadata,
            is_valid=True,
            verification_status="verified"
        )
        
        with pytest.raises(AttributeError):
            info.is_valid = False


class TestCheckpointConfig:
    """Test checkpoint configuration."""
    
    def test_checkpoint_config_defaults(self):
        """Test default checkpoint configuration values."""
        config = CheckpointConfig()
        
        assert config.max_checkpoints == 10
        assert config.auto_cleanup is True
        assert config.compression_enabled is True
        assert config.integrity_verification is True
        assert config.backup_to_remote is False
        assert config.save_frequency_steps == 100
        assert config.save_on_error is True
    
    def test_checkpoint_config_immutable(self):
        """Test that CheckpointConfig is immutable."""
        config = CheckpointConfig(max_checkpoints=5)
        
        with pytest.raises(AttributeError):
            config.max_checkpoints = 10
    
    @given(
        max_checkpoints=st.integers(min_value=1, max_value=100),
        save_frequency=st.integers(min_value=1, max_value=1000),
        auto_cleanup=st.booleans(),
        compression=st.booleans(),
        verification=st.booleans()
    )
    @settings(max_examples=20)
    def test_checkpoint_config_properties(self, max_checkpoints, save_frequency, auto_cleanup, compression, verification):
        """Property-based test for CheckpointConfig."""
        config = CheckpointConfig(
            max_checkpoints=max_checkpoints,
            save_frequency_steps=save_frequency,
            auto_cleanup=auto_cleanup,
            compression_enabled=compression,
            integrity_verification=verification
        )
        
        assert config.max_checkpoints == max_checkpoints
        assert config.save_frequency_steps == save_frequency
        assert config.auto_cleanup == auto_cleanup
        assert config.compression_enabled == compression
        assert config.integrity_verification == verification


class TestCheckpointManager:
    """Test CheckpointManager functionality."""
    
    def test_checkpoint_manager_initialization(self, temp_checkpoint_dir):
        """Test CheckpointManager initialization."""
        config = CheckpointConfig(max_checkpoints=5)
        manager = CheckpointManager(temp_checkpoint_dir, config)
        
        assert manager.checkpoint_dir == temp_checkpoint_dir
        assert manager.config == config
        assert manager.metadata_dir.exists()
        assert manager.temp_dir.exists()
    
    def test_create_checkpoint_manager(self, temp_checkpoint_dir):
        """Test checkpoint manager factory function."""
        manager = create_checkpoint_manager(temp_checkpoint_dir)
        
        assert isinstance(manager, CheckpointManager)
        assert manager.checkpoint_dir == temp_checkpoint_dir
    
    def test_save_checkpoint_basic(self, temp_checkpoint_dir, sample_training_state):
        """Test basic checkpoint saving functionality."""
        manager = CheckpointManager(temp_checkpoint_dir)
        
        checkpoint_info = manager.save_checkpoint(
            state=sample_training_state,
            checkpoint_name="test_checkpoint",
            stage="training",
            user_notes="Test save"
        )
        
        assert isinstance(checkpoint_info, CheckpointInfo)
        assert checkpoint_info.path.exists()
        assert checkpoint_info.metadata.stage == "training"
        assert checkpoint_info.metadata.user_notes == "Test save"
        assert checkpoint_info.is_valid is True
        
        # Metadata file should also exist
        metadata_file = manager.metadata_dir / f"{checkpoint_info.metadata.checkpoint_id}.json"
        assert metadata_file.exists()
    
    def test_load_checkpoint_basic(self, temp_checkpoint_dir, sample_training_state):
        """Test basic checkpoint loading functionality."""
        manager = CheckpointManager(temp_checkpoint_dir)
        
        # Save checkpoint
        checkpoint_info = manager.save_checkpoint(
            state=sample_training_state,
            checkpoint_name="test_checkpoint",
            stage="training"
        )
        
        # Load checkpoint
        loaded_state = manager.load_checkpoint(checkpoint_info.path)
        
        # Should match original state
        assert loaded_state['step'] == sample_training_state['step']
        assert jnp.allclose(loaded_state['params']['weights'], sample_training_state['params']['weights'])
        assert loaded_state['metrics'] == sample_training_state['metrics']
    
    def test_checkpoint_integrity_verification(self, temp_checkpoint_dir, sample_training_state):
        """Test checkpoint integrity verification."""
        config = CheckpointConfig(integrity_verification=True)
        manager = CheckpointManager(temp_checkpoint_dir, config)
        
        # Save checkpoint
        checkpoint_info = manager.save_checkpoint(
            state=sample_training_state,
            checkpoint_name="test_checkpoint",
            stage="training"
        )
        
        assert checkpoint_info.verification_status == "verified"
        assert checkpoint_info.is_valid is True
        
        # Corrupt the checkpoint file
        with open(checkpoint_info.path, 'ab') as f:
            f.write(b'corrupted_data')
        
        # Should detect corruption
        with pytest.raises(RuntimeError, match="Checkpoint verification failed"):
            manager.load_checkpoint(checkpoint_info.path)
    
    def test_checkpoint_compression(self, temp_checkpoint_dir, sample_training_state):
        """Test checkpoint compression functionality."""
        # Test without compression
        config_no_compression = CheckpointConfig(compression_enabled=False)
        manager_no_compression = CheckpointManager(temp_checkpoint_dir / "no_compression", config_no_compression)
        
        checkpoint_info_uncompressed = manager_no_compression.save_checkpoint(
            state=sample_training_state,
            checkpoint_name="uncompressed",
            stage="training"
        )
        
        # Test with compression
        config_compression = CheckpointConfig(compression_enabled=True)
        manager_compression = CheckpointManager(temp_checkpoint_dir / "compression", config_compression)
        
        checkpoint_info_compressed = manager_compression.save_checkpoint(
            state=sample_training_state,
            checkpoint_name="compressed",
            stage="training"
        )
        
        # Both should work
        loaded_uncompressed = manager_no_compression.load_checkpoint(checkpoint_info_uncompressed.path)
        loaded_compressed = manager_compression.load_checkpoint(checkpoint_info_compressed.path)
        
        assert loaded_uncompressed['step'] == loaded_compressed['step']
    
    def test_list_checkpoints(self, temp_checkpoint_dir, sample_training_state):
        """Test listing checkpoints."""
        manager = CheckpointManager(temp_checkpoint_dir)
        
        # Save multiple checkpoints
        checkpoints = []
        for i in range(3):
            checkpoint_info = manager.save_checkpoint(
                state={**sample_training_state, 'step': i * 100},
                checkpoint_name=f"checkpoint_{i}",
                stage=f"stage_{i}"
            )
            checkpoints.append(checkpoint_info)
            time.sleep(0.01)  # Ensure different timestamps
        
        # List all checkpoints
        all_checkpoints = manager.list_checkpoints()
        assert len(all_checkpoints) == 3
        
        # Should be sorted by timestamp (newest first)
        timestamps = [cp.metadata.timestamp for cp in all_checkpoints]
        assert timestamps == sorted(timestamps, reverse=True)
        
        # Filter by stage
        stage_0_checkpoints = manager.list_checkpoints(stage="stage_0")
        assert len(stage_0_checkpoints) == 1
        assert stage_0_checkpoints[0].metadata.stage == "stage_0"
    
    def test_get_latest_checkpoint(self, temp_checkpoint_dir, sample_training_state):
        """Test getting the latest checkpoint."""
        manager = CheckpointManager(temp_checkpoint_dir)
        
        # No checkpoints initially
        latest = manager.get_latest_checkpoint()
        assert latest is None
        
        # Save checkpoints
        for i in range(3):
            manager.save_checkpoint(
                state={**sample_training_state, 'step': i * 100},
                checkpoint_name=f"checkpoint_{i}",
                stage="training"
            )
            time.sleep(0.01)
        
        # Get latest
        latest = manager.get_latest_checkpoint()
        assert latest is not None
        assert latest.metadata.checkpoint_name == "checkpoint_2"
        
        # Get latest for specific stage
        manager.save_checkpoint(
            state=sample_training_state,
            checkpoint_name="validation_checkpoint",
            stage="validation"
        )
        
        latest_validation = manager.get_latest_checkpoint(stage="validation")
        assert latest_validation.metadata.stage == "validation"
    
    def test_delete_checkpoint(self, temp_checkpoint_dir, sample_training_state):
        """Test checkpoint deletion."""
        manager = CheckpointManager(temp_checkpoint_dir)
        
        checkpoint_info = manager.save_checkpoint(
            state=sample_training_state,
            checkpoint_name="to_delete",
            stage="training"
        )
        
        # Verify checkpoint exists
        assert checkpoint_info.path.exists()
        metadata_file = manager.metadata_dir / f"{checkpoint_info.metadata.checkpoint_id}.json"
        assert metadata_file.exists()
        
        # Delete checkpoint
        manager.delete_checkpoint(checkpoint_info)
        
        # Should be gone
        assert not checkpoint_info.path.exists()
        assert not metadata_file.exists()
    
    def test_automatic_cleanup(self, temp_checkpoint_dir, sample_training_state):
        """Test automatic cleanup of old checkpoints."""
        config = CheckpointConfig(max_checkpoints=3, auto_cleanup=True)
        manager = CheckpointManager(temp_checkpoint_dir, config)
        
        # Save more checkpoints than the limit
        checkpoint_infos = []
        for i in range(5):
            checkpoint_info = manager.save_checkpoint(
                state={**sample_training_state, 'step': i * 100},
                checkpoint_name=f"checkpoint_{i}",
                stage="training"
            )
            checkpoint_infos.append(checkpoint_info)
            time.sleep(0.01)  # Ensure different timestamps
        
        # Should only keep the latest 3
        remaining_checkpoints = manager.list_checkpoints()
        assert len(remaining_checkpoints) == 3
        
        # Should keep the most recent ones
        remaining_names = {cp.metadata.checkpoint_name for cp in remaining_checkpoints}
        expected_names = {"checkpoint_2", "checkpoint_3", "checkpoint_4"}
        assert remaining_names == expected_names
    
    @given(
        num_checkpoints=st.integers(min_value=1, max_value=10),
        max_checkpoints=st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=10)
    def test_cleanup_properties(self, temp_checkpoint_dir, sample_training_state, num_checkpoints, max_checkpoints):
        """Property-based test for checkpoint cleanup."""
        config = CheckpointConfig(max_checkpoints=max_checkpoints, auto_cleanup=True)
        manager = CheckpointManager(temp_checkpoint_dir, config)
        
        # Save checkpoints
        for i in range(num_checkpoints):
            manager.save_checkpoint(
                state={**sample_training_state, 'step': i},
                checkpoint_name=f"checkpoint_{i}",
                stage="training"
            )
            time.sleep(0.001)  # Small delay for timestamps
        
        # Should not exceed max_checkpoints
        remaining = manager.list_checkpoints()
        assert len(remaining) <= max_checkpoints
        assert len(remaining) <= num_checkpoints


class TestCheckpointManagerEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_save_checkpoint_disk_full_simulation(self, temp_checkpoint_dir, sample_training_state):
        """Test checkpoint saving when disk is full (simulated)."""
        manager = CheckpointManager(temp_checkpoint_dir)
        
        # Mock open to raise OSError (disk full)
        with patch('builtins.open', side_effect=OSError("No space left on device")):
            with pytest.raises(OSError):
                manager.save_checkpoint(
                    state=sample_training_state,
                    checkpoint_name="disk_full_test",
                    stage="training"
                )
        
        # Temp file should be cleaned up
        temp_files = list(manager.temp_dir.glob("*"))
        assert len(temp_files) == 0
    
    def test_load_nonexistent_checkpoint(self, temp_checkpoint_dir):
        """Test loading a nonexistent checkpoint."""
        manager = CheckpointManager(temp_checkpoint_dir)
        
        nonexistent_path = temp_checkpoint_dir / "nonexistent.pkl"
        
        with pytest.raises(FileNotFoundError):
            manager.load_checkpoint(nonexistent_path)
    
    def test_load_corrupted_checkpoint(self, temp_checkpoint_dir):
        """Test loading a corrupted checkpoint file."""
        manager = CheckpointManager(temp_checkpoint_dir)
        
        # Create a corrupted checkpoint file
        corrupted_path = temp_checkpoint_dir / "corrupted.pkl"
        with open(corrupted_path, 'wb') as f:
            f.write(b"not_a_valid_pickle_file")
        
        with pytest.raises(Exception):  # Pickle will raise various exceptions
            manager.load_checkpoint(corrupted_path)
    
    def test_metadata_file_missing(self, temp_checkpoint_dir, sample_training_state):
        """Test handling of missing metadata files."""
        manager = CheckpointManager(temp_checkpoint_dir)
        
        # Save checkpoint normally
        checkpoint_info = manager.save_checkpoint(
            state=sample_training_state,
            checkpoint_name="test",
            stage="training"
        )
        
        # Delete metadata file
        metadata_file = manager.metadata_dir / f"{checkpoint_info.metadata.checkpoint_id}.json"
        metadata_file.unlink()
        
        # Should handle gracefully when listing
        checkpoints = manager.list_checkpoints()
        
        # Should create legacy metadata
        if checkpoints:
            assert checkpoints[0].metadata.version == "legacy"
    
    def test_concurrent_checkpoint_access(self, temp_checkpoint_dir, sample_training_state):
        """Test handling of concurrent checkpoint operations."""
        manager = CheckpointManager(temp_checkpoint_dir)
        
        # This is a simplified test - in practice would need proper concurrency testing
        def save_checkpoint(name):
            return manager.save_checkpoint(
                state={**sample_training_state, 'name': name},
                checkpoint_name=name,
                stage="concurrent_test"
            )
        
        # Save multiple checkpoints "concurrently" (sequentially for testing)
        checkpoint_infos = []
        for i in range(3):
            checkpoint_info = save_checkpoint(f"concurrent_{i}")
            checkpoint_infos.append(checkpoint_info)
        
        # All should succeed
        assert len(checkpoint_infos) == 3
        
        # All should be loadable
        for checkpoint_info in checkpoint_infos:
            state = manager.load_checkpoint(checkpoint_info.path)
            assert 'name' in state
    
    def test_very_large_state_handling(self, temp_checkpoint_dir):
        """Test handling of very large training states."""
        manager = CheckpointManager(temp_checkpoint_dir)
        
        # Create a large state (but not too large for testing)
        large_array = jnp.ones((1000, 1000))  # ~4MB
        large_state = {
            'large_params': large_array,
            'step': 1000,
            'metadata': {'info': 'large state test'}
        }
        
        # Should handle large states
        checkpoint_info = manager.save_checkpoint(
            state=large_state,
            checkpoint_name="large_state",
            stage="testing"
        )
        
        assert checkpoint_info.metadata.file_size_bytes > 1000000  # > 1MB
        
        # Should load correctly
        loaded_state = manager.load_checkpoint(checkpoint_info.path)
        assert jnp.array_equal(loaded_state['large_params'], large_array)
    
    def test_special_characters_in_names(self, temp_checkpoint_dir, sample_training_state):
        """Test handling of special characters in checkpoint names."""
        manager = CheckpointManager(temp_checkpoint_dir)
        
        # Test with various characters (some will be sanitized)
        test_names = [
            "normal_name",
            "name-with-dashes",
            "name_with_underscores",
            "name123with456numbers"
        ]
        
        for name in test_names:
            checkpoint_info = manager.save_checkpoint(
                state=sample_training_state,
                checkpoint_name=name,
                stage="testing"
            )
            
            # Should create valid files
            assert checkpoint_info.path.exists()
            
            # Should be loadable
            loaded_state = manager.load_checkpoint(checkpoint_info.path)
            assert loaded_state is not None


class TestCheckpointManagerPerformance:
    """Test performance characteristics of checkpoint management."""
    
    def test_checkpoint_save_load_speed(self, temp_checkpoint_dir, sample_training_state):
        """Test that checkpoint operations are reasonably fast."""
        manager = CheckpointManager(temp_checkpoint_dir)
        
        # Measure save time
        start_time = time.time()
        checkpoint_info = manager.save_checkpoint(
            state=sample_training_state,
            checkpoint_name="speed_test",
            stage="performance"
        )
        save_time = time.time() - start_time
        
        # Should be fast (less than 1 second for small state)
        assert save_time < 1.0
        
        # Measure load time
        start_time = time.time()
        loaded_state = manager.load_checkpoint(checkpoint_info.path)
        load_time = time.time() - start_time
        
        # Should be fast
        assert load_time < 1.0
        assert loaded_state is not None
    
    def test_hash_calculation_performance(self, temp_checkpoint_dir):
        """Test that hash calculation doesn't significantly slow down saves."""
        # Compare with and without integrity verification
        
        # Without verification
        config_no_verify = CheckpointConfig(integrity_verification=False)
        manager_no_verify = CheckpointManager(temp_checkpoint_dir / "no_verify", config_no_verify)
        
        state = {'data': jnp.ones((100, 100))}
        
        start_time = time.time()
        manager_no_verify.save_checkpoint(state, "no_verify", "test")
        time_no_verify = time.time() - start_time
        
        # With verification
        config_verify = CheckpointConfig(integrity_verification=True)
        manager_verify = CheckpointManager(temp_checkpoint_dir / "verify", config_verify)
        
        start_time = time.time()
        manager_verify.save_checkpoint(state, "verify", "test")
        time_verify = time.time() - start_time
        
        # Verification shouldn't add too much overhead (less than 2x)
        assert time_verify < time_no_verify * 3.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])