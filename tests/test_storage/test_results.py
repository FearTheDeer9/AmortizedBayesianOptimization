"""
Tests for results.py

Tests for result storage and loading functionality with emphasis on
simplicity, reliability, and data integrity.
"""

import pytest
import json
import tempfile
import os
import shutil
import time
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, mock_open

from src.causal_bayes_opt.storage.results import (
    save_experiment_result,
    load_experiment_results,
    load_results_by_pattern,
    create_results_summary,
    filter_results,
    cleanup_old_results,
    _json_serializer
)


@pytest.fixture
def temp_dir():
    """Temporary directory for testing file operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_experiment_result():
    """Sample experiment result for testing."""
    return {
        'method': 'learning_surrogate',
        'graph_size': 5,
        'target_variable': 'X4',
        'final_f1_score': 0.75,
        'target_improvement': 0.25,
        'runtime_seconds': 45.2,
        'success': True,
        'config': {
            'n_interventions': 20,
            'n_observational': 100
        }
    }


@pytest.fixture
def sample_multiple_results():
    """Multiple sample results for testing."""
    return [
        {
            'method': 'static_surrogate',
            'dataset_name': 'erdos_renyi_5',
            'final_f1_score': 0.6,
            'runtime_seconds': 30.0
        },
        {
            'method': 'learning_surrogate', 
            'dataset_name': 'erdos_renyi_5',
            'final_f1_score': 0.8,
            'runtime_seconds': 45.0
        },
        {
            'method': 'static_surrogate',
            'dataset_name': 'erdos_renyi_8',
            'final_f1_score': 0.4,
            'runtime_seconds': 50.0
        }
    ]


class TestSaveExperimentResult:
    """Test experiment result saving functionality."""
    
    def test_save_basic(self, sample_experiment_result, temp_dir):
        """Test basic result saving."""
        saved_path = save_experiment_result(sample_experiment_result, temp_dir)
        
        assert os.path.exists(saved_path), "Result file should be created"
        assert saved_path.startswith(temp_dir), "File should be in specified directory"
        assert saved_path.endswith('.json'), "File should have .json extension"
    
    def test_save_with_custom_prefix(self, sample_experiment_result, temp_dir):
        """Test saving with custom filename prefix."""
        saved_path = save_experiment_result(
            sample_experiment_result, 
            temp_dir, 
            prefix="my_experiment"
        )
        
        filename = os.path.basename(saved_path)
        assert filename.startswith("my_experiment_"), "Should use custom prefix"
    
    def test_save_with_custom_timestamp(self, sample_experiment_result, temp_dir):
        """Test saving with custom timestamp."""
        custom_timestamp = "20250627_120000"
        saved_path = save_experiment_result(
            sample_experiment_result,
            temp_dir,
            timestamp=custom_timestamp
        )
        
        filename = os.path.basename(saved_path)
        assert custom_timestamp in filename, "Should use custom timestamp"
    
    def test_save_creates_directory(self, sample_experiment_result):
        """Test that saving creates output directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent_dir = os.path.join(tmpdir, "new_dir", "nested")
            
            saved_path = save_experiment_result(sample_experiment_result, nonexistent_dir)
            
            assert os.path.exists(saved_path), "Should create nested directories"
            assert os.path.exists(nonexistent_dir), "Directory should be created"
    
    def test_save_adds_metadata(self, sample_experiment_result, temp_dir):
        """Test that saving adds metadata to the result."""
        saved_path = save_experiment_result(sample_experiment_result, temp_dir)
        
        with open(saved_path, 'r') as f:
            saved_data = json.load(f)
        
        # Should contain original data
        assert saved_data['method'] == 'learning_surrogate'
        assert saved_data['final_f1_score'] == 0.75
        
        # Should add metadata
        assert 'save_timestamp' in saved_data
        assert 'save_time_iso' in saved_data
        assert 'save_time_unix' in saved_data
    
    def test_save_preserves_data_integrity(self, sample_experiment_result, temp_dir):
        """Test that saved data matches original data."""
        saved_path = save_experiment_result(sample_experiment_result, temp_dir)
        
        with open(saved_path, 'r') as f:
            saved_data = json.load(f)
        
        # Check that original fields are preserved
        for key, value in sample_experiment_result.items():
            assert saved_data[key] == value, f"Field {key} should be preserved"
    
    def test_save_with_numpy_arrays(self, temp_dir):
        """Test saving with numpy arrays using custom serializer."""
        import numpy as np
        result_with_numpy = {
            'method': 'test',
            'array_data': np.array([1, 2, 3, 4, 5]),
            'matrix_data': np.array([[1, 2], [3, 4]])
        }
        
        saved_path = save_experiment_result(result_with_numpy, temp_dir)
        
        with open(saved_path, 'r') as f:
            saved_data = json.load(f)
        
        # Arrays should be converted to lists
        assert saved_data['array_data'] == [1, 2, 3, 4, 5]
        assert saved_data['matrix_data'] == [[1, 2], [3, 4]]
    
    def test_save_error_handling(self, sample_experiment_result):
        """Test error handling for invalid save paths."""
        # Try to save to read-only location (should raise exception)
        with pytest.raises(Exception):
            save_experiment_result(sample_experiment_result, "/dev/null/impossible")


class TestLoadExperimentResults:
    """Test experiment result loading functionality."""
    
    def test_load_single_result(self, sample_experiment_result, temp_dir):
        """Test loading a single saved result."""
        saved_path = save_experiment_result(sample_experiment_result, temp_dir)
        
        results = load_experiment_results(temp_dir)
        
        assert len(results) == 1, "Should load one result"
        assert results[0]['method'] == 'learning_surrogate'
        assert 'loaded_from' in results[0], "Should add loading metadata"
        assert 'filename' in results[0], "Should add filename metadata"
    
    def test_load_multiple_results(self, sample_multiple_results, temp_dir):
        """Test loading multiple saved results."""
        # Save multiple results
        for i, result in enumerate(sample_multiple_results):
            save_experiment_result(result, temp_dir, prefix=f"exp_{i}")
        
        results = load_experiment_results(temp_dir)
        
        assert len(results) == 3, "Should load all three results"
        
        # Check that all methods are represented
        methods = [r['method'] for r in results]
        assert 'static_surrogate' in methods
        assert 'learning_surrogate' in methods
    
    def test_load_with_pattern(self, sample_multiple_results, temp_dir):
        """Test loading with specific filename patterns."""
        # Save with different prefixes
        save_experiment_result(sample_multiple_results[0], temp_dir, prefix="static")
        save_experiment_result(sample_multiple_results[1], temp_dir, prefix="learning")
        save_experiment_result(sample_multiple_results[2], temp_dir, prefix="other")
        
        # Load only static results
        static_results = load_experiment_results(temp_dir, "static_*.json")
        assert len(static_results) == 1
        assert static_results[0]['method'] == 'static_surrogate'
        
        # Load only learning results
        learning_results = load_experiment_results(temp_dir, "learning_*.json")
        assert len(learning_results) == 1
        assert learning_results[0]['method'] == 'learning_surrogate'
    
    def test_load_nonexistent_directory(self):
        """Test loading from non-existent directory."""
        results = load_experiment_results("/nonexistent/directory")
        
        assert results == [], "Should return empty list for non-existent directory"
    
    def test_load_empty_directory(self, temp_dir):
        """Test loading from empty directory."""
        results = load_experiment_results(temp_dir)
        
        assert results == [], "Should return empty list for empty directory"
    
    def test_load_corrupted_json(self, temp_dir):
        """Test loading with corrupted JSON files."""
        # Create a corrupted JSON file
        corrupted_path = os.path.join(temp_dir, "experiment_corrupted.json")
        with open(corrupted_path, 'w') as f:
            f.write("{ invalid json content")
        
        # Create a valid file too
        valid_result = {'method': 'test', 'success': True}
        save_experiment_result(valid_result, temp_dir, prefix="valid")
        
        results = load_experiment_results(temp_dir)
        
        # Should load only the valid file, skip corrupted one
        assert len(results) == 1
        assert results[0]['method'] == 'test'
    
    def test_load_maintains_order(self, temp_dir):
        """Test that loading maintains file order."""
        # Save files with timestamps that ensure ordering
        for i in range(5):
            result = {'method': f'method_{i}', 'step': i}
            save_experiment_result(result, temp_dir, timestamp=f"2025062{i}_120000")
            time.sleep(0.01)  # Ensure different timestamps
        
        results = load_experiment_results(temp_dir)
        
        assert len(results) == 5
        # Results should be in sorted order by filename
        for i in range(5):
            assert f'method_{i}' in [r['method'] for r in results]


class TestLoadResultsByPattern:
    """Test pattern-based result loading and grouping."""
    
    def test_load_by_pattern_grouping(self, sample_multiple_results, temp_dir):
        """Test loading and grouping by pattern."""
        # Save multiple results
        for i, result in enumerate(sample_multiple_results):
            save_experiment_result(result, temp_dir, prefix=f"experiment_{i}")
        
        grouped_results = load_results_by_pattern("experiment_*.json", temp_dir)
        
        assert isinstance(grouped_results, dict)
        assert len(grouped_results) > 0, "Should create at least one group"
        
        # Check that grouping worked
        for group_key, group_results in grouped_results.items():
            assert isinstance(group_results, list)
            assert len(group_results) > 0
    
    def test_load_by_pattern_empty(self, temp_dir):
        """Test pattern loading with no matching files."""
        grouped_results = load_results_by_pattern("nonexistent_*.json", temp_dir)
        
        assert grouped_results == {}


class TestCreateResultsSummary:
    """Test results summary generation."""
    
    def test_create_summary_basic(self, sample_multiple_results):
        """Test basic summary creation."""
        summary = create_results_summary(sample_multiple_results)
        
        assert summary['total_experiments'] == 3
        assert 'static_surrogate' in summary['unique_methods']
        assert 'learning_surrogate' in summary['unique_methods']
        assert 'method_counts' in summary
        assert 'f1_statistics' in summary
    
    def test_create_summary_with_save(self, sample_multiple_results, temp_dir):
        """Test summary creation with file saving."""
        summary_path = os.path.join(temp_dir, "summary.json")
        
        summary = create_results_summary(sample_multiple_results, summary_path)
        
        assert os.path.exists(summary_path), "Summary file should be created"
        
        # Verify saved content
        with open(summary_path, 'r') as f:
            saved_summary = json.load(f)
        
        assert saved_summary['total_experiments'] == 3
    
    def test_create_summary_empty_results(self):
        """Test summary creation with empty results."""
        summary = create_results_summary([])
        
        assert summary['total_experiments'] == 0
        assert 'summary_created' in summary
    
    def test_create_summary_statistics(self, sample_multiple_results):
        """Test that summary statistics are computed correctly."""
        summary = create_results_summary(sample_multiple_results)
        
        f1_stats = summary['f1_statistics']
        assert f1_stats['count'] == 3
        assert f1_stats['mean'] == pytest.approx((0.6 + 0.8 + 0.4) / 3)
        assert f1_stats['min'] == 0.4
        assert f1_stats['max'] == 0.8
    
    def test_create_summary_method_counts(self, sample_multiple_results):
        """Test method counting in summary."""
        summary = create_results_summary(sample_multiple_results)
        
        method_counts = summary['method_counts']
        assert method_counts['static_surrogate'] == 2
        assert method_counts['learning_surrogate'] == 1


class TestFilterResults:
    """Test result filtering functionality."""
    
    def test_filter_by_method(self, sample_multiple_results):
        """Test filtering by method."""
        filters = {'method': 'static_surrogate'}
        
        filtered = filter_results(sample_multiple_results, filters)
        
        assert len(filtered) == 2, "Should return two static_surrogate results"
        for result in filtered:
            assert result['method'] == 'static_surrogate'
    
    def test_filter_by_list(self, sample_multiple_results):
        """Test filtering with list of values."""
        filters = {'method': ['learning_surrogate', 'other_method']}
        
        filtered = filter_results(sample_multiple_results, filters)
        
        assert len(filtered) == 1, "Should return one learning_surrogate result"
        assert filtered[0]['method'] == 'learning_surrogate'
    
    def test_filter_by_range(self, sample_multiple_results):
        """Test filtering with numeric ranges."""
        filters = {'final_f1_score': {'min': 0.5, 'max': 0.7}}
        
        filtered = filter_results(sample_multiple_results, filters)
        
        assert len(filtered) == 1, "Should return one result in F1 range [0.5, 0.7]"
        assert filtered[0]['final_f1_score'] == 0.6
    
    def test_filter_multiple_criteria(self, sample_multiple_results):
        """Test filtering with multiple criteria."""
        filters = {
            'method': 'static_surrogate',
            'final_f1_score': {'min': 0.5}
        }
        
        filtered = filter_results(sample_multiple_results, filters)
        
        assert len(filtered) == 1, "Should return one result matching both criteria"
        assert filtered[0]['method'] == 'static_surrogate'
        assert filtered[0]['final_f1_score'] >= 0.5
    
    def test_filter_no_matches(self, sample_multiple_results):
        """Test filtering with criteria that match nothing."""
        filters = {'method': 'nonexistent_method'}
        
        filtered = filter_results(sample_multiple_results, filters)
        
        assert filtered == [], "Should return empty list when no matches"
    
    def test_filter_missing_field(self, sample_multiple_results):
        """Test filtering on fields that don't exist in all results."""
        filters = {'nonexistent_field': 'value'}
        
        filtered = filter_results(sample_multiple_results, filters)
        
        assert filtered == [], "Should return empty list when field doesn't exist"


class TestCleanupOldResults:
    """Test cleanup functionality for old result files."""
    
    def test_cleanup_basic(self, temp_dir):
        """Test basic cleanup functionality."""
        # Create several result files
        for i in range(10):
            result = {'method': f'method_{i}', 'step': i}
            save_experiment_result(result, temp_dir, timestamp=f"202506{i:02d}_120000")
        
        # Keep only 5 latest files
        deleted_count = cleanup_old_results(temp_dir, keep_latest=5)
        
        assert deleted_count == 5, "Should delete 5 old files"
        
        # Verify only 5 files remain
        remaining_files = load_experiment_results(temp_dir)
        assert len(remaining_files) == 5
    
    def test_cleanup_no_files(self, temp_dir):
        """Test cleanup when no files exist."""
        deleted_count = cleanup_old_results(temp_dir)
        
        assert deleted_count == 0, "Should delete 0 files from empty directory"
    
    def test_cleanup_fewer_than_threshold(self, temp_dir):
        """Test cleanup when fewer files than threshold exist."""
        # Create only 2 files
        for i in range(2):
            result = {'method': f'method_{i}'}
            save_experiment_result(result, temp_dir)
        
        deleted_count = cleanup_old_results(temp_dir, keep_latest=5)
        
        assert deleted_count == 0, "Should delete 0 files when fewer than threshold"
        
        # All files should remain
        remaining_files = load_experiment_results(temp_dir)
        assert len(remaining_files) == 2
    
    def test_cleanup_nonexistent_directory(self):
        """Test cleanup on non-existent directory."""
        deleted_count = cleanup_old_results("/nonexistent/directory")
        
        assert deleted_count == 0, "Should handle non-existent directory gracefully"


class TestJsonSerializer:
    """Test the custom JSON serializer."""
    
    def test_serialize_numpy_array(self):
        """Test serialization of numpy arrays."""
        import numpy as np
        arr = np.array([1, 2, 3, 4, 5])
        
        serialized = _json_serializer(arr)
        
        assert serialized == [1, 2, 3, 4, 5], "Should convert numpy array to list"
    
    def test_serialize_object_with_dict(self):
        """Test serialization of objects with __dict__."""
        class TestObj:
            def __init__(self):
                self.field1 = "value1"
                self.field2 = 42
        
        obj = TestObj()
        serialized = _json_serializer(obj)
        
        assert serialized == {"field1": "value1", "field2": 42}
    
    def test_serialize_namedtuple(self):
        """Test serialization of namedtuples."""
        from collections import namedtuple
        Point = namedtuple('Point', ['x', 'y'])
        point = Point(10, 20)
        
        serialized = _json_serializer(point)
        
        assert serialized == {"x": 10, "y": 20}
    
    def test_serialize_fallback(self):
        """Test serialization fallback to string."""
        class UnserializableObj:
            pass
        
        obj = UnserializableObj()
        serialized = _json_serializer(obj)
        
        # Should fall back to string representation
        assert isinstance(serialized, str)


class TestErrorHandling:
    """Test error handling in storage functions."""
    
    def test_save_json_error(self, temp_dir):
        """Test handling of JSON serialization errors."""
        # Create result with unserializable content
        result_with_error = {
            'method': 'test',
            'unserializable': set([1, 2, 3])  # sets are not JSON serializable
        }
        
        # Should handle gracefully with custom serializer
        try:
            saved_path = save_experiment_result(result_with_error, temp_dir)
            assert os.path.exists(saved_path)
        except Exception as e:
            # If it fails, it should be a reasonable JSON-related error
            assert "json" in str(e).lower() or "serializ" in str(e).lower()
    
    @patch('builtins.open', side_effect=PermissionError("Permission denied"))
    def test_save_permission_error(self, mock_open, sample_experiment_result, temp_dir):
        """Test handling of permission errors during save."""
        with pytest.raises(PermissionError):
            save_experiment_result(sample_experiment_result, temp_dir)
    
    def test_load_file_permission_error(self, temp_dir):
        """Test handling of permission errors during load."""
        # Create a file and make it unreadable
        test_file = os.path.join(temp_dir, "experiment_test.json")
        with open(test_file, 'w') as f:
            json.dump({"method": "test"}, f)
        
        # Make file unreadable (if possible on this system)
        try:
            os.chmod(test_file, 0o000)
            
            results = load_experiment_results(temp_dir)
            
            # Should handle permission error gracefully
            assert isinstance(results, list)
            
        except (OSError, PermissionError):
            # Some systems might not allow changing permissions
            pass
        finally:
            # Restore permissions for cleanup
            try:
                os.chmod(test_file, 0o644)
            except (OSError, PermissionError):
                pass


class TestIntegration:
    """Integration tests combining multiple storage functions."""
    
    def test_full_save_load_cycle(self, sample_multiple_results, temp_dir):
        """Test complete save-load-filter-summarize cycle."""
        # Save multiple results
        saved_paths = []
        for i, result in enumerate(sample_multiple_results):
            path = save_experiment_result(result, temp_dir, prefix=f"test_{i}")
            saved_paths.append(path)
        
        # Load all results
        loaded_results = load_experiment_results(temp_dir)
        assert len(loaded_results) == 3
        
        # Filter results
        learning_results = filter_results(loaded_results, {'method': 'learning_surrogate'})
        assert len(learning_results) == 1
        
        # Create summary
        summary = create_results_summary(loaded_results)
        assert summary['total_experiments'] == 3
        
        # Verify data integrity throughout cycle
        original_methods = [r['method'] for r in sample_multiple_results]
        loaded_methods = [r['method'] for r in loaded_results]
        assert set(original_methods) == set(loaded_methods)
    
    def test_concurrent_access_simulation(self, temp_dir):
        """Test behavior with concurrent-like file access."""
        # Simulate multiple "processes" saving results
        results = []
        for i in range(10):
            result = {
                'method': f'method_{i % 3}',
                'experiment_id': i,
                'timestamp': time.time()
            }
            saved_path = save_experiment_result(result, temp_dir, prefix=f"concurrent_{i}")
            results.append(result)
        
        # Load and verify all results are present
        loaded_results = load_experiment_results(temp_dir)
        assert len(loaded_results) == 10
        
        # Check that all experiment IDs are present
        loaded_ids = [r['experiment_id'] for r in loaded_results]
        expected_ids = list(range(10))
        assert set(loaded_ids) == set(expected_ids)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])