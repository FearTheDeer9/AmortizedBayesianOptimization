"""
Simple Results Storage

Functions for saving and loading experiment results as timestamped JSON files.
No complex schemas - just save everything and load on demand.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import glob

logger = logging.getLogger(__name__)


def save_experiment_result(
    result: Dict[str, Any],
    output_dir: str = "results",
    prefix: str = "experiment",
    timestamp: Optional[str] = None
) -> str:
    """
    Save experiment result to timestamped JSON file.
    
    Args:
        result: Experiment result dictionary to save
        output_dir: Directory to save results in
        prefix: Filename prefix
        timestamp: Optional custom timestamp (default: current time)
        
    Returns:
        Path to saved file
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create filename
    filename = f"{prefix}_{timestamp}.json"
    filepath = output_path / filename
    
    # Add metadata
    result_with_metadata = {
        **result,
        'save_timestamp': timestamp,
        'save_time_iso': datetime.now().isoformat(),
        'save_time_unix': time.time()
    }
    
    try:
        # Save as JSON with custom encoder for numpy arrays
        with open(filepath, 'w') as f:
            json.dump(result_with_metadata, f, indent=2, default=_json_serializer)
        
        logger.info(f"Saved experiment result to {filepath}")
        return str(filepath)
        
    except Exception as e:
        logger.error(f"Failed to save result to {filepath}: {e}")
        raise


def load_experiment_results(
    results_dir: str = "results",
    pattern: str = "experiment_*.json"
) -> List[Dict[str, Any]]:
    """
    Load all experiment results matching pattern.
    
    Args:
        results_dir: Directory containing results
        pattern: Glob pattern for matching files
        
    Returns:
        List of experiment result dictionaries
    """
    results_path = Path(results_dir)
    
    if not results_path.exists():
        logger.warning(f"Results directory {results_dir} does not exist")
        return []
    
    # Find matching files
    pattern_path = results_path / pattern
    result_files = glob.glob(str(pattern_path))
    
    if not result_files:
        logger.warning(f"No files found matching {pattern_path}")
        return []
    
    results = []
    
    for filepath in sorted(result_files):
        try:
            with open(filepath, 'r') as f:
                result = json.load(f)
            
            # Add file metadata
            result['loaded_from'] = filepath
            result['filename'] = Path(filepath).name
            
            results.append(result)
            
        except Exception as e:
            logger.warning(f"Failed to load result from {filepath}: {e}")
            continue
    
    logger.info(f"Loaded {len(results)} experiment results from {results_dir}")
    return results


def load_results_by_pattern(
    pattern: str,
    base_dir: str = "results"
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load results grouped by pattern matching.
    
    Args:
        pattern: Pattern to match (e.g., "*_method_*.json")
        base_dir: Base directory to search
        
    Returns:
        Dictionary mapping pattern matches to results
    """
    base_path = Path(base_dir)
    
    if not base_path.exists():
        logger.warning(f"Base directory {base_dir} does not exist")
        return {}
    
    # Find all matching files
    matching_files = glob.glob(str(base_path / pattern))
    
    if not matching_files:
        logger.warning(f"No files found matching pattern {pattern} in {base_dir}")
        return {}
    
    # Group by extracted key from filename
    grouped_results = {}
    
    for filepath in sorted(matching_files):
        try:
            with open(filepath, 'r') as f:
                result = json.load(f)
            
            # Extract grouping key from filename
            filename = Path(filepath).stem
            
            # Simple grouping by method if present in result
            method = result.get('method', 'unknown')
            dataset = result.get('dataset_name', 'unknown')
            group_key = f"{method}_{dataset}"
            
            if group_key not in grouped_results:
                grouped_results[group_key] = []
            
            result['loaded_from'] = filepath
            result['filename'] = Path(filepath).name
            grouped_results[group_key].append(result)
            
        except Exception as e:
            logger.warning(f"Failed to load result from {filepath}: {e}")
            continue
    
    logger.info(f"Loaded {sum(len(group) for group in grouped_results.values())} "
                f"results in {len(grouped_results)} groups")
    
    return grouped_results


def create_results_summary(
    results: List[Dict[str, Any]],
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a summary of experiment results.
    
    Args:
        results: List of experiment results
        save_path: Optional path to save summary
        
    Returns:
        Summary dictionary
    """
    if not results:
        summary = {
            'total_experiments': 0,
            'summary_created': datetime.now().isoformat()
        }
    else:
        # Basic statistics
        methods = [r.get('method', 'unknown') for r in results]
        datasets = [r.get('dataset_name', 'unknown') for r in results]
        f1_scores = [r.get('final_f1_score', 0) for r in results if r.get('final_f1_score') is not None]
        
        summary = {
            'total_experiments': len(results),
            'unique_methods': list(set(methods)),
            'unique_datasets': list(set(datasets)),
            'method_counts': {method: methods.count(method) for method in set(methods)},
            'dataset_counts': {dataset: datasets.count(dataset) for dataset in set(datasets)},
            'f1_statistics': {
                'count': len(f1_scores),
                'mean': float(sum(f1_scores) / len(f1_scores)) if f1_scores else 0,
                'min': float(min(f1_scores)) if f1_scores else 0,
                'max': float(max(f1_scores)) if f1_scores else 0
            },
            'time_range': {
                'earliest': min((r.get('save_time_iso') for r in results if r.get('save_time_iso')), default=None),
                'latest': max((r.get('save_time_iso') for r in results if r.get('save_time_iso')), default=None)
            },
            'summary_created': datetime.now().isoformat()
        }
    
    if save_path:
        try:
            with open(save_path, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Saved results summary to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save summary to {save_path}: {e}")
    
    return summary


def filter_results(
    results: List[Dict[str, Any]],
    filters: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Filter results based on criteria.
    
    Args:
        results: List of experiment results
        filters: Dictionary of filter criteria
        
    Returns:
        Filtered results
    """
    filtered = []
    
    for result in results:
        match = True
        
        for key, value in filters.items():
            if key not in result:
                match = False
                break
            
            result_value = result[key]
            
            # Handle different filter types
            if isinstance(value, list):
                # Value must be in list
                if result_value not in value:
                    match = False
                    break
            elif isinstance(value, dict):
                # Range filters
                if 'min' in value and result_value < value['min']:
                    match = False
                    break
                if 'max' in value and result_value > value['max']:
                    match = False
                    break
            else:
                # Exact match
                if result_value != value:
                    match = False
                    break
        
        if match:
            filtered.append(result)
    
    logger.info(f"Filtered {len(results)} results to {len(filtered)} matching criteria")
    return filtered


def cleanup_old_results(
    results_dir: str = "results",
    keep_latest: int = 100,
    pattern: str = "experiment_*.json"
) -> int:
    """
    Clean up old result files, keeping only the most recent.
    
    Args:
        results_dir: Directory containing results
        keep_latest: Number of latest files to keep
        pattern: Pattern for matching result files
        
    Returns:
        Number of files deleted
    """
    results_path = Path(results_dir)
    
    if not results_path.exists():
        logger.warning(f"Results directory {results_dir} does not exist")
        return 0
    
    # Find all matching files
    pattern_path = results_path / pattern
    result_files = glob.glob(str(pattern_path))
    
    if len(result_files) <= keep_latest:
        logger.info(f"Only {len(result_files)} files found, no cleanup needed")
        return 0
    
    # Sort by modification time (newest first)
    result_files.sort(key=lambda f: Path(f).stat().st_mtime, reverse=True)
    
    # Delete old files
    files_to_delete = result_files[keep_latest:]
    deleted_count = 0
    
    for filepath in files_to_delete:
        try:
            Path(filepath).unlink()
            deleted_count += 1
        except Exception as e:
            logger.warning(f"Failed to delete {filepath}: {e}")
    
    logger.info(f"Deleted {deleted_count} old result files, kept {keep_latest} latest")
    return deleted_count


def _json_serializer(obj):
    """Custom JSON serializer for non-standard types."""
    if hasattr(obj, 'tolist'):  # numpy arrays
        return obj.tolist()
    elif hasattr(obj, '__dict__'):  # objects with __dict__
        return obj.__dict__
    elif hasattr(obj, '_asdict'):  # namedtuples
        return obj._asdict()
    else:
        return str(obj)


# Export public functions
__all__ = [
    'save_experiment_result',
    'load_experiment_results',
    'load_results_by_pattern',
    'create_results_summary',
    'filter_results',
    'cleanup_old_results'
]