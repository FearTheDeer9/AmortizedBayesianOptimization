"""
Diagnostic utilities for tracking and analyzing catastrophic forgetting in policy training.

This module provides tools to:
1. Track quantile preferences over time
2. Analyze coefficient-quantile relationships
3. Detect forgetting patterns
4. Measure gradient conflicts
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


class QuantilePreferenceTracker:
    """
    Tracks which quantiles the policy selects for different coefficient patterns.
    """
    
    def __init__(self, window_size: int = 50):
        """
        Initialize tracker.
        
        Args:
            window_size: Size of rolling window for trend analysis
        """
        self.window_size = window_size
        
        # Track selections by coefficient sign
        self.positive_coeff_selections = deque(maxlen=window_size)
        self.negative_coeff_selections = deque(maxlen=window_size)
        
        # Track all selections with metadata
        self.all_selections = []
        
        # Track per-SCM patterns
        self.scm_patterns = {}
        
        # Track forgetting events
        self.forgetting_events = []
        
        # Performance tracking
        self.correct_choices = deque(maxlen=window_size)
        self.episode_count = 0
        
    def record_selection(
        self,
        episode: int,
        scm_id: str,
        variable: str,
        coefficient: float,
        quantile_idx: int,
        probability: float,
        target_var: str,
        is_parent: bool,
        optimization_direction: str = "MINIMIZE"
    ):
        """
        Record a quantile selection for a variable.
        
        Args:
            episode: Episode number
            scm_id: Unique SCM identifier
            variable: Variable name
            coefficient: True coefficient value (if parent)
            quantile_idx: Selected quantile (0=25%, 1=50%, 2=75%)
            probability: Selection probability
            target_var: Target variable name
            is_parent: Whether variable is true parent
            optimization_direction: MINIMIZE or MAXIMIZE
        """
        # Determine optimal quantile based on coefficient and optimization
        optimal_quantile = None
        if is_parent and coefficient != 0:
            if optimization_direction == "MINIMIZE":
                # For minimization: positive coeff → low values (25%), negative → high values (75%)
                optimal_quantile = 0 if coefficient > 0 else 2
            else:
                # For maximization: opposite
                optimal_quantile = 2 if coefficient > 0 else 0
        
        # Record selection
        selection = {
            'episode': episode,
            'scm_id': scm_id,
            'variable': variable,
            'coefficient': coefficient,
            'quantile_idx': quantile_idx,
            'quantile_name': ['25%', '50%', '75%'][quantile_idx],
            'probability': probability,
            'target_var': target_var,
            'is_parent': is_parent,
            'optimal_quantile': optimal_quantile,
            'is_correct': quantile_idx == optimal_quantile if optimal_quantile is not None else None,
            'optimization_direction': optimization_direction
        }
        
        self.all_selections.append(selection)
        
        # Track by coefficient sign (only for parents)
        if is_parent and coefficient != 0:
            if coefficient > 0:
                self.positive_coeff_selections.append(quantile_idx)
            else:
                self.negative_coeff_selections.append(quantile_idx)
            
            # Track correctness
            if selection['is_correct'] is not None:
                self.correct_choices.append(1 if selection['is_correct'] else 0)
        
        # Track SCM-specific patterns
        if scm_id not in self.scm_patterns:
            self.scm_patterns[scm_id] = {
                'selections': [],
                'coefficient_signs': {},
                'dominant_strategy': None
            }
        
        self.scm_patterns[scm_id]['selections'].append(selection)
        if is_parent:
            self.scm_patterns[scm_id]['coefficient_signs'][variable] = np.sign(coefficient)
        
        self.episode_count = max(self.episode_count, episode)
        
    def detect_forgetting(self, threshold: float = 0.7) -> List[Dict]:
        """
        Detect forgetting events where previously learned patterns are lost.
        
        Args:
            threshold: Accuracy threshold for considering a pattern "learned"
            
        Returns:
            List of forgetting events
        """
        forgetting_events = []
        
        # Analyze each SCM's learning trajectory
        for scm_id, data in self.scm_patterns.items():
            selections = data['selections']
            if len(selections) < 10:  # Need enough data
                continue
            
            # Group by episode
            episodes = defaultdict(list)
            for sel in selections:
                if sel['is_parent'] and sel['is_correct'] is not None:
                    episodes[sel['episode']].append(sel['is_correct'])
            
            # Find peaks and subsequent drops
            episode_nums = sorted(episodes.keys())
            for i in range(len(episode_nums) - 1):
                curr_acc = np.mean(episodes[episode_nums[i]])
                next_acc = np.mean(episodes[episode_nums[i + 1]])
                
                if curr_acc >= threshold and next_acc < threshold - 0.2:
                    forgetting_events.append({
                        'scm_id': scm_id,
                        'episode': episode_nums[i + 1],
                        'accuracy_drop': curr_acc - next_acc,
                        'from_accuracy': curr_acc,
                        'to_accuracy': next_acc
                    })
        
        return forgetting_events
    
    def get_quantile_preferences(self) -> Dict[str, Any]:
        """
        Get current quantile preferences for positive and negative coefficients.
        
        Returns:
            Dictionary with preference statistics
        """
        preferences = {
            'positive_coefficients': {},
            'negative_coefficients': {},
            'overall_accuracy': 0,
            'recent_accuracy': 0
        }
        
        # Analyze positive coefficient preferences
        if self.positive_coeff_selections:
            pos_counts = np.bincount(list(self.positive_coeff_selections), minlength=3)
            preferences['positive_coefficients'] = {
                '25%': int(pos_counts[0]),
                '50%': int(pos_counts[1]),
                '75%': int(pos_counts[2]),
                'dominant': ['25%', '50%', '75%'][np.argmax(pos_counts)],
                'correct_choice': '25%'  # For minimization
            }
        
        # Analyze negative coefficient preferences
        if self.negative_coeff_selections:
            neg_counts = np.bincount(list(self.negative_coeff_selections), minlength=3)
            preferences['negative_coefficients'] = {
                '25%': int(neg_counts[0]),
                '50%': int(neg_counts[1]),
                '75%': int(neg_counts[2]),
                'dominant': ['25%', '50%', '75%'][np.argmax(neg_counts)],
                'correct_choice': '75%'  # For minimization
            }
        
        # Calculate accuracies
        if self.correct_choices:
            preferences['recent_accuracy'] = np.mean(list(self.correct_choices))
            
        if self.all_selections:
            correct_selections = [s for s in self.all_selections 
                                 if s['is_correct'] is not None]
            if correct_selections:
                preferences['overall_accuracy'] = np.mean([s['is_correct'] 
                                                           for s in correct_selections])
        
        return preferences
    
    def get_learning_curve(self) -> Dict[str, List[float]]:
        """
        Get learning curves for different coefficient types.
        
        Returns:
            Dictionary with episode-wise accuracy curves
        """
        # Group selections by episode
        episodes = defaultdict(lambda: {'positive': [], 'negative': [], 'all': []})
        
        for sel in self.all_selections:
            if sel['is_parent'] and sel['is_correct'] is not None:
                ep = sel['episode']
                episodes[ep]['all'].append(sel['is_correct'])
                
                if sel['coefficient'] > 0:
                    episodes[ep]['positive'].append(sel['is_correct'])
                elif sel['coefficient'] < 0:
                    episodes[ep]['negative'].append(sel['is_correct'])
        
        # Calculate per-episode accuracies
        curve = {
            'episodes': [],
            'positive_accuracy': [],
            'negative_accuracy': [],
            'overall_accuracy': []
        }
        
        for ep in sorted(episodes.keys()):
            curve['episodes'].append(ep)
            
            # Positive coefficient accuracy
            if episodes[ep]['positive']:
                curve['positive_accuracy'].append(np.mean(episodes[ep]['positive']))
            else:
                curve['positive_accuracy'].append(None)
            
            # Negative coefficient accuracy
            if episodes[ep]['negative']:
                curve['negative_accuracy'].append(np.mean(episodes[ep]['negative']))
            else:
                curve['negative_accuracy'].append(None)
            
            # Overall accuracy
            if episodes[ep]['all']:
                curve['overall_accuracy'].append(np.mean(episodes[ep]['all']))
            else:
                curve['overall_accuracy'].append(None)
        
        return curve
    
    def save(self, filepath: Path):
        """Save tracker state to file."""
        data = {
            'all_selections': self.all_selections,
            'scm_patterns': self.scm_patterns,
            'forgetting_events': self.detect_forgetting(),
            'preferences': self.get_quantile_preferences(),
            'learning_curve': self.get_learning_curve(),
            'episode_count': self.episode_count
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    @classmethod
    def load(cls, filepath: Path) -> 'QuantilePreferenceTracker':
        """Load tracker state from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        tracker = cls()
        tracker.all_selections = data['all_selections']
        tracker.scm_patterns = data['scm_patterns']
        tracker.episode_count = data['episode_count']
        
        # Rebuild deques
        for sel in data['all_selections']:
            if sel['is_parent'] and sel['coefficient'] != 0:
                if sel['coefficient'] > 0:
                    tracker.positive_coeff_selections.append(sel['quantile_idx'])
                else:
                    tracker.negative_coeff_selections.append(sel['quantile_idx'])
                
                if sel['is_correct'] is not None:
                    tracker.correct_choices.append(1 if sel['is_correct'] else 0)
        
        return tracker


class GradientConflictAnalyzer:
    """
    Analyzes gradient conflicts between different SCM types.
    """
    
    def __init__(self):
        """Initialize analyzer."""
        self.gradient_records = []
        self.conflict_events = []
        
    def record_gradient(
        self,
        episode: int,
        scm_id: str,
        coefficient_pattern: str,  # e.g., "positive", "negative", "mixed"
        gradient_norm: float,
        loss: float
    ):
        """
        Record gradient information.
        
        Args:
            episode: Episode number
            scm_id: SCM identifier
            coefficient_pattern: Pattern of coefficients in SCM
            gradient_norm: L2 norm of gradients
            loss: Training loss
        """
        self.gradient_records.append({
            'episode': episode,
            'scm_id': scm_id,
            'coefficient_pattern': coefficient_pattern,
            'gradient_norm': gradient_norm,
            'loss': loss
        })
    
    def detect_conflicts(self, window: int = 10) -> List[Dict]:
        """
        Detect gradient conflicts by analyzing direction changes.
        
        Args:
            window: Window size for conflict detection
            
        Returns:
            List of detected conflicts
        """
        if len(self.gradient_records) < window:
            return []
        
        conflicts = []
        
        # Group by coefficient pattern
        patterns = defaultdict(list)
        for record in self.gradient_records:
            patterns[record['coefficient_pattern']].append(record)
        
        # Look for rapid gradient norm changes between patterns
        for i in range(len(self.gradient_records) - 1):
            curr = self.gradient_records[i]
            next = self.gradient_records[i + 1]
            
            if curr['coefficient_pattern'] != next['coefficient_pattern']:
                # Pattern switch - check for conflict
                norm_ratio = next['gradient_norm'] / (curr['gradient_norm'] + 1e-8)
                
                if norm_ratio > 2.0 or norm_ratio < 0.5:
                    conflicts.append({
                        'episode': next['episode'],
                        'from_pattern': curr['coefficient_pattern'],
                        'to_pattern': next['coefficient_pattern'],
                        'norm_ratio': norm_ratio,
                        'severity': 'high' if norm_ratio > 3.0 or norm_ratio < 0.33 else 'medium'
                    })
        
        return conflicts


class DiagnosticLogger:
    """
    Centralized diagnostic logging for training analysis.
    """
    
    def __init__(self, log_dir: Path, experiment_name: str):
        """
        Initialize diagnostic logger.
        
        Args:
            log_dir: Directory for log files
            experiment_name: Name of experiment
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        self.quantile_tracker = QuantilePreferenceTracker()
        self.gradient_analyzer = GradientConflictAnalyzer()
        
        # Episode-level logs
        self.episode_logs = []
        
        # SCM transition logs
        self.scm_transitions = []
        
    def log_episode_start(self, episode: int, scm_info: Dict):
        """Log start of episode with SCM information."""
        self.episode_logs.append({
            'episode': episode,
            'event': 'start',
            'scm_info': scm_info,
            'timestamp': str(np.datetime64('now'))
        })
    
    def log_intervention(
        self,
        episode: int,
        intervention_idx: int,
        variable: str,
        quantile_idx: int,
        value: float,
        probability: float,
        coefficient: Optional[float] = None,
        is_parent: bool = False,
        target_var: Optional[str] = None
    ):
        """Log an intervention selection."""
        log_entry = {
            'episode': episode,
            'intervention_idx': intervention_idx,
            'variable': variable,
            'quantile_idx': quantile_idx,
            'quantile_name': ['25%', '50%', '75%'][quantile_idx],
            'value': value,
            'probability': probability,
            'coefficient': coefficient,
            'is_parent': is_parent,
            'target_var': target_var,
            'timestamp': str(np.datetime64('now'))
        }
        
        self.episode_logs.append({
            'episode': episode,
            'event': 'intervention',
            'data': log_entry
        })
        
        # Also track in quantile tracker if we have coefficient info
        if coefficient is not None and target_var:
            self.quantile_tracker.record_selection(
                episode=episode,
                scm_id=f"scm_{episode}",  # Simple ID for now
                variable=variable,
                coefficient=coefficient,
                quantile_idx=quantile_idx,
                probability=probability,
                target_var=target_var,
                is_parent=is_parent
            )
    
    def log_scm_transition(self, episode: int, from_scm: Dict, to_scm: Dict, reason: str):
        """Log SCM transition."""
        self.scm_transitions.append({
            'episode': episode,
            'from_scm': from_scm,
            'to_scm': to_scm,
            'reason': reason,
            'timestamp': str(np.datetime64('now'))
        })
    
    def log_gradient_info(
        self,
        episode: int,
        gradient_norm: float,
        loss: float,
        coefficient_pattern: str
    ):
        """Log gradient information."""
        self.gradient_analyzer.record_gradient(
            episode=episode,
            scm_id=f"scm_{episode}",
            coefficient_pattern=coefficient_pattern,
            gradient_norm=gradient_norm,
            loss=loss
        )
    
    def save_checkpoint(self, episode: int):
        """Save diagnostic checkpoint."""
        checkpoint_dir = self.log_dir / f"checkpoint_ep{episode}"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save quantile tracker
        self.quantile_tracker.save(checkpoint_dir / "quantile_tracker.json")
        
        # Save episode logs
        with open(checkpoint_dir / "episode_logs.json", 'w') as f:
            json.dump(self.episode_logs[-1000:], f, indent=2, default=str)  # Last 1000 entries
        
        # Save SCM transitions
        with open(checkpoint_dir / "scm_transitions.json", 'w') as f:
            json.dump(self.scm_transitions, f, indent=2, default=str)
        
        # Save gradient analysis
        conflicts = self.gradient_analyzer.detect_conflicts()
        with open(checkpoint_dir / "gradient_conflicts.json", 'w') as f:
            json.dump(conflicts, f, indent=2, default=str)
        
        logger.info(f"Saved diagnostic checkpoint at episode {episode}")
    
    def get_summary(self) -> Dict:
        """Get summary of diagnostics."""
        preferences = self.quantile_tracker.get_quantile_preferences()
        forgetting_events = self.quantile_tracker.detect_forgetting()
        gradient_conflicts = self.gradient_analyzer.detect_conflicts()
        
        return {
            'experiment_name': self.experiment_name,
            'total_episodes': self.quantile_tracker.episode_count,
            'quantile_preferences': preferences,
            'forgetting_events': len(forgetting_events),
            'gradient_conflicts': len(gradient_conflicts),
            'scm_transitions': len(self.scm_transitions),
            'recent_accuracy': preferences.get('recent_accuracy', 0),
            'overall_accuracy': preferences.get('overall_accuracy', 0)
        }
    
    def print_summary(self):
        """Print diagnostic summary to console."""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print(f"DIAGNOSTIC SUMMARY: {summary['experiment_name']}")
        print("="*60)
        print(f"Total episodes: {summary['total_episodes']}")
        print(f"SCM transitions: {summary['scm_transitions']}")
        print(f"Forgetting events: {summary['forgetting_events']}")
        print(f"Gradient conflicts: {summary['gradient_conflicts']}")
        print(f"Recent accuracy: {summary['recent_accuracy']:.2%}")
        print(f"Overall accuracy: {summary['overall_accuracy']:.2%}")
        
        prefs = summary['quantile_preferences']
        if prefs.get('positive_coefficients'):
            print(f"\nPositive coefficients prefer: {prefs['positive_coefficients']['dominant']}")
            print(f"  (Should prefer: {prefs['positive_coefficients']['correct_choice']})")
        
        if prefs.get('negative_coefficients'):
            print(f"Negative coefficients prefer: {prefs['negative_coefficients']['dominant']}")
            print(f"  (Should prefer: {prefs['negative_coefficients']['correct_choice']})")
        
        print("="*60 + "\n")