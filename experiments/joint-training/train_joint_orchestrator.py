#!/usr/bin/env python3
"""
Simple Joint Training Orchestrator

Alternates between surrogate and policy training phases based on time.
Manages checkpoint passing between phases.
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class JointTrainingOrchestrator:
    """Orchestrates alternating training between surrogate and policy models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize orchestrator with configuration.
        
        Args:
            config: Configuration dictionary with orchestration, surrogate, and policy configs
        """
        self.config = config
        self.orchestration = config['orchestration']
        
        # Time management
        self.total_minutes = self.orchestration['total_training_minutes']
        self.phase_minutes = self.orchestration['minutes_per_phase']
        self.start_time = time.time()
        self.phase_count = 0
        
        # Checkpoint tracking
        self.checkpoint_dir = Path(self.orchestration['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints = {
            'surrogate': None,
            'policy': None
        }
        
        # Initialize from config if provided
        if 'initial_surrogate_checkpoint' in self.orchestration:
            initial_surrogate = Path(self.orchestration['initial_surrogate_checkpoint'])
            if initial_surrogate.exists():
                self.checkpoints['surrogate'] = initial_surrogate
                logger.info(f"Initialized surrogate checkpoint from config: {initial_surrogate}")
            else:
                logger.warning(f"Initial surrogate checkpoint not found: {initial_surrogate}")
        
        if 'initial_policy_checkpoint' in self.orchestration:
            initial_policy = Path(self.orchestration['initial_policy_checkpoint'])
            if initial_policy.exists():
                self.checkpoints['policy'] = initial_policy
                logger.info(f"Initialized policy checkpoint from config: {initial_policy}")
            else:
                logger.warning(f"Initial policy checkpoint not found: {initial_policy}")
        
        # Current phase
        self.current_phase = self.orchestration.get('start_phase', 'surrogate')
        
        logger.info(f"Initialized JointTrainingOrchestrator:")
        logger.info(f"  Total training time: {self.total_minutes} minutes")
        logger.info(f"  Time per phase: {self.phase_minutes} minutes")
        logger.info(f"  Starting phase: {self.current_phase}")
        logger.info(f"  Checkpoint dir: {self.checkpoint_dir}")
    
    def get_elapsed_minutes(self) -> float:
        """Get elapsed time in minutes."""
        return (time.time() - self.start_time) / 60.0
    
    def get_remaining_minutes(self) -> float:
        """Get remaining training time in minutes."""
        return max(0, self.total_minutes - self.get_elapsed_minutes())
    
    def run_surrogate_phase(self, time_limit_minutes: int) -> Optional[str]:
        """
        Run surrogate training phase.
        
        Args:
            time_limit_minutes: Maximum time for this phase
            
        Returns:
            Path to saved checkpoint or None if failed
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"SURROGATE TRAINING PHASE {self.phase_count}")
        logger.info(f"{'='*60}")
        logger.info(f"Time limit: {time_limit_minutes} minutes")
        
        # Build command
        surrogate_script = Path(__file__).parent.parent / "surrogate-only-training/scripts/train_avici_style.py"
        checkpoint_path = self.checkpoint_dir / f"surrogate_phase_{self.phase_count}.pkl"
        
        cmd = [
            sys.executable,
            str(surrogate_script),
            "--max-time-minutes", str(time_limit_minutes),
            "--checkpoint-output", str(checkpoint_path)
        ]
        
        # Add config parameters
        surrogate_config = self.config.get('surrogate_config', {})
        if 'hidden_dim' in surrogate_config:
            cmd.extend(["--hidden-dim", str(surrogate_config['hidden_dim'])])
        if 'num_layers' in surrogate_config:
            cmd.extend(["--num-layers", str(surrogate_config['num_layers'])])
        if 'lr' in surrogate_config:
            cmd.extend(["--lr", str(surrogate_config['lr'])])
        if 'batch_size' in surrogate_config:
            cmd.extend(["--batch-size", str(surrogate_config['batch_size'])])
        if 'min_vars' in surrogate_config:
            cmd.extend(["--min-vars", str(surrogate_config['min_vars'])])
        if 'max_vars' in surrogate_config:
            cmd.extend(["--max-vars", str(surrogate_config['max_vars'])])
        if 'num_steps' in surrogate_config:
            cmd.extend(["--num-steps", str(surrogate_config['num_steps'])])
        if 'structure_types' in surrogate_config:
            cmd.extend(["--structure-types"] + surrogate_config['structure_types'])
        if 'disable_weighted_loss' in surrogate_config and surrogate_config['disable_weighted_loss']:
            cmd.extend(["--use-weighted-loss"])  # Flag disables weighting now
        
        # Resume from previous checkpoint if exists
        if self.checkpoints['surrogate']:
            cmd.extend(["--checkpoint", str(self.checkpoints['surrogate'])])
            logger.info(f"Resuming from: {self.checkpoints['surrogate']}")
        
        # Log command
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Run training (with real-time output)
        try:
            # Use Popen to stream output in real-time
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            
            # Stream output line by line
            for line in iter(process.stdout.readline, ''):
                if line:
                    print(f"[SURROGATE] {line.rstrip()}")
            
            # Wait for completion
            process.wait()
            
            if process.returncode == 0:
                logger.info("Surrogate training phase completed successfully")
            else:
                raise subprocess.CalledProcessError(process.returncode, cmd)
            
            # Check if checkpoint was created
            if checkpoint_path.exists():
                self.checkpoints['surrogate'] = checkpoint_path
                logger.info(f"Saved checkpoint: {checkpoint_path}")
                return str(checkpoint_path)
            else:
                logger.warning("Checkpoint file not found after training")
                return None
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Surrogate training failed: {e}")
            logger.error(f"Stdout: {e.stdout}")
            logger.error(f"Stderr: {e.stderr}")
            return None
    
    def run_policy_phase(self, time_limit_minutes: int) -> Optional[str]:
        """
        Run policy training phase.
        
        Args:
            time_limit_minutes: Maximum time for this phase
            
        Returns:
            Path to saved checkpoint or None if failed
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"POLICY TRAINING PHASE {self.phase_count}")
        logger.info(f"{'='*60}")
        logger.info(f"Time limit: {time_limit_minutes} minutes")
        
        # Build command - using per-batch rotation script to prevent hyperspecialization
        policy_script = Path(__file__).parent.parent / "policy-only-training/train_grpo_per_batch_rotation_simple.py"
        checkpoint_path = self.checkpoint_dir / f"policy_phase_{self.phase_count}.pkl"
        
        cmd = [
            sys.executable,
            str(policy_script),
            "--max-time-minutes", str(time_limit_minutes),
            "--checkpoint-output", str(checkpoint_path)
        ]
        
        # Add config parameters
        policy_config = self.config.get('policy_config', {})
        if 'episodes' in policy_config:
            cmd.extend(["--episodes", str(policy_config['episodes'])])
        if 'patience' in policy_config:
            cmd.extend(["--patience", str(policy_config['patience'])])
        if 'threshold' in policy_config:
            cmd.extend(["--threshold", str(policy_config['threshold'])])
        if 'learning_rate' in policy_config:
            cmd.extend(["--learning-rate", str(policy_config['learning_rate'])])
        if 'max_interventions' in policy_config:
            cmd.extend(["--max-interventions", str(policy_config['max_interventions'])])
        if 'obs_per_episode' in policy_config:
            cmd.extend(["--obs-per-episode", str(policy_config['obs_per_episode'])])
        
        # Pass variable range if specified
        if 'min_vars' in policy_config:
            cmd.extend(["--min-vars", str(policy_config['min_vars'])])
        if 'max_vars' in policy_config:
            cmd.extend(["--max-vars", str(policy_config['max_vars'])])
        
        # Pass structure types if specified
        if 'structure_types' in policy_config:
            cmd.extend(["--structure-types"] + policy_config['structure_types'])
        
        # Pass reward weights if specified
        if 'reward_weights' in policy_config:
            weights = policy_config['reward_weights']
            if 'target' in weights:
                cmd.extend(["--target-weight", str(weights['target'])])
            if 'parent' in weights:
                cmd.extend(["--parent-weight", str(weights['parent'])])
            if 'info_gain' in weights:
                cmd.extend(["--info-weight", str(weights['info_gain'])])
        
        # Pass reward type if specified
        if 'reward_type' in policy_config:
            cmd.extend(["--reward-type", str(policy_config['reward_type'])])
        
        # Pass info gain type if specified
        if 'info_gain_type' in policy_config:
            cmd.extend(["--info-gain-type", str(policy_config['info_gain_type'])])
        
        # Pass surrogate checkpoint if available
        if self.checkpoints['surrogate']:
            cmd.extend(["--surrogate-checkpoint", str(self.checkpoints['surrogate'])])
            logger.info(f"Using surrogate: {self.checkpoints['surrogate']}")
        
        # Resume from previous policy checkpoint if exists
        if self.checkpoints['policy']:
            cmd.extend(["--policy-checkpoint", str(self.checkpoints['policy'])])
            logger.info(f"Resuming from: {self.checkpoints['policy']}")
        
        # Log command
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Run training (with real-time output)
        try:
            # Use Popen to stream output in real-time
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            
            # Stream output line by line
            for line in iter(process.stdout.readline, ''):
                if line:
                    print(f"[POLICY] {line.rstrip()}")
            
            # Wait for completion
            process.wait()
            
            if process.returncode == 0:
                logger.info("Policy training phase completed successfully")
            else:
                raise subprocess.CalledProcessError(process.returncode, cmd)
            
            # Check if checkpoint was created
            if checkpoint_path.exists():
                self.checkpoints['policy'] = checkpoint_path
                logger.info(f"Saved checkpoint: {checkpoint_path}")
                return str(checkpoint_path)
            else:
                logger.warning("Checkpoint file not found after training")
                return None
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Policy training failed: {e}")
            logger.error(f"Stdout: {e.stdout}")
            logger.error(f"Stderr: {e.stderr}")
            return None
    
    def run(self):
        """Run the orchestrated training loop."""
        logger.info(f"\n{'='*70}")
        logger.info("STARTING JOINT TRAINING ORCHESTRATION")
        logger.info(f"{'='*70}\n")
        
        # Save config
        config_path = self.checkpoint_dir / "orchestrator_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        logger.info(f"Saved config to: {config_path}")
        
        # Training loop
        while self.get_remaining_minutes() > 0:
            self.phase_count += 1
            
            # Calculate phase duration
            remaining = self.get_remaining_minutes()
            phase_duration = min(self.phase_minutes, remaining)
            
            logger.info(f"\n📊 Phase {self.phase_count}")
            logger.info(f"   Type: {self.current_phase}")
            logger.info(f"   Duration: {phase_duration:.1f} minutes")
            logger.info(f"   Total elapsed: {self.get_elapsed_minutes():.1f} minutes")
            logger.info(f"   Remaining: {remaining:.1f} minutes")
            
            # Run appropriate phase
            if self.current_phase == 'surrogate':
                checkpoint = self.run_surrogate_phase(int(phase_duration))
                if checkpoint:
                    self.current_phase = 'policy'  # Switch to policy next
            else:  # policy phase
                checkpoint = self.run_policy_phase(int(phase_duration))
                if checkpoint:
                    self.current_phase = 'surrogate'  # Switch to surrogate next
            
            # Check if we should stop early
            if phase_duration < self.phase_minutes:
                logger.info(f"Stopping - insufficient time for full phase")
                break
        
        # Final summary
        logger.info(f"\n{'='*70}")
        logger.info("TRAINING COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"Total phases: {self.phase_count}")
        logger.info(f"Total time: {self.get_elapsed_minutes():.1f} minutes")
        logger.info(f"Final checkpoints:")
        logger.info(f"  Surrogate: {self.checkpoints['surrogate']}")
        logger.info(f"  Policy: {self.checkpoints['policy']}")
        logger.info(f"All checkpoints saved to: {self.checkpoint_dir}")


def create_default_config() -> Dict[str, Any]:
    """Create default configuration."""
    return {
        'orchestration': {
            'total_training_minutes': 60,
            'minutes_per_phase': 5,
            'start_phase': 'surrogate',
            'checkpoint_dir': f"experiments/joint-training/checkpoints/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        },
        'surrogate_config': {
            'hidden_dim': 128,
            'num_layers': 8,
            'lr': 0.001,
            'batch_size': 32,
            'min_vars': 3,
            'max_vars': 100,
            'structure_types': ['random', 'chain', 'fork', 'collider', 'mixed']  # Add structure types
        },
        'policy_config': {
            'episodes': 200,  # Max, but limited by time
            'patience': 3,
            'threshold': 0.9,
            'learning_rate': 5e-4
        }
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Joint Training Orchestrator")
    
    # Time configuration
    parser.add_argument('--total-minutes', type=int, default=60,
                       help='Total training time in minutes')
    parser.add_argument('--phase-minutes', type=int, default=5,
                       help='Time per phase in minutes')
    parser.add_argument('--start-phase', choices=['surrogate', 'policy'], default='surrogate',
                       help='Which model to train first')
    
    # Config file
    parser.add_argument('--config', type=str, default=None,
                       help='Path to JSON config file (overrides other args)')
    
    # Surrogate config
    parser.add_argument('--surrogate-hidden-dim', type=int, default=128,
                       help='Surrogate hidden dimension')
    parser.add_argument('--surrogate-layers', type=int, default=8,
                       help='Surrogate number of layers')
    parser.add_argument('--surrogate-lr', type=float, default=0.001,
                       help='Surrogate learning rate')
    
    # Policy config
    parser.add_argument('--policy-episodes', type=int, default=200,
                       help='Max policy episodes (limited by time)')
    parser.add_argument('--policy-lr', type=float, default=5e-4,
                       help='Policy learning rate')
    
    # Other
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                       help='Directory for checkpoints')
    
    # Initial checkpoints for resuming
    parser.add_argument('--initial-surrogate-checkpoint', type=str, default=None,
                       help='Path to initial surrogate checkpoint to resume from (must exist)')
    parser.add_argument('--initial-policy-checkpoint', type=str, default=None,
                       help='Path to initial policy checkpoint to resume from (must exist)')
    
    args = parser.parse_args()
    
    # Validate initial checkpoints if provided
    if args.initial_surrogate_checkpoint:
        checkpoint_path = Path(args.initial_surrogate_checkpoint)
        if not checkpoint_path.exists():
            logger.error(f"Initial surrogate checkpoint not found: {checkpoint_path}")
            sys.exit(1)
        logger.info(f"Will resume surrogate from: {checkpoint_path}")
    
    if args.initial_policy_checkpoint:
        checkpoint_path = Path(args.initial_policy_checkpoint)
        if not checkpoint_path.exists():
            logger.error(f"Initial policy checkpoint not found: {checkpoint_path}")
            sys.exit(1)
        logger.info(f"Will resume policy from: {checkpoint_path}")
    
    # Load or create config
    if args.config:
        logger.info(f"Loading config from: {args.config}")
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_default_config()
        
        # Override with command line args
        config['orchestration']['total_training_minutes'] = args.total_minutes
        config['orchestration']['minutes_per_phase'] = args.phase_minutes
        config['orchestration']['start_phase'] = args.start_phase
        
        if args.checkpoint_dir:
            config['orchestration']['checkpoint_dir'] = args.checkpoint_dir
        
        config['surrogate_config']['hidden_dim'] = args.surrogate_hidden_dim
        config['surrogate_config']['num_layers'] = args.surrogate_layers
        config['surrogate_config']['lr'] = args.surrogate_lr
        
        config['policy_config']['episodes'] = args.policy_episodes
        config['policy_config']['learning_rate'] = args.policy_lr
    
    # Create orchestrator
    orchestrator = JointTrainingOrchestrator(config)
    
    # Set initial checkpoints if provided
    if args.initial_surrogate_checkpoint:
        orchestrator.checkpoints['surrogate'] = Path(args.initial_surrogate_checkpoint)
        logger.info(f"Initialized surrogate checkpoint: {orchestrator.checkpoints['surrogate']}")
    
    if args.initial_policy_checkpoint:
        orchestrator.checkpoints['policy'] = Path(args.initial_policy_checkpoint)
        logger.info(f"Initialized policy checkpoint: {orchestrator.checkpoints['policy']}")
    
    # Run training
    orchestrator.run()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())