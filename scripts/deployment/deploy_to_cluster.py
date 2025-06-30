#!/usr/bin/env python3
"""
Cluster Deployment Orchestrator

Master script for deploying and managing ACBO training on Imperial's GPU cluster.
Automates the complete workflow from data collection to model training.

Features:
- One-command deployment setup
- Job submission and monitoring
- Dependency chain management
- Resource optimization
- Error recovery and retry logic

Usage:
    python scripts/deploy_to_cluster.py --user your_username --setup
    python scripts/deploy_to_cluster.py --user your_username --run-pipeline
    python scripts/deploy_to_cluster.py --user your_username --monitor
"""

import argparse
import subprocess
import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Slurm job status enumeration."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"
    UNKNOWN = "UNKNOWN"


@dataclass
class JobInfo:
    """Information about a Slurm job."""
    job_id: str
    name: str
    status: JobStatus
    runtime: str
    nodes: str
    exit_code: Optional[str] = None


@dataclass
class PipelineConfig:
    """Configuration for the training pipeline."""
    # Dataset configuration
    dataset_size: str = "medium"  # small/medium/large/xlarge
    difficulty_levels: List[str] = None
    
    # Training configuration
    surrogate_epochs: int = 100
    acquisition_bc_epochs: int = 50
    acquisition_grpo_episodes: int = 1000
    
    # Resource configuration
    data_collection_time: str = "24:00:00"
    surrogate_training_time: str = "48:00:00"
    acquisition_training_time: str = "72:00:00"
    
    # Retry configuration
    max_retries: int = 2
    retry_delay: int = 300  # seconds
    
    def __post_init__(self):
        if self.difficulty_levels is None:
            self.difficulty_levels = ["all"]


class ClusterDeployment:
    """Manages ACBO deployment on Imperial's GPU cluster."""
    
    def __init__(self, username: str, cluster_host: str = "gpucluster2.doc.ic.ac.uk"):
        self.username = username
        self.cluster_host = cluster_host
        self.remote_project_dir = f"/vol/bitbucket/{username}/causal_bayes_opt"
        
        self.job_history: Dict[str, JobInfo] = {}
        self.pipeline_state_file = f"/vol/bitbucket/{username}/causal_bayes_opt/pipeline_state.json"
        
        logger.info(f"Cluster deployment initialized for {username}@{cluster_host}")
    
    def setup_deployment(self) -> bool:
        """Setup complete deployment environment."""
        logger.info("Setting up deployment environment...")
        
        steps = [
            ("Testing connection", self._test_connection),
            ("Syncing code", self._sync_code),
            ("Setting up environment", self._setup_environment),
            ("Validating setup", self._validate_setup)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"Step: {step_name}")
            if not step_func():
                logger.error(f"❌ Setup failed at: {step_name}")
                return False
            logger.info(f"✅ {step_name} completed")
        
        logger.info("🚀 Deployment setup completed successfully!")
        return True
    
    def run_pipeline(self, config: PipelineConfig) -> bool:
        """Run the complete training pipeline."""
        logger.info("Starting ACBO training pipeline...")
        
        # Save pipeline configuration
        self._save_pipeline_state({"config": config.__dict__, "status": "starting"})
        
        pipeline_steps = [
            ("Data Collection", self._submit_data_collection, config),
            ("Data Preprocessing", self._submit_preprocessing, config),
            ("Surrogate Training", self._submit_surrogate_training, config),
            ("Acquisition Training", self._submit_acquisition_training, config)
        ]
        
        job_dependencies = {}
        
        for step_name, step_func, step_config in pipeline_steps:
            logger.info(f"Pipeline step: {step_name}")
            
            # Submit job with dependencies
            prev_job_id = list(job_dependencies.values())[-1] if job_dependencies else None
            job_id = step_func(step_config, dependency=prev_job_id)
            
            if job_id:
                job_dependencies[step_name] = job_id
                logger.info(f"✅ {step_name} submitted: {job_id}")
            else:
                logger.error(f"❌ Failed to submit {step_name}")
                return False
        
        # Update pipeline state
        self._save_pipeline_state({
            "config": config.__dict__,
            "status": "running",
            "jobs": job_dependencies,
            "start_time": time.time()
        })
        
        logger.info("🚀 Pipeline submitted successfully!")
        logger.info("Job dependencies:")
        for step, job_id in job_dependencies.items():
            logger.info(f"  {step}: {job_id}")
        
        return True
    
    def monitor_pipeline(self, follow: bool = False) -> bool:
        """Monitor pipeline progress."""
        logger.info("Monitoring pipeline progress...")
        
        # Load pipeline state
        pipeline_state = self._load_pipeline_state()
        if not pipeline_state:
            logger.error("No active pipeline found")
            return False
        
        jobs = pipeline_state.get("jobs", {})
        if not jobs:
            logger.error("No jobs found in pipeline state")
            return False
        
        while True:
            # Check status of all jobs
            all_completed = True
            any_failed = False
            
            print(f"\n{'='*60}")
            print(f"Pipeline Status - {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*60}")
            
            for step_name, job_id in jobs.items():
                job_info = self._get_job_status(job_id)
                status_icon = self._get_status_icon(job_info.status)
                
                print(f"{status_icon} {step_name:<20} {job_id:<12} {job_info.status.value:<12} {job_info.runtime}")
                
                if job_info.status in [JobStatus.PENDING, JobStatus.RUNNING]:
                    all_completed = False
                elif job_info.status in [JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.TIMEOUT]:
                    any_failed = True
            
            # Check overall status
            if any_failed:
                print(f"\n❌ Pipeline failed - some jobs failed")
                self._save_pipeline_state({**pipeline_state, "status": "failed"})
                break
            elif all_completed:
                print(f"\n✅ Pipeline completed successfully!")
                self._save_pipeline_state({**pipeline_state, "status": "completed"})
                break
            
            if not follow:
                break
            
            # Wait before next check
            time.sleep(30)
        
        return not any_failed
    
    def cancel_pipeline(self) -> bool:
        """Cancel active pipeline jobs."""
        logger.info("Cancelling pipeline...")
        
        pipeline_state = self._load_pipeline_state()
        if not pipeline_state:
            logger.error("No active pipeline found")
            return False
        
        jobs = pipeline_state.get("jobs", {})
        cancelled_jobs = []
        
        for step_name, job_id in jobs.items():
            try:
                result = subprocess.run([
                    "ssh", f"{self.username}@{self.cluster_host}",
                    f"scancel {job_id}"
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    cancelled_jobs.append(job_id)
                    logger.info(f"✅ Cancelled {step_name}: {job_id}")
                else:
                    logger.warning(f"⚠️ Could not cancel {step_name}: {job_id}")
                    
            except Exception as e:
                logger.error(f"❌ Error cancelling {step_name}: {e}")
        
        # Update pipeline state
        if cancelled_jobs:
            self._save_pipeline_state({**pipeline_state, "status": "cancelled"})
            logger.info(f"Pipeline cancellation completed. Cancelled {len(cancelled_jobs)} jobs.")
        
        return len(cancelled_jobs) > 0
    
    def _test_connection(self) -> bool:
        """Test SSH connection to cluster."""
        try:
            result = subprocess.run([
                "ssh", f"{self.username}@{self.cluster_host}",
                "echo 'Connection test successful'"
            ], capture_output=True, text=True, timeout=30)
            return result.returncode == 0
        except:
            return False
    
    def _sync_code(self) -> bool:
        """Sync code to cluster."""
        try:
            # Use the sync script
            result = subprocess.run([
                "python", "scripts/sync_to_cluster.py",
                "--user", self.username,
                "--cluster-host", self.cluster_host,
                "--sync-code"
            ], timeout=600)
            return result.returncode == 0
        except:
            return False
    
    def _setup_environment(self) -> bool:
        """Setup environment on cluster."""
        try:
            result = subprocess.run([
                "ssh", f"{self.username}@{self.cluster_host}",
                f"cd {self.remote_project_dir} && bash cluster/scripts/setup_env.sh"
            ], timeout=1800)
            return result.returncode == 0
        except:
            return False
    
    def _validate_setup(self) -> bool:
        """Validate deployment setup."""
        try:
            # Test environment activation and basic imports
            result = subprocess.run([
                "ssh", f"{self.username}@{self.cluster_host}",
                f"cd {self.remote_project_dir} && source activate_env.sh && python -c 'import jax; print(jax.devices())'"
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0 and "gpu" in result.stdout.lower():
                return True
            else:
                logger.warning(f"Environment validation output: {result.stdout}")
                return False
        except:
            return False
    
    def _submit_data_collection(self, config: PipelineConfig, dependency: Optional[str] = None) -> Optional[str]:
        """Submit data collection job."""
        # Prepare environment variables
        env_vars = {
            "DATASET_SIZE": config.dataset_size,
            "DIFFICULTY_LEVELS": " ".join(config.difficulty_levels),
            "WORKERS": "8",
            "BATCH_SIZE": "100"
        }
        
        return self._submit_job(
            "cluster/jobs/collect_data.sbatch",
            env_vars=env_vars,
            dependency=dependency,
            time_limit=config.data_collection_time
        )
    
    def _submit_preprocessing(self, config: PipelineConfig, dependency: Optional[str] = None) -> Optional[str]:
        """Submit data preprocessing job."""
        # Create a simple preprocessing job
        preprocessing_script = f"""#!/bin/bash
#SBATCH --job-name=acbo_preprocess
#SBATCH --partition=training
#SBATCH --qos=training
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/vol/bitbucket/{self.username}/causal_bayes_opt/logs/preprocessing_%j.out
#SBATCH --error=/vol/bitbucket/{self.username}/causal_bayes_opt/logs/preprocessing_%j.err

cd /vol/bitbucket/{self.username}/causal_bayes_opt
source activate_env.sh

# Process collected data
python scripts/prepare_sft_data.py data/raw --output data/processed --curriculum

# Split dataset
python scripts/split_dataset.py data/processed --curriculum --stratify difficulty graph_type

echo "Preprocessing completed successfully"
"""
        
        # Write preprocessing script
        script_path = f"/tmp/preprocess_{self.username}_{int(time.time())}.sbatch"
        with open(script_path, 'w') as f:
            f.write(preprocessing_script)
        
        # Copy script to cluster and submit
        try:
            # Copy script
            subprocess.run([
                "scp", script_path, 
                f"{self.username}@{self.cluster_host}:/tmp/"
            ], check=True)
            
            # Submit job
            cmd = ["ssh", f"{self.username}@{self.cluster_host}", f"sbatch"]
            if dependency:
                cmd.append(f"--dependency=afterok:{dependency}")
            cmd.append(f"/tmp/{Path(script_path).name}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                job_id = result.stdout.strip().split()[-1]
                return job_id
            
        except Exception as e:
            logger.error(f"Failed to submit preprocessing job: {e}")
        
        return None
    
    def _submit_surrogate_training(self, config: PipelineConfig, dependency: Optional[str] = None) -> Optional[str]:
        """Submit surrogate training job."""
        env_vars = {
            "EPOCHS": str(config.surrogate_epochs),
            "BATCH_SIZE": "32",
            "LEARNING_RATE": "1e-3"
        }
        
        return self._submit_job(
            "cluster/jobs/train_surrogate.sbatch",
            env_vars=env_vars,
            dependency=dependency,
            time_limit=config.surrogate_training_time
        )
    
    def _submit_acquisition_training(self, config: PipelineConfig, dependency: Optional[str] = None) -> Optional[str]:
        """Submit acquisition training job."""
        env_vars = {
            "BC_EPOCHS": str(config.acquisition_bc_epochs),
            "GRPO_EPISODES": str(config.acquisition_grpo_episodes),
            "BATCH_SIZE": "64"
        }
        
        return self._submit_job(
            "cluster/jobs/train_acquisition.sbatch",
            env_vars=env_vars,
            dependency=dependency,
            time_limit=config.acquisition_training_time
        )
    
    def _submit_job(self, script_path: str, env_vars: Optional[Dict[str, str]] = None,
                   dependency: Optional[str] = None, time_limit: Optional[str] = None) -> Optional[str]:
        """Submit a Slurm job."""
        try:
            cmd = ["ssh", f"{self.username}@{self.cluster_host}"]
            
            # Build sbatch command
            sbatch_cmd = "sbatch"
            
            if dependency:
                sbatch_cmd += f" --dependency=afterok:{dependency}"
            
            if time_limit:
                sbatch_cmd += f" --time={time_limit}"
            
            # Add environment variables
            if env_vars:
                env_str = " ".join([f"{k}={v}" for k, v in env_vars.items()])
                sbatch_cmd = f"{env_str} {sbatch_cmd}"
            
            sbatch_cmd += f" {self.remote_project_dir}/{script_path}"
            
            cmd.append(sbatch_cmd)
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Extract job ID from output
                job_id = result.stdout.strip().split()[-1]
                return job_id
            else:
                logger.error(f"Job submission failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Error submitting job: {e}")
            return None
    
    def _get_job_status(self, job_id: str) -> JobInfo:
        """Get status of a Slurm job."""
        try:
            result = subprocess.run([
                "ssh", f"{self.username}@{self.cluster_host}",
                f"squeue -j {job_id} --format='%i|%j|%T|%M|%N' --noheader"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and result.stdout.strip():
                # Parse squeue output
                fields = result.stdout.strip().split('|')
                if len(fields) >= 5:
                    return JobInfo(
                        job_id=fields[0],
                        name=fields[1],
                        status=JobStatus(fields[2]),
                        runtime=fields[3],
                        nodes=fields[4]
                    )
            
            # Job not in queue, check if it completed
            result = subprocess.run([
                "ssh", f"{self.username}@{self.cluster_host}",
                f"sacct -j {job_id} --format='JobID,JobName,State,ExitCode' --noheader"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if job_id in line and not '.batch' in line:
                        fields = line.split()
                        if len(fields) >= 3:
                            state = fields[2]
                            exit_code = fields[3] if len(fields) > 3 else None
                            
                            # Map sacct states to our enum
                            status_map = {
                                'COMPLETED': JobStatus.COMPLETED,
                                'FAILED': JobStatus.FAILED,
                                'CANCELLED': JobStatus.CANCELLED,
                                'TIMEOUT': JobStatus.TIMEOUT
                            }
                            
                            return JobInfo(
                                job_id=job_id,
                                name=fields[1] if len(fields) > 1 else "unknown",
                                status=status_map.get(state, JobStatus.UNKNOWN),
                                runtime="finished",
                                nodes="none",
                                exit_code=exit_code
                            )
            
        except Exception as e:
            logger.warning(f"Error getting job status for {job_id}: {e}")
        
        return JobInfo(job_id=job_id, name="unknown", status=JobStatus.UNKNOWN, runtime="unknown", nodes="unknown")
    
    def _get_status_icon(self, status: JobStatus) -> str:
        """Get icon for job status."""
        icons = {
            JobStatus.PENDING: "⏳",
            JobStatus.RUNNING: "🔄",
            JobStatus.COMPLETED: "✅",
            JobStatus.FAILED: "❌",
            JobStatus.CANCELLED: "🛑",
            JobStatus.TIMEOUT: "⏰",
            JobStatus.UNKNOWN: "❓"
        }
        return icons.get(status, "❓")
    
    def _save_pipeline_state(self, state: Dict[str, Any]):
        """Save pipeline state to cluster."""
        try:
            state_json = json.dumps(state, indent=2, default=str)
            
            # Write to temporary file and copy to cluster
            temp_file = f"/tmp/pipeline_state_{self.username}.json"
            with open(temp_file, 'w') as f:
                f.write(state_json)
            
            subprocess.run([
                "scp", temp_file, 
                f"{self.username}@{self.cluster_host}:{self.pipeline_state_file}"
            ], check=True)
            
        except Exception as e:
            logger.warning(f"Could not save pipeline state: {e}")
    
    def _load_pipeline_state(self) -> Optional[Dict[str, Any]]:
        """Load pipeline state from cluster."""
        try:
            temp_file = f"/tmp/pipeline_state_{self.username}_load.json"
            
            result = subprocess.run([
                "scp", 
                f"{self.username}@{self.cluster_host}:{self.pipeline_state_file}",
                temp_file
            ], capture_output=True)
            
            if result.returncode == 0:
                with open(temp_file, 'r') as f:
                    return json.load(f)
            
        except Exception as e:
            logger.warning(f"Could not load pipeline state: {e}")
        
        return None


def create_pipeline_config(args: argparse.Namespace) -> PipelineConfig:
    """Create pipeline configuration from arguments."""
    return PipelineConfig(
        dataset_size=args.dataset_size,
        difficulty_levels=args.difficulty if args.difficulty else ["all"],
        surrogate_epochs=args.surrogate_epochs,
        acquisition_bc_epochs=args.acquisition_bc_epochs,
        acquisition_grpo_episodes=args.acquisition_grpo_episodes
    )


def main():
    """CLI entry point for cluster deployment."""
    parser = argparse.ArgumentParser(
        description="Deploy and manage ACBO training on Imperial's GPU cluster",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Setup deployment environment
  python scripts/deploy_to_cluster.py --user your_username --setup
  
  # Run complete pipeline
  python scripts/deploy_to_cluster.py --user your_username --run-pipeline
  
  # Monitor pipeline progress
  python scripts/deploy_to_cluster.py --user your_username --monitor --follow
  
  # Cancel active pipeline
  python scripts/deploy_to_cluster.py --user your_username --cancel
        """
    )
    
    parser.add_argument("--user", required=True,
                       help="Imperial username for cluster access")
    parser.add_argument("--cluster-host", default="gpucluster2.doc.ic.ac.uk",
                       help="Cluster hostname")
    
    # Actions
    parser.add_argument("--setup", action="store_true",
                       help="Setup deployment environment")
    parser.add_argument("--run-pipeline", action="store_true",
                       help="Run complete training pipeline")
    parser.add_argument("--monitor", action="store_true",
                       help="Monitor pipeline progress")
    parser.add_argument("--cancel", action="store_true",
                       help="Cancel active pipeline")
    parser.add_argument("--follow", action="store_true",
                       help="Follow progress in real-time (with --monitor)")
    
    # Pipeline configuration
    parser.add_argument("--dataset-size", choices=["small", "medium", "large", "xlarge"],
                       default="medium", help="Dataset size for training")
    parser.add_argument("--difficulty", nargs="+",
                       choices=["difficulty_1", "difficulty_2", "difficulty_3", "difficulty_4", "difficulty_5"],
                       help="Difficulty levels for curriculum learning")
    parser.add_argument("--surrogate-epochs", type=int, default=100,
                       help="Epochs for surrogate model training")
    parser.add_argument("--acquisition-bc-epochs", type=int, default=50,
                       help="Epochs for behavioral cloning phase")
    parser.add_argument("--acquisition-grpo-episodes", type=int, default=1000,
                       help="Episodes for GRPO training phase")
    
    args = parser.parse_args()
    
    try:
        # Create deployment manager
        deployment = ClusterDeployment(args.user, args.cluster_host)
        
        success = True
        
        # Execute requested actions
        if args.setup:
            success = deployment.setup_deployment()
        
        elif args.run_pipeline:
            config = create_pipeline_config(args)
            success = deployment.run_pipeline(config)
        
        elif args.monitor:
            success = deployment.monitor_pipeline(follow=args.follow)
        
        elif args.cancel:
            success = deployment.cancel_pipeline()
        
        else:
            print("No action specified. Use --setup, --run-pipeline, --monitor, or --cancel")
            return 1
        
        if success:
            print(f"\n✅ Operation completed successfully!")
        else:
            print(f"\n❌ Operation failed. Check logs for details.")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️ Operation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Operation failed: {e}")
        logger.exception("Operation failed")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())