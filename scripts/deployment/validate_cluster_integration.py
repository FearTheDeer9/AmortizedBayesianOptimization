#!/usr/bin/env python3
"""
Cluster Integration Validation Script

Step-by-step validation of GPU cluster integration for ACBO SFT data collection.
This script guides you through testing each component before running large-scale 
data collection.

Usage:
    python scripts/validate_cluster_integration.py --user your_username --step all
    python scripts/validate_cluster_integration.py --user your_username --step sync
    python scripts/validate_cluster_integration.py --user your_username --step environment  
    python scripts/validate_cluster_integration.py --user your_username --step mini-collection
    python scripts/validate_cluster_integration.py --user your_username --step full-pipeline
"""

import argparse
import subprocess
import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ValidationStep:
    """Validation step configuration."""
    name: str
    description: str
    commands: List[str]
    success_criteria: List[str]
    estimated_time: str


class ClusterValidator:
    """Validates Imperial GPU cluster integration for ACBO."""
    
    def __init__(self, username: str, cluster_host: str = "login.hpc.ic.ac.uk"):
        self.username = username
        self.cluster_host = cluster_host
        self.project_dir = f"/vol/bitbucket/{username}/causal_bayes_opt"
        self.local_project_dir = Path(__file__).parent.parent
        
        # Define validation steps
        self.steps = {
            "sync": ValidationStep(
                name="Project Synchronization",
                description="Sync project code to cluster and verify structure",
                commands=[
                    f"python scripts/sync_to_cluster.py --user {username} --sync-code",
                    f"ssh {username}@{cluster_host} 'ls -la {self.project_dir}/scripts/'"
                ],
                success_criteria=[
                    "Project files synced successfully",
                    "Scripts directory contains collect_sft_dataset.py",
                    "Environment setup script present"
                ],
                estimated_time="2-3 minutes"
            ),
            
            "environment": ValidationStep(
                name="Environment Setup",
                description="Set up and validate Python environment with JAX GPU support",
                commands=[
                    f"ssh {username}@{cluster_host} 'cd {self.project_dir} && source cluster/scripts/setup_env.sh'",
                    f"ssh {username}@{cluster_host} 'cd {self.project_dir} && python -c \"import jax; print(f\\\"JAX devices: {{jax.devices()}}\\\")\"'"
                ],
                success_criteria=[
                    "Environment activated successfully",
                    "JAX imported without errors",
                    "GPU devices detected (if available)"
                ],
                estimated_time="3-5 minutes"
            ),
            
            "mini-collection": ValidationStep(
                name="Mini Data Collection Test",
                description="Run small data collection job to validate PARENT_SCALE integration",
                commands=[
                    f"ssh {username}@{cluster_host} 'cd {self.project_dir} && python scripts/dev_workflow.py --quick-test'"
                ],
                success_criteria=[
                    "Environment check passes",
                    "Import validation succeeds", 
                    "PARENT_SCALE integration works",
                    "Expert collection completes"
                ],
                estimated_time="5-10 minutes"
            ),
            
            "slurm-test": ValidationStep(
                name="Slurm Job Submission Test",
                description="Submit and monitor a small Slurm job",
                commands=[
                    f"scp cluster/jobs/collect_data.sbatch {username}@{cluster_host}:{self.project_dir}/",
                    f"ssh {username}@{cluster_host} 'cd {self.project_dir} && sbatch --export=DATASET_SIZE=small,BATCH_SIZE=5 collect_data.sbatch'"
                ],
                success_criteria=[
                    "Job submitted successfully",
                    "Job appears in squeue",
                    "Output files generated"
                ],
                estimated_time="10-15 minutes"
            ),
            
            "full-pipeline": ValidationStep(
                name="Full Pipeline Validation",
                description="Run complete medium-scale collection with validation",
                commands=[
                    f"python scripts/deploy_to_cluster.py --user {username} --run-pipeline --size medium"
                ],
                success_criteria=[
                    "Data collection job completes",
                    "Validation passes",
                    "Dataset ready for training"
                ],
                estimated_time="2-4 hours"
            )
        }
    
    def run_command(self, command: str, timeout: int = 300) -> tuple[bool, str]:
        """Run a shell command and return success status and output."""
        try:
            logger.info(f"Executing: {command}")
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True, 
                timeout=timeout
            )
            
            if result.returncode == 0:
                logger.info(f"✅ Command succeeded")
                return True, result.stdout
            else:
                logger.error(f"❌ Command failed (exit code {result.returncode})")
                logger.error(f"Error output: {result.stderr}")
                return False, result.stderr
                
        except subprocess.TimeoutExpired:
            logger.error(f"❌ Command timed out after {timeout} seconds")
            return False, "Command timed out"
        except Exception as e:
            logger.error(f"❌ Command failed with exception: {e}")
            return False, str(e)
    
    def validate_step(self, step_name: str) -> bool:
        """Validate a specific step."""
        if step_name not in self.steps:
            logger.error(f"Unknown validation step: {step_name}")
            return False
        
        step = self.steps[step_name]
        logger.info(f"\n{'='*60}")
        logger.info(f"🔍 VALIDATING: {step.name}")
        logger.info(f"📝 Description: {step.description}")
        logger.info(f"⏱️  Estimated time: {step.estimated_time}")
        logger.info(f"{'='*60}")
        
        all_success = True
        
        for i, command in enumerate(step.commands, 1):
            logger.info(f"\n📋 Step {i}/{len(step.commands)}")
            success, output = self.run_command(command)
            
            if success:
                logger.info(f"✅ Step {i} completed successfully")
                if output.strip():
                    logger.info(f"Output: {output.strip()[:200]}...")
            else:
                logger.error(f"❌ Step {i} failed")
                all_success = False
                break
        
        if all_success:
            logger.info(f"\n🎉 {step.name} - ALL CHECKS PASSED")
            logger.info("Success criteria met:")
            for criterion in step.success_criteria:
                logger.info(f"  ✅ {criterion}")
        else:
            logger.error(f"\n💥 {step.name} - VALIDATION FAILED")
            logger.error("Please check the errors above and retry")
        
        return all_success
    
    def validate_all(self) -> bool:
        """Run all validation steps in sequence."""
        logger.info("🚀 Starting comprehensive cluster validation...")
        
        step_order = ["sync", "environment", "mini-collection", "slurm-test", "full-pipeline"]
        results = {}
        
        for step_name in step_order:
            success = self.validate_step(step_name)
            results[step_name] = success
            
            if not success:
                logger.error(f"\n🛑 Validation failed at step: {step_name}")
                logger.error("Please fix the issues and restart from this step")
                break
            
            # Brief pause between steps
            if step_name != step_order[-1]:
                logger.info("\n⏸️  Pausing 5 seconds before next step...")
                time.sleep(5)
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("📊 VALIDATION SUMMARY")
        logger.info(f"{'='*60}")
        
        for step_name, success in results.items():
            status = "✅ PASSED" if success else "❌ FAILED"
            logger.info(f"{self.steps[step_name].name}: {status}")
        
        overall_success = all(results.values())
        if overall_success:
            logger.info(f"\n🎉 ALL VALIDATIONS PASSED!")
            logger.info("✅ Cluster is ready for production data collection")
            logger.info("\nNext steps:")
            logger.info("1. Run large-scale collection: sbatch --export=DATASET_SIZE=large cluster/jobs/collect_data.sbatch")
            logger.info("2. Monitor with: squeue -u your_username")
            logger.info("3. Check logs: tail -f /vol/bitbucket/your_username/causal_bayes_opt/logs/collection/*.out")
        else:
            logger.error(f"\n💥 VALIDATION INCOMPLETE")
            logger.error("Please fix the failed steps before proceeding to production")
        
        return overall_success
    
    def show_next_steps(self):
        """Show recommended next steps after validation."""
        logger.info(f"\n📋 RECOMMENDED NEXT STEPS:")
        logger.info(f"")
        logger.info(f"1. 🔄 Sync project to cluster:")
        logger.info(f"   python scripts/sync_to_cluster.py --user {self.username} --sync-code")
        logger.info(f"")
        logger.info(f"2. 🧪 Run environment test:")
        logger.info(f"   ssh {self.username}@{self.cluster_host} 'cd {self.project_dir} && python scripts/dev_workflow.py --quick-test'")
        logger.info(f"")
        logger.info(f"3. 📊 Submit small test job:")
        logger.info(f"   ssh {self.username}@{self.cluster_host} 'cd {self.project_dir} && sbatch --export=DATASET_SIZE=small cluster/jobs/collect_data.sbatch'")
        logger.info(f"")
        logger.info(f"4. 📈 Monitor job progress:")
        logger.info(f"   ssh {self.username}@{self.cluster_host} 'squeue -u {self.username}'")
        logger.info(f"")
        logger.info(f"5. 🎯 Check results:")
        logger.info(f"   ssh {self.username}@{self.cluster_host} 'ls -la {self.project_dir}/data/raw'")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Validate Imperial GPU cluster integration for ACBO"
    )
    parser.add_argument(
        "--user", 
        required=True,
        help="Imperial College username"
    )
    parser.add_argument(
        "--step",
        choices=["sync", "environment", "mini-collection", "slurm-test", "full-pipeline", "all"],
        default="all",
        help="Validation step to run"
    )
    parser.add_argument(
        "--cluster-host",
        default="login.hpc.ic.ac.uk",
        help="Cluster login host"
    )
    parser.add_argument(
        "--show-steps",
        action="store_true",
        help="Show next steps without running validation"
    )
    
    args = parser.parse_args()
    
    validator = ClusterValidator(args.user, args.cluster_host)
    
    if args.show_steps:
        validator.show_next_steps()
        return
    
    if args.step == "all":
        success = validator.validate_all()
        exit(0 if success else 1)
    else:
        success = validator.validate_step(args.step)
        exit(0 if success else 1)


if __name__ == "__main__":
    main()