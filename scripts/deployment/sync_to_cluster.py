#!/usr/bin/env python3
"""
Cluster Data Synchronization Script

Efficiently syncs project code and data to Imperial's GPU cluster.
Features:

- Incremental sync with rsync for speed
- Selective sync of code vs data
- Compression for large files
- Bandwidth optimization
- Resume capability for interrupted transfers

Usage:
    python scripts/sync_to_cluster.py --user your_username --sync-code
    python scripts/sync_to_cluster.py --user your_username --sync-data data/raw
    python scripts/sync_to_cluster.py --user your_username --full-sync
"""

import argparse
import subprocess
import logging
from pathlib import Path
from typing import List, Optional
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ClusterSync:
    """Handles synchronization with Imperial's GPU cluster."""
    
    def __init__(self, username: str, cluster_host: str = "login.hpc.ic.ac.uk"):
        self.username = username
        self.cluster_host = cluster_host
        self.local_project_root = Path(__file__).parent.parent
        self.remote_project_dir = f"/vol/bitbucket/{username}/causal_bayes_opt"
        
        # Sync configuration
        self.rsync_base_args = [
            "rsync", "-avz", "--progress", "--human-readable",
            "--compress-level=6", "--partial", "--append-verify"
        ]
        
        logger.info(f"Cluster sync initialized")
        logger.info(f"Local project: {self.local_project_root}")
        logger.info(f"Remote target: {self.cluster_host}:{self.remote_project_dir}")
    
    def test_connection(self) -> bool:
        """Test SSH connection to cluster."""
        logger.info("Testing cluster connection...")
        
        try:
            result = subprocess.run([
                "ssh", f"{self.username}@{self.cluster_host}", 
                "echo 'Connection successful'"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logger.info("✅ Cluster connection successful")
                return True
            else:
                logger.error(f"❌ Connection failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Connection timeout")
            return False
        except Exception as e:
            logger.error(f"❌ Connection error: {e}")
            return False
    
    def create_remote_structure(self):
        """Create remote directory structure."""
        logger.info("Creating remote directory structure...")
        
        directories = [
            self.remote_project_dir,
            f"{self.remote_project_dir}/data/{{raw,processed,splits,checkpoints}}",
            f"{self.remote_project_dir}/logs/{{collection,training,validation}}",
            f"{self.remote_project_dir}/results/{{models,evaluation,plots}}",
            f"{self.remote_project_dir}/configs"
        ]
        
        for dir_path in directories:
            cmd = [
                "ssh", f"{self.username}@{self.cluster_host}",
                f"mkdir -p {dir_path}"
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to create directory {dir_path}: {e}")
        
        logger.info("✅ Remote directory structure created")
    
    def sync_code(self, exclude_patterns: Optional[List[str]] = None) -> bool:
        """Sync project code to cluster."""
        logger.info("Syncing project code...")
        
        # Default exclusions for code sync
        default_excludes = [
            ".git/", "__pycache__/", "*.pyc", ".pytest_cache/",
            "data/", "logs/", "results/", "demonstrations/",
            "checkpoints/", "*.pkl", "*.h5", "*.npz",
            ".venv/", "venv/", ".env", "*.log",
            ".DS_Store", "Thumbs.db"
        ]
        
        if exclude_patterns:
            default_excludes.extend(exclude_patterns)
        
        # Build rsync command
        cmd = self.rsync_base_args.copy()
        
        # Add exclusions
        for pattern in default_excludes:
            cmd.extend(["--exclude", pattern])
        
        # Add delete flag to remove deleted files
        cmd.append("--delete")
        
        # Source and destination
        cmd.append(f"{self.local_project_root}/")
        cmd.append(f"{self.username}@{self.cluster_host}:{self.remote_project_dir}/")
        
        logger.info(f"Executing: {' '.join(cmd[:5])} ... [with exclusions]")
        
        try:
            result = subprocess.run(cmd, timeout=600)  # 10 minute timeout
            
            if result.returncode == 0:
                logger.info("✅ Code sync completed successfully")
                return True
            else:
                logger.error(f"❌ Code sync failed with exit code: {result.returncode}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Code sync timeout")
            return False
        except Exception as e:
            logger.error(f"❌ Code sync error: {e}")
            return False
    
    def sync_data(self, data_path: str, remote_subdir: str = "data") -> bool:
        """Sync specific data directory to cluster."""
        local_data_path = self.local_project_root / data_path
        
        if not local_data_path.exists():
            logger.error(f"❌ Local data path not found: {local_data_path}")
            return False
        
        logger.info(f"Syncing data: {local_data_path} -> {remote_subdir}")
        
        # Build rsync command for data
        cmd = self.rsync_base_args.copy()
        
        # Add compression for large files
        cmd.extend(["--compress-level=9"])
        
        # Skip some file types that don't compress well
        cmd.extend(["--skip-compress", "gz,bz2,Z,zip,rar,7z,jpg,jpeg,png,mp4,avi"])
        
        # Source and destination
        cmd.append(f"{local_data_path}/")
        cmd.append(f"{self.username}@{self.cluster_host}:{self.remote_project_dir}/{remote_subdir}/")
        
        logger.info(f"Executing: {' '.join(cmd[:5])} ... [data sync]")
        
        try:
            result = subprocess.run(cmd, timeout=3600)  # 1 hour timeout for data
            
            if result.returncode == 0:
                logger.info("✅ Data sync completed successfully")
                return True
            else:
                logger.error(f"❌ Data sync failed with exit code: {result.returncode}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Data sync timeout")
            return False
        except Exception as e:
            logger.error(f"❌ Data sync error: {e}")
            return False
    
    def sync_results_back(self, remote_subdir: str = "results", 
                         local_subdir: str = "results_from_cluster") -> bool:
        """Sync results back from cluster to local machine."""
        local_results_path = self.local_project_root / local_subdir
        local_results_path.mkdir(exist_ok=True)
        
        logger.info(f"Syncing results back: {remote_subdir} -> {local_results_path}")
        
        # Build rsync command (reverse direction)
        cmd = self.rsync_base_args.copy()
        
        # Source (remote) and destination (local)
        cmd.append(f"{self.username}@{self.cluster_host}:{self.remote_project_dir}/{remote_subdir}/")
        cmd.append(f"{local_results_path}/")
        
        logger.info(f"Executing: {' '.join(cmd[:5])} ... [results back]")
        
        try:
            result = subprocess.run(cmd, timeout=1800)  # 30 minute timeout
            
            if result.returncode == 0:
                logger.info("✅ Results sync completed successfully")
                return True
            else:
                logger.error(f"❌ Results sync failed with exit code: {result.returncode}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Results sync timeout")
            return False
        except Exception as e:
            logger.error(f"❌ Results sync error: {e}")
            return False
    
    def check_remote_space(self) -> bool:
        """Check available space on remote cluster."""
        logger.info("Checking remote disk space...")
        
        try:
            result = subprocess.run([
                "ssh", f"{self.username}@{self.cluster_host}",
                f"df -h /vol/bitbucket/{self.username}"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logger.info("Remote disk space:")
                print(result.stdout)
                
                # Parse available space (rough check)
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    fields = lines[1].split()
                    if len(fields) >= 4:
                        available = fields[3]
                        if 'G' in available and float(available.replace('G', '')) < 5:
                            logger.warning("⚠️ Low disk space on cluster (<5GB available)")
                        else:
                            logger.info("✅ Sufficient disk space available")
                
                return True
            else:
                logger.warning(f"Could not check disk space: {result.stderr}")
                return False
                
        except Exception as e:
            logger.warning(f"Disk space check failed: {e}")
            return False
    
    def deploy_environment(self) -> bool:
        """Deploy and setup environment on cluster."""
        logger.info("Deploying environment on cluster...")
        
        try:
            # Run environment setup script
            result = subprocess.run([
                "ssh", f"{self.username}@{self.cluster_host}",
                f"cd {self.remote_project_dir} && bash cluster/scripts/setup_env.sh"
            ], timeout=1800)  # 30 minute timeout for environment setup
            
            if result.returncode == 0:
                logger.info("✅ Environment deployment successful")
                return True
            else:
                logger.error(f"❌ Environment deployment failed")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Environment deployment timeout")
            return False
        except Exception as e:
            logger.error(f"❌ Environment deployment error: {e}")
            return False


def main():
    """CLI entry point for cluster synchronization."""
    parser = argparse.ArgumentParser(
        description="Sync ACBO project to Imperial's GPU cluster",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sync code only
  python scripts/sync_to_cluster.py --user your_username --sync-code
  
  # Sync specific data directory
  python scripts/sync_to_cluster.py --user your_username --sync-data data/raw
  
  # Full sync (code + data)
  python scripts/sync_to_cluster.py --user your_username --full-sync
  
  # Deploy environment
  python scripts/sync_to_cluster.py --user your_username --deploy-env
  
  # Sync results back
  python scripts/sync_to_cluster.py --user your_username --sync-results-back
        """
    )
    
    parser.add_argument("--user", required=True,
                       help="Imperial username for cluster access")
    parser.add_argument("--cluster-host", default="gpucluster2.doc.ic.ac.uk",
                       help="Cluster hostname")
    
    # Sync options
    parser.add_argument("--sync-code", action="store_true",
                       help="Sync project code")
    parser.add_argument("--sync-data", type=str,
                       help="Sync specific data directory")
    parser.add_argument("--sync-results-back", action="store_true",
                       help="Sync results back from cluster")
    parser.add_argument("--full-sync", action="store_true",
                       help="Sync everything (code + common data dirs)")
    parser.add_argument("--deploy-env", action="store_true",
                       help="Deploy and setup environment on cluster")
    
    # Additional options
    parser.add_argument("--exclude", nargs="*",
                       help="Additional patterns to exclude from sync")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be synced without doing it")
    
    args = parser.parse_args()
    
    try:
        # Create sync object
        sync = ClusterSync(args.user, args.cluster_host)
        
        # Test connection
        if not sync.test_connection():
            print("❌ Cannot connect to cluster. Please check:")
            print("  1. SSH key is set up correctly")
            print("  2. You're connected to Imperial network/VPN")
            print("  3. Username is correct")
            return 1
        
        # Check disk space
        sync.check_remote_space()
        
        # Create remote directory structure
        sync.create_remote_structure()
        
        success = True
        
        # Perform requested sync operations
        if args.sync_code or args.full_sync:
            success &= sync.sync_code(args.exclude)
        
        if args.sync_data:
            success &= sync.sync_data(args.sync_data)
        
        if args.full_sync:
            # Sync common data directories if they exist
            common_data_dirs = ["data/raw", "data/processed", "configs"]
            for data_dir in common_data_dirs:
                if (sync.local_project_root / data_dir).exists():
                    success &= sync.sync_data(data_dir)
        
        if args.deploy_env:
            success &= sync.deploy_environment()
        
        if args.sync_results_back:
            success &= sync.sync_results_back()
        
        if success:
            print("\n✅ Sync completed successfully!")
            print(f"\nNext steps:")
            print(f"1. SSH to cluster: ssh {args.user}@{args.cluster_host}")
            print(f"2. Navigate to project: cd /vol/bitbucket/{args.user}/causal_bayes_opt")
            print(f"3. Activate environment: source activate_env.sh")
            print(f"4. Submit jobs: sbatch cluster/jobs/collect_data.sbatch")
        else:
            print("\n❌ Some sync operations failed. Check logs above.")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️ Sync interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Sync failed: {e}")
        logger.exception("Sync failed")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())