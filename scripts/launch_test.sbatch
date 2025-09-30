#!/bin/bash
#SBATCH --account=a-g34                   # account to use
#SBATCH --job-name=test-run             # Job name
#SBATCH --output=./slurm_out/job_%j_out.txt   # Output file (job ID and task number)
#SBATCH --error=./slurm_out/job_%j_err.txt    # Error file (job ID and task number)
#SBATCH --partition=normal                # Partitions to submit the job
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1                        # Number of tasks (1 for single task)
#SBATCH --cpus-per-task=128               # Number of CPU cores per task
#SBATCH --mem=256G                        # Memory allocation
#SBATCH --gpus=4                          # Number of GPUs
#SBATCH --time=1:30:00                    # Time limit

# Print detailed job metadata
JOB_TIME_LIMIT=$(squeue -j $SLURM_JOB_ID -h --Format TimeLimit)
CURRENT_TIMESTAMP_FULL=$(date --utc)
CURRENT_TIMESTAMP_SHORT=$(date --utc -d "$CURRENT_TIMESTAMP_FULL" +"%Y%m%d_%H%M%S")
echo "================== JOB METADATA =================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "User ID: $SLURM_JOB_USER"
echo "Job Submit Directory: $SLURM_SUBMIT_DIR"
echo "Node List: $SLURM_JOB_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Number of Nodes: $SLURM_JOB_NUM_NODES"
echo "CPUs per Task: $SLURM_CPUS_PER_TASK"
echo "Memory Allocated: $SLURM_MEM_PER_NODE"
echo "GPUs Allocated: $SLURM_GPUS"
echo "Time Limit: $JOB_TIME_LIMIT"
echo "Job Start Time: $CURRENT_TIMESTAMP_FULL"
echo "==================================================="

ENV_VARS="
  export TQDM_MININTERVAL=120
  # export TQDM_DISABLE=1
"

srun --environment=/users/ljiayong/projects/LIC_TCM/docker/alps/clariden.toml bash -c "
  ${ENV_VARS}
  cd error_bounded_compression_pipeline
  pip install zarr
  pip install -e .
  python ./scripts/test.py
"
