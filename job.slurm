#!/bin/sh
#SBATCH --partition=WORKQ
#SBATCH --time=00:01:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name="run_test"
#SBATCH --output=output/test-srun-%A_%a.out
#SBATCH --mail-user=pcthrane@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --array=0-9%4
 
WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "We are running from this directory: $SLURM_SUBMIT_DIR"
echo "The name of the job is: $SLURM_JOB_NAME"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Cores used: $SLURM_CPUS_ON_NODE"
echo "Cores per node: $SLURM_CPUS_ON_NODE"
echo "Total number of cores: $SLURM_NTASKS"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "The job ID is: $SLURM_JOB_ID"
 
module purge
module load foss/2016a
module load Python/3.5.1
source ../pythonenvs/S4_environment/bin/activate

#Run script
time srun python job_test.py -p 10 -n $SLURM_ARRAY_TASK_ID


#deactivate
#module purge
