#!/bin/sh
#SBATCH --partition=WORKQ
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1000
#SBATCH --job-name="S4_NB300"
#SBATCH --output="output/arbabi_ellipse_NB300_76x76_h500/arbabi_ellipse_NB300_76x76_h500-%A_%a.out"
#SBATCH --mail-user=pcthrane@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --array=0-9%5
 
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

export OPENBLAS_NUM_THREADS=1
 
module purge
module load foss/2016a
module load Python/3.5.1
source /lustre1/work/pcthrane/pythonenvs/S4_environment/bin/activate

#Run script
time srun python job.py -p 10 -n $SLURM_ARRAY_TASK_ID
#time srun python job.py -p 1 -n 0
