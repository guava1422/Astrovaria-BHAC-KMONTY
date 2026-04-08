#!/bin/bash
#SBATCH -t 0-02:00:00
#SBATCH --job-name=MC
#SBATCH --ntasks=48
#SBATCH --ntasks-per-node=48
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -p all
#SBATCH -o out
#SBATCH -e err
module purge
module load os
module load mpi

export OMPI_MCA_mtl='^ofi'
export OMPI_MCA_btl='^openib,ofi'

$BHAC_DIR/setup.pl -arch=gfortran
make clean bhac

mkdir output
# only works on headnode:
#cp /tmp/rad-BHAC/MAD/a+15o16/output/data2000.dat output
cp definitions.h particles.par amrvacusr.t mod_particles_user.t output

mpirun -np $SLURM_NTASKS ./bhac -restart 2000 -i particles.par
