#!/bin/bash
#
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28
#SBATCH --job-name=voxel
#SBATCH --partition=secondary
#SBATCH --output=myjob.o%j
##SBATCH --error=myjob.e%j
##SBATCH --mail-user=NetID@illinois.edu
##SBATCH --mail-type=BEGIN,END
#SBATCH --mem=100G
#
# End of embedded SBATCH options
#

# Run the hello world executable (a.out)


python voxelize.py

