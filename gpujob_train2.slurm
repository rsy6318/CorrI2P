#!/bin/bash
#SBATCH --partition=gpu_7d1g
#SBATCH --nodes=1                # 1 computer nodes
#SBATCH --ntasks-per-node=1      # 1 MPI tasks on EACH NODE
#SBATCH --cpus-per-task=6        # 6 OpenMP threads on EACH MPI TASK
#SBATCH --gres=gpu:1             # Using 1 GPU card
#SBATCH --mem=192GB               # Request 50GB memory
#SBATCH --time=7-00:00:00        # Time limit day-hrs:min:sec
#SBATCH --output=./error_out/gpujob_%j.log    # Standard output
#SBATCH --error=./error_out/gpujob_%j.err    # Standard error log



cd /gpfs1/home/siyuren2/DenseI2P_softmax/


/gpfs1/home/siyuren2/anaconda3/envs/mink/bin/python ./train.py  --dist_thres=1 --pos_margin=0 --neg_margin=1.4
/gpfs1/home/siyuren2/anaconda3/envs/mink/bin/python ./train.py  --dist_thres=1 --pos_margin=0 --neg_margin=1.8
/gpfs1/home/siyuren2/anaconda3/envs/mink/bin/python ./train.py  --dist_thres=2 --pos_margin=0 --neg_margin=1.4
/gpfs1/home/siyuren2/anaconda3/envs/mink/bin/python ./train.py  --dist_thres=2 --pos_margin=0 --neg_margin=1.8