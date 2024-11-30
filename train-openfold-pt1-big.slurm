#!/bin/bash
#SBATCH -A m4351
#SBATCH -J of-pt1-training
#SBATCH -o of-pt1-training-%j.out
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 5
#SBATCH --gpus=20
#SBATCH --ntasks-per-node=4
#SBATCH -t 08:00:00
#SBATCH --signal=USR1@60

source ~/.bashrc
scontrol write batch_script ${SLURM_JOB_ID} -
cd /global/cfs/cdirs/m4351/sk844/openfold-pytorch2-instability-issue/openfold
conda activate ../openfold-pytorch1-env
module load cudatoolkit/11.7

export NUM_NODES=$SLURM_NNODES
export LD_LIBRARY_PATH=/global/cfs/cdirs/m4351/sk844/openfold-pytorch2-env/lib:$LD_LIBRARY_PATH
export DATA_DIR=/global/cfs/cdirs/m4351/openfold_reference_datasets/

srun python train_openfold.py $DATA_DIR/pdb_data/mmcif_files $DATA_DIR/alignment_data/alignments \
	$DATA_DIR/pdb_data/mmcif_files \
	/global/cfs/cdirs/m4351/sk844/openfold-pytorch2-instability-issue/output_dir/$1 \
	2021-10-10 \
	--train_chain_data_cache_path $DATA_DIR/pdb_data/data_caches/chain_data_cache.json \
	--template_release_dates_cache_path $DATA_DIR/pdb_data/data_caches/mmcif_cache.json \
	--config_preset initial_training \
	--seed 42 \
	--obsolete_pdbs_file_path $DATA_DIR/pdb_data/obsolete.dat \
	--num_nodes $NUM_NODES \
	--gpus 4 \
	--train_epoch_len 2000 \
	--max_epochs -1 \
	--precision bf16 \
	--log_lr --checkpoint_every_epoch \
	--deepspeed_config_path deepspeed_config.json \
	--val_data_dir /global/cfs/cdirs/m4351/s3_data/validation_set_cameo/cameo/mmcif_files/ \
	--val_alignment_dir /global/cfs/cdirs/m4351/s3_data/validation_set_cameo/cameo/alignments/ \
	--wandb --wandb_entity openfold --wandb_project openfold-pt-upgrade --experiment_name $2
