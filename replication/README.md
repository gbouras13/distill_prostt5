# Replication

* All base models were trained on [Setonix](https://pawsey.org.au/systems/setonix/) using GPU-highmem nodes consisting of 8x64GB AMD MI250x GPUs
* All base models except 1.2B were trained with 1 node. 1.2B was trained with 8 nodes.
* All base models were trained using this specific v0.4.1 container build available [here](https://quay.io/repository/gbouras13/distill_prostt5?tab=tags)
* The profile model was trained on 6x NVIDIA L40s GPUs using a development code branch that was eventually incorporated in v1.0.0

## Dataset Preprocessing

## base models (AA -> 3Di prediction)

* You must first download and ungzip the required FASTAs from HuggingFace [here](https://huggingface.co/datasets/gbouras13/ModernProst-base) containing the ~19M proteins containing 4.3 million non-singleton clusters, which, when expanded to up to 20 most diverse members
* You can of course simply use the preprocessed `.h5` files there also

```bash

containerImage="distill_prostt5_0.4.1.sif"

singularity exec $containerImage distill_prostt5 precompute --no_logits -i prostT5_filt_aa.fasta  -c prostT5_filt_ss.fasta  -p prostT5_training.h5 -m 512

singularity exec $containerImage  distill_prostt5 precompute --no_logits -i 10000clusters.fasta -c 10000clusters_ss.fasta -p prostT5_validation.h5 -m 512
```

## modernprost-profiles (AA -> 3Di PSSM)

* This uses 3Di PSSMs computed using a Foldseek search of the 4.3M non-singleton cluster representatives of the training dataset against AFDB50.
* This is available on HuggingFace to download as `afdb_combined_nosingletons_out_pssm.h5` file [here](https://huggingface.co/datasets/Victor1306/ModernProst-profiles)

## Training - modernprost-base

```bash
NUM_HEADS=14
NUM_LAYERS=24
BATCH_SIZE=72
HIDDEN_DIM=448
INTERMEDIATE_DIM=896
WARMUP_RATIO="0.05"
OUTDIR="50M"

srun --time=23:40:00 -N1 -n1 -c64 --gres=gpu:8 singularity exec --env NCCL_DEBUG=INFO --rocm  $containerImage \
        python distill_prostt5/run.py train -p prostT5_training.h5 -e  prostT5_validation.h5  -o $OUTDIR --learning_rate 8e-4  --no_logits --warmup_ratio $WARMUP_RATIO  -a 1 --epochs 10 -b $BATCH_SIZE --logging_eval_steps 2500  --num_workers 8 --num_heads $NUM_HEADS --num_layers $NUM_LAYERS --hidden_size $HIDDEN_DIM --intermediate_size $INTERMEDIATE_DIM --save_steps 5000 
        
        # to continue training add the path to the checkpoint e.g. -m checkpoint-${CHECKPOINT}
```

## Profiles

* Note that `--task pssm` randomly samples 1000 cluster 3Di PSSMs to use as a validation dataset (so there is no separate `.h5` file)

```bash

NUM_HEADS=14
NUM_LAYERS=24
BATCH_SIZE=50
HIDDEN_DIM=448
INTERMEDIATE_DIM=896
WARMUP_RATIO="0.05"
OUTDIR="50M_profiles"

# restart_path - this is the checkpoint used for `modernprost-base` and the output of above

torchrun --nproc_per_node=6 --nnodes=1 run.py train --task pssm -p afdb_combined_nosingletons_out_pssm.h5 -e afdb_combined_nosingletons_out_pssm.h5 \
 -o $OUTDIR --restart --restart_path 50M/checkpoint-334150 \
 --num_heads $NUM_HEADS --num_layers $NUM_LAYERS --hidden_size $HIDDEN_DIM --intermediate_size $INTERMEDIATE_DIM --batch_size $BATCH_SIZE --logging_eval_steps 1000 --save_steps 10000
```

## Other Models

### 345M

```bash
NUM_HEADS=16
NUM_LAYERS=28
BATCH_SIZE=48
HIDDEN_DIM=1024
INTERMEDIATE_DIM=2624
WARMUP_RATIO="0.05"
OUTDIR="345M"

srun --time=23:40:00 -N1 -n1 -c64 --gres=gpu:8 singularity exec  --env NCCL_DEBUG=INFO --rocm  $containerImage \
	python run.py train -p prostT5_training.h5 -e  prostT5_validation.h5  -o $OUTDIR --learning_rate 8e-4  --no_logits --warmup_ratio $WARMUP_RATIO  -a 1 --epochs 12 -b $BATCH_SIZE --logging_eval_steps 5000  --num_workers 8 --num_heads $NUM_HEADS --num_layers $NUM_LAYERS --hidden_size $HIDDEN_DIM --intermediate_size $INTERMEDIATE_DIM --save_steps 1000  # -m checkpoint-${CHECKPOINT}
```

### 176M

```bash

NUM_HEADS=14
NUM_LAYERS=24
BATCH_SIZE=72
HIDDEN_DIM=896
INTERMEDIATE_DIM=1536
WARMUP_RATIO="0.05"
OUTDIR="176M"

srun --time=23:40:00 -N1 -n1 -c64 --gres=gpu:8 singularity exec --env NCCL_DEBUG=INFO --rocm  $containerImage \
	python run.py train -p prostT5_training.h5 -e  prostT5_validation.h5  -o $OUTDIR --learning_rate 8e-4  --no_logits --warmup_ratio $WARMUP_RATIO  -a 1 --epochs 12 -b $BATCH_SIZE --logging_eval_steps 5000  --num_workers 8 --num_heads $NUM_HEADS --num_layers $NUM_LAYERS --hidden_size $HIDDEN_DIM --intermediate_size $INTERMEDIATE_DIM --save_steps 1000  # -m checkpoint-${CHECKPOINT}


```

### 110M 

```bash
NUM_HEADS=12
NUM_LAYERS=22
BATCH_SIZE=90
HIDDEN_DIM=768
INTERMEDIATE_DIM=1152
WARMUP_RATIO="0.1"
OUTDIR="${NUM_HEADS}_${NUM_LAYERS}_${BATCH_SIZE}_${HIDDEN_DIM}_${INTERMEDIATE_DIM}_from_scratch_${WARMUP_RATIO}"

srun --time=23:40:00 -N1 -n1 -c64 --gres=gpu:8 singularity exec  --env NCCL_DEBUG=INFO --rocm  $containerImage \
	python run.py train -p prostT5_training.h5 -e  prostT5_validation.h5  -o $OUTDIR --learning_rate 8e-4  --no_logits --warmup_ratio $WARMUP_RATIO  -a 1 --epochs 12 -b $BATCH_SIZE --logging_eval_steps 1000  --num_workers 8 --num_heads $NUM_HEADS --num_layers $NUM_LAYERS --hidden_size $HIDDEN_DIM --intermediate_size $INTERMEDIATE_DIM --save_steps 1000  # -m checkpoint-${CHECKPOINT}
```

### 87M

```bash


NUM_HEADS=14
NUM_LAYERS=28
BATCH_SIZE=72
HIDDEN_DIM=560
INTERMEDIATE_DIM=1120
WARMUP_RATIO="0.05"
OUTDIR="87M"

srun --time=23:40:00 -N1 -n1 -c64 --gres=gpu:8 singularity exec --env NCCL_DEBUG=INFO --rocm  $containerImage \
	python run.py train -p prostT5_training.h5 -e  prostT5_validation.h5  -o $OUTDIR --learning_rate 8e-4  --no_logits --warmup_ratio $WARMUP_RATIO  -a 1 --epochs 10 -b $BATCH_SIZE --logging_eval_steps 2500  --num_workers 8 --num_heads $NUM_HEADS --num_layers $NUM_LAYERS --hidden_size $HIDDEN_DIM --intermediate_size $INTERMEDIATE_DIM --save_steps 5000 #  -m checkpoint-${CHECKPOINT} 
```

### 75M

```bash
NUM_HEADS=12
NUM_LAYERS=32
BATCH_SIZE=72
HIDDEN_DIM=480
INTERMEDIATE_DIM=960
WARMUP_RATIO="0.05"
OUTDIR="75M"

srun --time=23:40:00 -N1 -n1 -c64 --gres=gpu:8 singularity exec  --env NCCL_DEBUG=INFO --rocm  $containerImage \
	python run.py train -p prostT5_training.h5 -e  prostT5_validation.h5  -o $OUTDIR --learning_rate 8e-4  --no_logits --warmup_ratio $WARMUP_RATIO  -a 1 --epochs 10 -b $BATCH_SIZE --logging_eval_steps 2500  --num_workers 8 --num_heads $NUM_HEADS --num_layers $NUM_LAYERS --hidden_size $HIDDEN_DIM --intermediate_size $INTERMEDIATE_DIM --save_steps 5000  # -m checkpoint-${CHECKPOINT} 
```

### 25M


```bash
NUM_HEADS=10
NUM_LAYERS=22
BATCH_SIZE=100
HIDDEN_DIM=320
INTERMEDIATE_DIM=640
WARMUP_RATIO="0.05"
OUTDIR="25M"

singularity exec --rocm $containerImage python -c "import torch; print(torch.cuda.device_count())"
srun --time=23:40:00 -N1 -n1 -c64 --gres=gpu:8 singularity exec  --env NCCL_DEBUG=INFO --rocm  $containerImage \
	python run.py train -p prostT5_training.h5 -e  prostT5_validation.h5  -o $OUTDIR --learning_rate 8e-4  --no_logits --warmup_ratio $WARMUP_RATIO  -a 1 --epochs 10 -b $BATCH_SIZE --logging_eval_steps 2500  --num_workers 8 --num_heads $NUM_HEADS --num_layers $NUM_LAYERS --hidden_size $HIDDEN_DIM --intermediate_size $INTERMEDIATE_DIM --save_steps 5000  # -m checkpoint-${CHECKPOINT} 
```

### 15M

```bash
NUM_HEADS=16
NUM_LAYERS=26
BATCH_SIZE=100
HIDDEN_DIM=256
INTERMEDIATE_DIM=384
WARMUP_RATIO="0.05"
OUTDIR="15M"

#singularity exec --rocm $containerImage rocminfo
singularity exec --rocm $containerImage python -c "import torch; print(torch.cuda.device_count())"
srun --time=23:40:00 -N1 -n1 -c64 --gres=gpu:8 singularity exec --env NCCL_DEBUG=INFO --rocm  $containerImage \
	python run.py train -p prostT5_training.h5 -e  prostT5_validation.h5  -o $OUTDIR --learning_rate 8e-4  --no_logits --warmup_ratio $WARMUP_RATIO  -a 1 --epochs 10 -b $BATCH_SIZE --logging_eval_steps 2500  --num_workers 8 --num_heads $NUM_HEADS --num_layers $NUM_LAYERS --hidden_size $HIDDEN_DIM --intermediate_size $INTERMEDIATE_DIM --save_steps 5000 # -m checkpoint-${CHECKPOINT} 
```

### 10M

```bash
NUM_HEADS=10
NUM_LAYERS=20
BATCH_SIZE=100
HIDDEN_DIM=240
INTERMEDIATE_DIM=360
WARMUP_RATIO="0.05"
OUTDIR="10M"

singularity exec --rocm $containerImage python -c "import torch; print(torch.cuda.device_count())"
srun --time=23:40:00 -N1 -n1 -c64 --gres=gpu:8 singularity exec  --env NCCL_DEBUG=INFO --rocm  $containerImage \
	python run.py train -p prostT5_training.h5 -e prostT5_validation.h5  -o $OUTDIR --learning_rate 8e-4  --no_logits --warmup_ratio $WARMUP_RATIO  -a 1 --epochs 10 -b $BATCH_SIZE --logging_eval_steps 2500  --num_workers 8 --num_heads $NUM_HEADS --num_layers $NUM_LAYERS --hidden_size $HIDDEN_DIM --intermediate_size $INTERMEDIATE_DIM --save_steps 5000  # -m checkpoint-${CHECKPOINT} 
```

### 5M

```bash
NUM_HEADS=10
NUM_LAYERS=22
BATCH_SIZE=100
HIDDEN_DIM=160
INTERMEDIATE_DIM=240
WARMUP_RATIO="0.05"
OUTDIR="5M"

singularity exec --rocm $containerImage python -c "import torch; print(torch.cuda.device_count())"
srun --time=23:40:00 -N1 -n1 -c64 --gres=gpu:8 singularity exec s--env NCCL_DEBUG=INFO --rocm  $containerImage \
	python run.py train -p prostT5_training.h5 -e  prostT5_validation.h5  -o $OUTDIR --learning_rate 8e-4  --no_logits --warmup_ratio $WARMUP_RATIO  -a 1 --epochs 10 -b $BATCH_SIZE --logging_eval_steps 2500  --num_workers 8 --num_heads $NUM_HEADS --num_layers $NUM_LAYERS --hidden_size $HIDDEN_DIM --intermediate_size $INTERMEDIATE_DIM --save_steps 5000 #  -m checkpoint-${CHECKPOINT} 
```

### 5M

```bash
NUM_HEADS=10
NUM_LAYERS=22
BATCH_SIZE=100
HIDDEN_DIM=160
INTERMEDIATE_DIM=240
WARMUP_RATIO="0.05"
OUTDIR="5M"

singularity exec --rocm $containerImage python -c "import torch; print(torch.cuda.device_count())"
srun --time=23:40:00 -N1 -n1 -c64 --gres=gpu:8 singularity exec s--env NCCL_DEBUG=INFO --rocm  $containerImage \
	python run.py train -p prostT5_training.h5 -e  prostT5_validation.h5  -o $OUTDIR --learning_rate 8e-4  --no_logits --warmup_ratio $WARMUP_RATIO  -a 1 --epochs 10 -b $BATCH_SIZE --logging_eval_steps 2500  --num_workers 8 --num_heads $NUM_HEADS --num_layers $NUM_LAYERS --hidden_size $HIDDEN_DIM --intermediate_size $INTERMEDIATE_DIM --save_steps 5000 #  -m checkpoint-${CHECKPOINT} 
```

### 1M

```bash
NUM_HEADS="6"
NUM_LAYERS="14"
HIDDEN_DIM="96"
INTERMEDIATE_DIM="144"
BATCH_SIZE="512"
WARMUP_RATIO="0.005"

srun --time=23:40:00 -N1 -n1 -c64 --gres=gpu:8 singularity exec s--env NCCL_DEBUG=INFO --rocm  $containerImage \
	python run.py train -p prostT5_training.h5 -e  prostT5_validation.h5  -o $OUTDIR --learning_rate 8e-4  --no_logits --warmup_ratio $WARMUP_RATIO  -a 1 --epochs 10 -b $BATCH_SIZE --logging_eval_steps 2500  --num_workers 8 --num_heads $NUM_HEADS --num_layers $NUM_LAYERS --hidden_size $HIDDEN_DIM --intermediate_size $INTERMEDIATE_DIM --save_steps 5000 #  -m checkpoint-${CHECKPOINT} 
```




### 1.2B

* 8 nodes were used

```bash

NUM_HEADS=20
NUM_LAYERS=50
BATCH_SIZE=8
HIDDEN_DIM=1520
INTERMEDIATE_DIM=3600
WARMUP_RATIO="0.025"
OUTDIR="1B_8node"

export NUM_PYTORCH_PROCESSES=8
echo "NUM_PYTORCH_PROCESSES='$NUM_PYTORCH_PROCESSES'"

export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export RCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

srun -c 64 singularity exec   --rocm $containerImage torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$NUM_PYTORCH_PROCESSES \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$RDZV_HOST:$RDZV_PORT" \
    distill_prostt5/run.py train -p prostT5_training.h5 -e  prostT5_validation.h5  -o $OUTDIR --learning_rate 5e-4  --no_logits --warmup_ratio $WARMUP_RATIO  -a 1 --epochs 50 -b $BATCH_SIZE --logging_eval_steps 5000  --num_workers 8 --num_heads $NUM_HEADS --num_layers $NUM_LAYERS --hidden_size $HIDDEN_DIM --intermediate_size $INTERMEDIATE_DIM --save_steps 500  # -m $CKPT_BASE/${CHECKPOINT}
```



