# distill_prostt5

* This is a somewhat experimental training and inference repository for creating ModernProst 
    * ModernProst is a [ModernBERT](https://arxiv.org/abs/2412.13663)-based protein language model that directly translates Amino Acid tokens to 3Di 
    * The final model is 48M parameters and is available as [modernprost-base](https://huggingface.co/gbouras13/modernprost-base) which translates AA->3Di 
    * We also make available [modernprost-profile]() which infers a predicted 3Di PSSM directly from AA input. This model was post-trained on top of modernprost-base.
* Note: You can also use this repository to directly distill [ProstT5](https://github.com/mheinzinger/ProstT5) although no distillation was actually used in the final ModernProst model as training was not substantially different to training `modernprost-base` from scratch
* Note: There are other experimental features/parameters (e.g. anything involing `plddt`) that may be included that are not related to the training of `modernprost`. Please use them with caution.
* For directly replication of the exact `modernprost-base` training commands and versions, please see the `replication` subdirectory
* More to come

## Installation

```
git clone https://github.com/gbouras13/distill_prostt5
cd distill_prostt5
pip install -e . 
```

## Training

* Training is conducted in 2 steps: (1) dataset pre-computation (2) training

## Example - Training Model From Scratch (no distillation)

* This is how `modernprost-base` was trained

### Step 1 - precompute input data

* This takes an amino acid protein FASTA file with corresponding 3Di FASTA (from AF2/ColabFold structures) as inputs, and will write out a `.h5` file with tokenized input ready for training
* To save a precomputed dataset for training from scratch, recommended to only use `--no_logits`.
    * This means you can do it all on CPU as no ProstT5 logs will be generated
    * To precompute for distillation, do not use this - but you will need a GPU available for fast ProstT5 inference
* e.g. to save your dataset in maximum chunks of 512AA

```bash
distill_prostt5 precompute --no_logits -i tests/test_data/phrog_3922_db_aa.fasta -c tests/test_data/phrog_3922_db_ss.fasta  -p test.h5 -m 512
distill_prostt5 precompute --no_logits -i tests/test_data/swissprot_subset_aa_500.fasta -c tests/test_data/swissprot_subset_ss_500.fasta -p swissprot_subset_aa_500.h5
```

```bash
Usage: distill_prostt5 precompute [OPTIONS]

  precomputes training datasets

Options:
  -h, --help                      Show this message and exit.
  -i, --input PATH                Path to protein amino acid input file in
                                  FASTA format  [required]
  -c, --colabfold PATH            Path to 3Di colabfold input file in FASTA
                                  format
  -p, --precompute_path PATH      Path to output file where you want to save
                                  hdf5 embeddings and other data required for
                                  the distillation. Use suffix .h5 (for use
                                  with merge)  [required]
  -m, --max_length INTEGER        Max length of input (sequences above this
                                  length will be truncated to this many
                                  characters).
  --no_logits                     Only tokenize & randomly crop sequences, do
                                  not embed and calculate logits.
  --pssm_dtype [float16|float32]  (pssm only) dtype used to store labels in
                                  HDF5  [default: float16]
  --pssm_chunk_size INTEGER       (pssm only) chunked write size to HDF5
                                  [default: 4096]
  --task [classification|pssm]    classification: AA FASTA + 3Di FASTA ->
                                  precomputed HDF5 for 3Di distillation pssm:
                                  prefix to
                                  .profiles.bin/.seqs.bin/.profiles.idx ->
                                  precomputed HDF5 for PSSM training
                                  [default: classification]
```

## Step 2 - train 

* Trains mini ProstT5 distilled model
* Use `-a 1 --no_logits` to enforce only Cross-Entropy loss (`-a` controls the contribution of Cross-Entropy vs KL loss, if using ProstT5 logits for distillation)
* 11M params by default - you can see the exact architecutre in `distill_prostt5/classes/MPROSTT5_bert.py`
* It is vanilla with 6 layers, 8 attention heads, and hidden dimension of 512 and intermediate size of 512 (approx 11M params)
    * Note: `modernprost` has 14 attention heads, 24 layers, hidden dimension of 448 and an intermediate dimension of 896 (approx 48M params)


```bash
 distill_prostt5 train -p test.h5 -e swissprot_subset_aa_500.h5  -o toy_model_output --no_logits  -a 1
```

```bash

Usage: distill_prostt5 train [OPTIONS]

  Trains distilled Mini ProstT5 model

Options:
  -h, --help                      Show this message and exit.
  -p, --train_path PATH           Path to .h5 file containing training data
                                  processed with the precompute subcommand
                                  [required]
  -e, --eval_path PATH            Path to .h5 file containing evaluation data
                                  processed with the precompute subcommand
                                  [required]
  -o, --output_dir PATH           Output directory where checkpoints will be
                                  saved   [required]
  -m, --model_ckpt PATH           Model checkpoint directory (to restart
                                  training from here)
  -b, --batch_size INTEGER        Batch size per device - 192 can fit in MI250
                                  GPU memory
  --epochs INTEGER                Epochs
  -a, --alpha FLOAT               Weighted contribution of Colabfold CE loss
                                  to total loss
  --activation TEXT               activation type - choose gelu or swiglu,
                                  defaults to swiglu
  --num_layers INTEGER            Number of layers (default to 6)
  --num_heads INTEGER             Number of attention heads (default to 8)
  --hidden_size INTEGER           Hidden size (default to 512)
  --intermediate_size INTEGER     Intermediate size size (default to 512)
  --learning_rate FLOAT           learning rate (default to 3e-4)
  --save_steps INTEGER            Save checkpoint this many steps (default to
                                  1000)
  --logging_eval_steps INTEGER    Eval and log at this many steps (default to
                                  25)
  --num_workers INTEGER           Number of workers for dataloader (default to
                                  1)
  --warmup_ratio FLOAT            warmup ratio
  --no_logits                     Only tokenize & randomly crop sequences, do
                                  not embed and calculate logits.
  --lr_scheduler_type [linear|cosine|cosine_with_restarts|polynomial|constant|constant_with_warmup|inverse_sqrt|reduce_lr_on_plateau|cosine_with_min_lr|warmup_stable_decay]
                                  Type of learning rate scheduler to use.
                                  [default: linear]
  --initialise                    Use a smaller model to initialise the
                                  weights of the larger model
  --base_path PATH                Base model path
  --base_activation TEXT          base model activation type - choose gelu or
                                  swiglu, defaults to swiglu
  --base_num_layers INTEGER       Number of layers (default to 6)
  --base_num_heads INTEGER        Number of attention heads (default to 8)
  --base_hidden_size INTEGER      Hidden size (default to 512)
  --base_intermediate_size INTEGER
                                  Base model Intermediate size size (default
                                  to 512)
  --restart                       Restart training from a checkpoint but
                                  different training args
  --restart_path PATH             Restart model path
  --use_focal                     Use focal loss
  --gamma FLOAT                   Focal loss gamma - controls how much you
                                  down weight easy samples. Gamma = 0 is cross
                                  entropy loss
  --no_reweight                   No rewighting classes for focal loss
  --step_down                     Changes single layer projection
                                  (hidden_size, 20) to a 2-layer step down
                                  with SWIglu activation and intermediate
                                  dimension hidden_size // step_down_ratio
  --step_down_ratio INTEGER       Controls the intermediate dimension in the
                                  2-layer step down intermediate dimension
                                  hidden_size // step_down_ratio
  --debug_overfit                 Enable overfitting on the first data point
                                  for debugging.
  --task [classification|pssm]    Task type: classification for discrete 3Di,
                                  pssm for 20-dimensional PSSM profile output
```













 ## Step 4 - infer

 ```bash
 distill_prostt5 infer -i tests/test_data/swissprot_subset_aa_50.fasta -o test_infer -m checkpoint-308000/
 ```

 ```bash
Usage: distill_prostt5 infer [OPTIONS]

  Infers 3Di from input AA FASTA

Options:
  -h, --help                    Show this message and exit.
  -i, --input PATH              Input FASTA  [required]
  -o, --output_dir PATH         Output directory  [required]
  -m, --model_ckpt PATH         Model checkpoint directory (to predict 3Di
                                using this)
  --num_layers INTEGER          Number of layers (default to 6)
  --num_heads INTEGER           Number of attention heads (default to 8)
  --hidden_size INTEGER         Hidden size (default to 512)
  --intermediate_size INTEGER   Intermediate size size (default to 512)
  --mask_threshold INTEGER      Mask residues below this confidence threshold
                                - between 0 and 100
  --cpu                         Use cpus only.
  --phold                       Phold output format.
  --step_down                   Changes single layer projection (hidden_size,
                                20) to a 2-layer step down with SWIglu
                                activation and intermediate dimension
                                hidden_size // step_down_ratio
  --step_down_ratio INTEGER     Controls the intermediate dimension in the
                                2-layer step down intermediate dimension
                                hidden_size // step_down_ratio
  --plddt_head                  Infer with plddt head
  --batch_size INTEGER          Controls inference batchsize
  --profile_mmseqs_db FILE      Prefix path of the MMseqs2 sequence database
                                for the input, e.g. '/path/to/db' (so that
                                '/path/to/db.index' and '/path/to/db.lookup'
                                exist). Used to write Foldseek profile_ss DB
                                directly when --task pssm.
  --task [classification|pssm]  Task type: classification for discrete 3Di,
                                pssm for 20-dimensional PSSM profile output
  --max_residues INTEGER        Max residues per batch
  --half                        Use half precision
  --fast                        try fast inference
  --autobatch                   Autobatch
  --step INTEGER                step size for autobatch
  --max_batch INTEGER           max batch size for autobatch
  --sample_seqs INTEGER         number of seqs to subsample for autobatch
  --chunk_len INTEGER           chunk length
  --threads INTEGER             number of threads (only for cpu mode)
  ```


# 25 August 2025.  


* Note that is is best to build the container on top of the known working v0.4.1 container. Pytorch nightly builds caused strange, non-convergent behaviour for updated training runs

# 21 Dec

* Update to building on top of Pawsey pytorch 2.7.1 container

* Try context extension -> 

```bash

* Actual command

```bash
python scripts/filter_fastas.py -c 10000clusters.fasta -i fasta/prostT5_dataset.fasta -o prostT5_filt_aa.fasta
python scripts/filter_fastas.py -c 10000clusters.fasta -i fasta/prostT5_dataset_ss.fasta  -o prostT5_filt_ss.fasta

distill_prostt5 precompute --no_logits -i prostT5_filt_aa.fasta -c prostT5_filt_ss.fasta -p prostT5_training.h5
distill_prostt5 precompute --no_logits -i 10000clusters.fasta -c 10000clusters_ss.fasta -p prostT5_validation.h5
```

module  load pawseyenv/2023.08
module load singularity/3.11.4-slurm
containerImage="distill_prostt5_0.4.1.sif"

singularity exec --rocm  $containerImage distill_prostt5 precompute --help



python ../distill_prostt5/scripts/extend_dataset_context_length.py -i prostT5_filt_aa.fasta -o prostT5_filt_aa_context_length_extension.fasta -l 768

python ../distill_prostt5/scripts/extend_dataset_context_length.py -i prostT5_filt_ss.fasta -o prostT5_filt_ss_context_length_extension.fasta -l 768



singularity exec --rocm  $containerImage  distill_prostt5 precompute --no_logits -i prostT5_filt_aa_context_length_extension.fasta -c prostT5_filt_ss_context_length_extension.fasta -p prostT5_training_2048.h5 -m 2048

singularity exec --rocm  $containerImage  distill_prostt5 precompute --no_logits -i 10000clusters.fasta -c 10000clusters_ss.fasta -p prostT5_validation_2048.h5 -m 2048

```

