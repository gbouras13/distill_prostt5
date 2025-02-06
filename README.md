# distill_prostt5
Distillation Commands for ProstT5

# Install

```
git clone https://github.com/gbouras13/distill_prostt5
cd distill_prostt5
pip install -e . 
```

## Step 1 - precompute ProstT5 embeddings

* This takes an amino acid protein FASTA file with corresponding 3Di FASTA as input, and will write out a `.h5` file with the ProstT5-CNN logits and the tokenized input

e.g.

```bash
distill_prostt5 precompute -i tests/test_data/phrog_3922_db_aa.fasta -c tests/test_data/phrog_3922_db_ss.fasta -p phrog_3922_db_aa.h5
distill_prostt5 precompute -i tests/test_data/swissprot_subset_aa_500.fasta -c tests/test_data/swissprot_subset_ss_500.fasta -p swissprot_subset_aa_500.h5
```

## Step 2 - merge ProstT5 embeddings

* This takes a directory of `.h5` files made in step 1 as input and will write out a single `.h5` file (for training)
* Idea is to batch on e.g. Pawsey for preparing a full training dataset

```bash
distill_prostt5 merge -d tests/test_h5s/ -p merged.h5
```

## Step 3 - train 

* Trains mini ProstT5 distilled model
* Uses the ModernBertModel https://huggingface.co/docs/transformers/en/model_doc/modernbert#transformers.ModernBertModel
* 11M params - you can see the exact architecutre in `distill_prostt5/classes/MPROSTT5_bert.py`
* It is vanilla with 6 layers, 8 attention heads, and hidden dimension of 512
* Will ablate this
* The training loss is `loss = (1-alpha)* kl_loss + alpha * ce_loss` (`alpha` is 0.3 for now, will ablate)
    * `kl_loss` is vs the ProstT5-CNN logits 
    * `ce_loss` is vs the colabfold 3Di "ground truth"



```bash
distill_prostt5 train -p swissprot_subset_aa_500.h5 -e phrog_3922_db_aa.h5  -o test_out_500 
```

```bash
Usage: distill_prostt5 train [OPTIONS]

  Trains distilled Mini ProstT5 model

Options:
  -h, --help                Show this message and exit.
  -p, --train_path PATH     Path to .h5 file containing training data
                            processed with the precompute subcommand
                            [required]
  -e, --eval_path PATH      Path to .h5 file containing evaluation data
                            processed with the precompute subcommand
                            [required]
  -o, --output_dir PATH     Output directory where checkpoints will be saved
                            [required]
  -m, --model_ckpt PATH     Model checkpoint directory (to restart training
                            from here)
  -b, --batch_size INTEGER  Batch size
  -e, --epochs INTEGER      Epochs
  ```