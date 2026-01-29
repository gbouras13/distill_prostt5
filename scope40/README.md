# Scop40 Benchmark

* To run the scope40 benchmark from a checkpoint



## Step 1 inference of 3Di 

```bash

NUM_HEADS=14
NUM_LAYERS=24
HIDDEN_DIM=448
INTERMEDIATE_DIM=896

distill_prostt5 infer -i all_scope40_new.fasta -o all_scope_infer -m "checkpoint-$CHECKPOINT/" \
 --num_heads $NUM_HEADS --num_layers $NUM_LAYERS --hidden_size $HIDDEN_DIM --intermediate_size $INTERMEDIATE_DIM
```

## Step 2 - create Foldseek db 

* The easiest way to implement this in 1 command is using [Phold](https://github.com/gbouras13/phold) - recommended to install via bioconda

```bash
phold createdb --fasta_aa all_scope40_new.fasta --fasta_3di all_scope_infer/output_3di.fasta -o all_scope40_fs_db -f
```

## Step 3 - Run Foldseek

* Matches the [foldseek-analysis](https://github.com/steineggerlab/foldseek-analysis) repo
* Requires Foldseek to be in $PATH

```bash
mkdir -p rawoutput
THREADS=32
foldseek easy-search all_scope40_fs_db/phold_foldseek_db scop_foldseekdb/scop  rawoutput/mini_vs_pdb tmp/ --threads $THREADS -s 9.5 --max-seqs 2000 -e 10
foldseek easy-search all_scope40_fs_db/phold_foldseek_db all_scope40_fs_db/phold_foldseek_db rawoutput/mini_vs_mini tmp/ --threads $THREADS -s 9.5 --max-seqs 2000 -e 10
```

## Step 4 -  calculate results

```bash
mkdir -p rocx

./bench.awk scop_lookup.fix.tsv <(cat rawoutput/mini_vs_pdb) > rocx/mini_vs_pdb.rocx
./bench.awk scop_lookup.fix.tsv <(cat rawoutput/mini_vs_mini) > rocx/mini_vs_mini.rocx

## calculate auc
awk '{ famsum+=$3; supfamsum+=$4; foldsum+=$5}END{print famsum/NR,supfamsum/NR,foldsum/NR}' rocx/mini_vs_pdb.rocx
awk '{ famsum+=$3; supfamsum+=$4; foldsum+=$5}END{print famsum/NR,supfamsum/NR,foldsum/NR}' rocx/mini_vs_mini.rocx
```

* The results should be printed to the terminal :)

