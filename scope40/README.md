# 13 May 2025

* Update to use the scop DB in `scop_foldseekdb` from the Steinegger lab server and Foldseek v10.941cd33 (bioconda) 
* Also added the ability to mask residue that are low confidence

```bash
# whatever the checkpoint is
THRESHOLD="25000"
conda activate pholdENV
foldseek convert2fasta scop_foldseekdb/scop all_scope40_new.fasta

# inference of 3Di from the mini models
conda activate distill_prostt5
distill_prostt5 infer -i all_scope40_new.fasta -o all_scope_infer -m "../checkpoint-$THRESHOLD/" --num_heads 16 --num_layers 24 --hidden_size 960 
distill_prostt5 infer -i all_scope40_new.fasta -o all_scope_infer_25 -m "../checkpoint-$THRESHOLD/" --num_heads 16 --num_layers 24 --hidden_size 960 --mask_threshold 25

# ProstT5 inference
conda activate pholdENV
phold proteins-predict -i all_scope40_new.fasta -d ../../phold_db -t 1 -o all_scope_phold_proteins_predict -f --mask_threshold 0 
phold proteins-predict -i all_scope40_new.fasta -d ../../phold_db -t 1 -o all_scope_phold_proteins_predict_25 -f --mask_threshold 25

# Foldseek db creation
phold createdb --fasta_aa all_scope40_new.fasta --fasta_3di all_scope_infer/output_3di.fasta -o all_scope40_fs_db -f
phold createdb --fasta_aa all_scope40_new.fasta --fasta_3di all_scope_infer_25/output_3di.fasta -o all_scope40_fs_db_25 -f

phold createdb --fasta_aa all_scope40_new.fasta --fasta_3di all_scope_phold_proteins_predict/phold_3di.fasta -o all_scope40_prostt5_fs_db -f
phold createdb --fasta_aa all_scope40_new.fasta --fasta_3di all_scope_phold_proteins_predict_25/phold_3di.fasta -o all_scope40_prostt5_fs_db_25 -f


# Run foldseek
mkdir -p rawoutput
THREADS=32
foldseek easy-search all_scope40_fs_db/phold_foldseek_db scop_foldseekdb/scop  rawoutput/mini_vs_pdb tmp/ --threads $THREADS -s 9.5 --max-seqs 2000 -e 10
foldseek easy-search all_scope40_fs_db/phold_foldseek_db all_scope40_fs_db/phold_foldseek_db rawoutput/mini_vs_mini tmp/ --threads $THREADS -s 9.5 --max-seqs 2000 -e 10
foldseek easy-search all_scope40_fs_db_25/phold_foldseek_db scop_foldseekdb/scop  rawoutput/mini_vs_pdb_25 tmp/ --threads $THREADS -s 9.5 --max-seqs 2000 -e 10
foldseek easy-search all_scope40_fs_db_25/phold_foldseek_db all_scope40_fs_db_25/phold_foldseek_db rawoutput/mini_vs_mini_25 tmp/ --threads $THREADS -s 9.5 --max-seqs 2000 -e 10
foldseek easy-search all_scope40_prostt5_fs_db/phold_foldseek_db scop_foldseekdb/scop  rawoutput/prostt5_vs_pdb tmp/ --threads $THREADS -s 9.5 --max-seqs 2000 -e 10
foldseek easy-search all_scope40_prostt5_fs_db/phold_foldseek_db all_scope40_prostt5_fs_db/phold_foldseek_db rawoutput/prostt5_vs_prostt5 tmp/ --threads $THREADS -s 9.5 --max-seqs 2000 -e 10
foldseek easy-search all_scope40_prostt5_fs_db_25/phold_foldseek_db scop_foldseekdb/scop  rawoutput/prostt5_vs_pdb_25 tmp/ --threads $THREADS -s 9.5 --max-seqs 2000 -e 10
foldseek easy-search all_scope40_prostt5_fs_db_25/phold_foldseek_db all_scope40_prostt5_fs_db_25/phold_foldseek_db rawoutput/prostt5_vs_prostt5_25 tmp/ --threads $THREADS -s 9.5 --max-seqs 2000 -e 10
foldseek easy-search scop_foldseekdb/scop  scop_foldseekdb/scop  rawoutput/foldseekaln tmp/ --threads $THREADS -s 9.5 --max-seqs 2000 -e 10
foldseek easy-search scop_foldseekdb/scop  scop_foldseekdb/scop rawoutput/foldseekaln_nocalpha tmp/ --threads $THREADS -s 9.5 --max-seqs 2000 -e 10 --sort-by-structure-bits 0


# calculate results

mkdir -p rocx

./bench.awk scop_lookup.fix.tsv <(cat rawoutput/mini_vs_pdb) > rocx/mini_vs_pdb.rocx
./bench.awk scop_lookup.fix.tsv <(cat rawoutput/mini_vs_mini) > rocx/mini_vs_mini.rocx
./bench.awk scop_lookup.fix.tsv <(cat rawoutput/mini_vs_pdb_25) > rocx/mini_vs_pdb_25.rocx
./bench.awk scop_lookup.fix.tsv <(cat rawoutput/mini_vs_mini_25) > rocx/mini_vs_mini_25.rocx
./bench.awk scop_lookup.fix.tsv <(cat rawoutput/prostt5_vs_pdb) > rocx/prostt5_vs_pdb.rocx
./bench.awk scop_lookup.fix.tsv <(cat rawoutput/prostt5_vs_prostt5) > rocx/prostt5_vs_prostt5.rocx
./bench.awk scop_lookup.fix.tsv <(cat rawoutput/prostt5_vs_pdb_25) > rocx/prostt5_vs_pdb_25.rocx
./bench.awk scop_lookup.fix.tsv <(cat rawoutput/prostt5_vs_prostt5_25) > rocx/prostt5_vs_prostt5_25.rocx
./bench.awk scop_lookup.fix.tsv <(cat rawoutput/foldseekaln) > rocx/foldseekaln.rocx
./bench.awk scop_lookup.fix.tsv <(cat rawoutput/foldseekaln_nocalpha) > rocx/foldseekaln_nocalpha.rocx

## calculate auc
awk '{ famsum+=$3; supfamsum+=$4; foldsum+=$5}END{print famsum/NR,supfamsum/NR,foldsum/NR}' rocx/mini_vs_pdb.rocx
awk '{ famsum+=$3; supfamsum+=$4; foldsum+=$5}END{print famsum/NR,supfamsum/NR,foldsum/NR}' rocx/mini_vs_mini.rocx
awk '{ famsum+=$3; supfamsum+=$4; foldsum+=$5}END{print famsum/NR,supfamsum/NR,foldsum/NR}' rocx/mini_vs_pdb_25.rocx
awk '{ famsum+=$3; supfamsum+=$4; foldsum+=$5}END{print famsum/NR,supfamsum/NR,foldsum/NR}' rocx/mini_vs_mini_25.rocx
awk '{ famsum+=$3; supfamsum+=$4; foldsum+=$5}END{print famsum/NR,supfamsum/NR,foldsum/NR}' rocx/prostt5_vs_pdb.rocx
awk '{ famsum+=$3; supfamsum+=$4; foldsum+=$5}END{print famsum/NR,supfamsum/NR,foldsum/NR}' rocx/prostt5_vs_prostt5.rocx
awk '{ famsum+=$3; supfamsum+=$4; foldsum+=$5}END{print famsum/NR,supfamsum/NR,foldsum/NR}' rocx/prostt5_vs_pdb_25.rocx
awk '{ famsum+=$3; supfamsum+=$4; foldsum+=$5}END{print famsum/NR,supfamsum/NR,foldsum/NR}' rocx/prostt5_vs_prostt5_25.rocx
awk '{ famsum+=$3; supfamsum+=$4; foldsum+=$5}END{print famsum/NR,supfamsum/NR,foldsum/NR}' rocx/foldseekaln.rocx
awk '{ famsum+=$3; supfamsum+=$4; foldsum+=$5}END{print famsum/NR,supfamsum/NR,foldsum/NR}' rocx/foldseekaln_nocalpha.rocx
```



# Deprecated


1. Download scope40 

We downloaded the SCOPe40 structures (available at https://wwwuser.gwdg.de/~compbiol/foldseek/scop40pdb.tar.gz).

The SCOPe benchmark set consists of single domains with an average length of 174 residues. In our benchmark, we compare the domains all-versus-all. Per domain, we measured the fraction of detected TPs up to the first FP. For family-level, superfamily-level and fold-level recognition, TPs were defined as same family, same superfamily and not same family and same fold and not same superfamily, respectively. Hits from different folds are FPs.

```
wget https://wwwuser.gwdg.de/~compbiol/foldseek/scop40pdb.tar.gz
tar -xzf scop40pdb.tar.gz
mv pdb scop-pdb
```

* To get AA FASTA files for each PDB file

```
python extract_fasta_pdb.py -i scop-pdb -o scope40_fasta -f
cat scope40_fasta/* > all_scope40.fasta
```

2. Run benchmarks

* Get 3Dis

```bash
distill_prostt5 infer -i all_scope40.fasta -o all_scope_infer -m ../checkpoint-184000/
# for the larger model
# distill_prostt5 infer -i all_scope40.fasta -o all_scope_infer -m ../checkpoint-215000/ --num_heads 10 --num_layers 10 --hidden_size 560
conda activate pholdENV
phold proteins-predict -i all_scope40.fasta -d ../../phold_db -t 1 -o all_scope_phold_proteins_predict
phold createdb --fasta_aa all_scope40.fasta --fasta_3di all_scope_infer/output_3di.fasta -o all_scope40_fs_db -f
phold createdb --fasta_aa all_scope40.fasta --fasta_3di all_scope_phold_proteins_predict/phold_3di.fasta -o all_scope40_prostt5_fs_db
```
* Run foldseek (v10 via conda)
* https://github.com/steineggerlab/foldseek-analysis/blob/main/scopbenchmark/scripts/runFoldseek.sh

```
mkdir -p rawoutput
THREADS=32
foldseek easy-search all_scope40_fs_db/phold_foldseek_db scop-pdb/ rawoutput/mini_vs_pdb tmp/ --threads $THREADS -s 9.5 --max-seqs 2000 -e 10
foldseek easy-search all_scope40_fs_db/phold_foldseek_db all_scope40_fs_db/phold_foldseek_db rawoutput/mini_vs_mini tmp/ --threads $THREADS -s 9.5 --max-seqs 2000 -e 10
foldseek easy-search all_scope40_prostt5_fs_db/phold_foldseek_db scop-pdb/ rawoutput/prostt5_vs_pdb tmp/ --threads $THREADS -s 9.5 --max-seqs 2000 -e 10
foldseek easy-search all_scope40_prostt5_fs_db/phold_foldseek_db all_scope40_prostt5_fs_db/phold_foldseek_db rawoutput/prostt5_vs_prostt5 tmp/ --threads $THREADS -s 9.5 --max-seqs 2000 -e 10
foldseek easy-search scop-pdb scop-pdb/ rawoutput/foldseekaln tmp/ --threads $THREADS -s 9.5 --max-seqs 2000 -e 10
foldseek easy-search scop-pdb scop-pdb/ rawoutput/foldseekaln_nocalpha tmp/ --threads $THREADS -s 9.5 --max-seqs 2000 -e 10 --sort-by-structure-bits 0
```


# get tjhe analysis code

* https://github.com/steineggerlab/foldseek-analysis/blob/main/scopbenchmark/scripts/bench.noselfhit.awk
* `scop_lookup.fix.tsv` from https://github.com/steineggerlab/foldseek-analysis/blob/main/scopbenchmark/data/scop_lookup.fix.tsv
```

mkdir -p rocx

./bench.awk scop_lookup.fix.tsv <(cat rawoutput/mini_vs_pdb) > rocx/mini_vs_pdb.rocx
./bench.awk scop_lookup.fix.tsv <(cat rawoutput/mini_vs_mini) > rocx/mini_vs_mini.rocx
./bench.awk scop_lookup.fix.tsv <(cat rawoutput/prostt5_vs_pdb) > rocx/prostt5_vs_pdb.rocx
./bench.awk scop_lookup.fix.tsv <(cat rawoutput/prostt5_vs_prostt5) > rocx/prostt5_vs_prostt5.rocx
./bench.awk scop_lookup.fix.tsv <(cat rawoutput/foldseekaln) > rocx/foldseekaln.rocx
./bench.awk scop_lookup.fix.tsv <(cat rawoutput/foldseekaln_nocalpha) > rocx/foldseekaln_nocalpha.rocx
```

```
## calculate auc
awk '{ famsum+=$3; supfamsum+=$4; foldsum+=$5}END{print famsum/NR,supfamsum/NR,foldsum/NR}' rocx/mini_vs_pdb.rocx
awk '{ famsum+=$3; supfamsum+=$4; foldsum+=$5}END{print famsum/NR,supfamsum/NR,foldsum/NR}' rocx/mini_vs_mini.rocx
awk '{ famsum+=$3; supfamsum+=$4; foldsum+=$5}END{print famsum/NR,supfamsum/NR,foldsum/NR}' rocx/prostt5_vs_pdb.rocx
awk '{ famsum+=$3; supfamsum+=$4; foldsum+=$5}END{print famsum/NR,supfamsum/NR,foldsum/NR}' rocx/prostt5_vs_prostt5.rocx
awk '{ famsum+=$3; supfamsum+=$4; foldsum+=$5}END{print famsum/NR,supfamsum/NR,foldsum/NR}' rocx/foldseekaln.rocx
awk '{ famsum+=$3; supfamsum+=$4; foldsum+=$5}END{print famsum/NR,supfamsum/NR,foldsum/NR}' rocx/foldseekaln_nocalpha.rocx
```


