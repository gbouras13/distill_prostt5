 # Preparing ProstT5 dataset 

* To get the FASTA files in 100,000 protein chunks (computer has 20 cores)
* They will be written in the format `train_aa_0_100000.fasta` where 0-100000 is the range in the python loop (i.e. record 1-100000)
* Overall there are 17070828 sequences in the training set
* `train_aa_17000000_17100000.fasta` will include 17000001 up to 17070828
* For the test and valid there are 474 each

```bash
seq 0 100000 17000000 | parallel -j 20 'start={}; end=$((start + 100000)); python download_decode_prostt5_dataset.py -p train -o train_3di -e 3di --start $start --end $end'
seq 0 100000 17000000 | parallel -j 20 'start={}; end=$((start + 100000)); python download_decode_prostt5_dataset.py -p train -o train_aa -e aa --start $start --end $end'

python download_decode_prostt5_dataset.py -p test -o test_aa -e aa --start 0 --end 474
python download_decode_prostt5_dataset.py -p test -o test_3di -e 3di --start 0 --end 474
python download_decode_prostt5_dataset.py -p valid -o test_aa -e aa --start 0 --end 474
python download_decode_prostt5_dataset.py -p valid -o valid_aa -e 3di --start 0 --end 474
```

* Then prepare the ProstT5-CNN logits for the train set on Pawsey with array jobs 

```bash

export containerImage=distill_prostt5_0.1.0.sif
export TRANSFORMERS_CACHE="$PWD/cache"
export HF_HOME="$PWD/cache"

start_values=($(seq 0 100000 17200000))

echo "${start_values[@]}"

OUTDIR="precomputed_embeddings"
mkdir -p $OUTDIR

# Get the task ID for this job instance
task_id=${SLURM_ARRAY_TASK_ID}

# Get the input value for this task
st=${start_values[$task_id-1]}
end=${start_values[$task_id]}

# Define a function that performs the task for a single input value
perform_task() {
    local start="$1"
    local end="$2"

    # Add your actual task here

echo "start $start"
echo "end $end"

# need to bind the cnn into the container

singularity exec --rocm --bind /home/gbouras/gbouras/distill_prostt5/distill_prostt5/src/distill_prostt5/cnn/cnn_chkpnt:/opt/miniforge3/lib/python3.12/site-packages/distill_prostt5/cnn/cnn_chkpnt  $containerImage distill_prostt5 precompute -i train_aa/train_aa_${start}_${end}.fasta -c train_3di/train_3di_${start}_${end}.fasta -p $OUTDIR/${start}_${end}.h5 -m 512

}
```
 
* Then combine and reformat - works from v0.2.0 , need to change the format of the hdf5 file
* Change from group per protein to 4 datasets, and arrays in the datasets for easy fast lookup

```bash
OUTDIR="precomputed_embeddings"
mkdir -p $OUTDIR
distill_prostt5 merge -d $OUTDIR -p train.h5
```

* Did this on Adelaide HPC due to superior Lustre file system
* test and valid simpler

```bash
mkdir -p test_precomputed_embeddings
mkdir -p valid_precomputed_embeddings
distill_prostt5 precompute -i test_aa/test_aa_0_474.fasta -c test_3di/test_aa_0_474.fasta -p test_precomputed_embeddings/test.h5 -m 512
distill_prostt5 merge -d test_precomputed_embeddings/test.h5  -p test.h5 
distill_prostt5 precompute -i valid_aa/valid_aa_0_474.fasta -c valid_3di/valid_aa_0_474.fasta -p valid_precomputed_embeddings/valid.h5 -m 512
distill_prostt5 merge -d valid_precomputed_embeddings/valid.h5  -p valid.h5 
```