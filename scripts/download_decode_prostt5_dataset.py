from tqdm import tqdm
from datasets import load_dataset
from transformers import T5Tokenizer
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from argparse import RawTextHelpFormatter
from Bio import SeqIO
import sys
import os
import argparse


"""
overall there are 17070828 sequences

* train_aa_17000000_17100000.fasta will include 17000001 up to 17070828 and the same with 3di

# to run the main stuff in parallel - chunk in sequence batches of 1000000

seq 0 100000 17000000 | parallel -j 20 'start={}; end=$((start + 100000)); python download_decode_prostt5_dataset.py -p train -o train_3di -e 3di --start $start --end $end'
seq 0 100000 17000000 | parallel -j 20 'start={}; end=$((start + 100000)); python download_decode_prostt5_dataset.py -p train -o train_aa -e aa --start $start --end $end'
"""


def get_input():
    """gets input for decoding prostt5 dataset
    :return: args
    """
    parser = argparse.ArgumentParser(
        description="pharokka_plotter.py: pharokka plotting function",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "-p",
        "--partition",
        action="store",
        required=True,
        help="Dataset partition - must be 'train', 'test' or 'valid'",
    )
    parser.add_argument(
        "-o", "--outdir", action="store", default="", help="output directory."
    )
    parser.add_argument(
        "-e",
        "--enc_type",
        action="store",
        required=True,
        help='Encoding type - must be aa or 3di ',
    )
    parser.add_argument(
        "--start",
        action="store",
        default=0,
        type=int,
        help="Starting sequence - 0 indexed. Sequences 1-100000 would mean using --start 0 --end 100000",
    )
    parser.add_argument(
        "--end",
        action="store",
        default=0,
        type=int,
        help="End sequence - indexed python style.",
    )

    args = parser.parse_args()
    return args


def decode_dataset(dataset_partition, outdir, fasta_file, input_name, start=0, end=None):
    """
    dataset_partition = [dataset['train'], dataset['test'], dataset['valid']]
    fasta_file = name of output fasta file
    input_name = ["input_id_x", "input_id_y"] - x is 3Di, y is AA
    start and end are the sequence indices to save
    """

    # Set end to the length of dataset_partition if None
    end = end if end is not None else len(dataset_partition)
    if end < start:
        print('end is less than start. exiting')
        sys.exit()

    os.makedirs(outdir, exist_ok=True)
    fasta_output = os.path.join(outdir, fasta_file)

    seq_records = []
    for i, tokenized_seq in tqdm(enumerate(dataset_partition[start:end][input_name], start=start), total=end-start, desc="Decoding Sequences"):
        i += 1
        try:
            
            if tokenized_seq is None:
                print(f"Warning: Missing {input_name} for sample {i}")
            
            original_seq = tokenizer.decode(tokenized_seq, skip_special_tokens=True).upper()   # store both 3Di and AA as upper case

            # Remove whitespace 
            original_seq_clean = original_seq.replace(" ", "")

            seq_record = SeqRecord(Seq(original_seq_clean), id=f"seq_{i}", description="")
            seq_records.append(seq_record)

        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue

    # Save sequences to FASTA file
    with open(fasta_output, "w") as fasta_file:
        SeqIO.write(seq_records, fasta_file, "fasta")

if __name__ == "__main__":
    args = get_input()


    # Load dataset
    dataset_name = "Rostlab/ProstT5Dataset"
    cache_dir = "ProstT5Dataset_cache"  
    os.makedirs(cache_dir, exist_ok=True)
    dataset = load_dataset(dataset_name, cache_dir=cache_dir)

    prost_model_name = "Rostlab/ProstT5"
    tokenizer = T5Tokenizer.from_pretrained(prost_model_name, do_lower_case=False)

    # # these are the 3Di
    # tokenized_seq = dataset['train'][0]['input_id_x']  
    # # these are the AA
    # tokenized_seq = dataset['train'][0]['input_id_y']  

    # tokenized_seq = dataset['train'][0]['input_id_x']  
    # original_seq_threedi = tokenizer.decode(tokenized_seq, skip_special_tokens=True)

    dataset_partition = dataset[args.partition]

    if args.enc_type == "aa":
        enc_type = "input_id_y"
    elif args.enc_type == "3di":
        enc_type = "input_id_x"

    fasta_file_name = f"{args.partition}_{args.enc_type}_{args.start}_{args.end}.fasta"

    

    decode_dataset(dataset_partition, args.outdir, fasta_file_name, enc_type, start=args.start, end=args.end)

