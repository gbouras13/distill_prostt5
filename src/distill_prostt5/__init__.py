#!/usr/bin/env python3

import click
import os
import torch
import numpy as np
import sys
import glob
import torch.nn as nn
from pathlib import Path
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import TrainingArguments, Trainer, T5Tokenizer, T5EncoderModel
from tqdm import tqdm
#from MPROSTT5_bert import MPROSTT5, CustomTokenizer  # Import the mini ProstT5 model
import h5py
from Bio import SeqIO
from loguru import logger

from distill_prostt5.classes.MPROSTT5_bert import MPROSTT5, CustomTokenizer
from distill_prostt5.classes.datasets import ProteinDataset, PrecomputedProteinDataset


log_fmt = (
    "[<green>{time:YYYY-MM-DD HH:mm:ss}</green>] <level>{level: <8}</level> | "
    "<level>{message}</level>"
)



@click.group()
@click.help_option("--help", "-h")
def main_cli():
    "Model Distillation Scripts for Mini ProstT5 Model"
    pass



"""
precompute command
"""


@main_cli.command()
@click.help_option("--help", "-h")
@click.pass_context
@click.option(
    "-i",
    "--input",
    help="Path to protein amino acid input file in FASTA format",
    type=click.Path(),
    required=True,
)
@click.option(
    "-c",
    "--colabfold",
    help="Path to 3Di colabfold input file in FASTA format",
    type=click.Path(),
    required=True,
)
@click.option(
    "-p",
    "--precompute_path",
    help="Path to output file where you want to save hdf5 embeddings and other data required for the distillation. Use suffix .h5 (for use with merge)",
    type=click.Path(),
    required=True,
)
@click.option(
    "-m",
    "--max_length",
    help="Max length of input (sequences above this length will be truncated to this many characters).",
    type=int,
    default=512,
)
def precompute(
    ctx,
    input,
    colabfold,
    precompute_path,
    max_length,
    **kwargs,
):
    """precomputes ProstT5 embeddings for distillation and tokenises input"""


    logger.info("Beginning precomputation of embeddings")

    # Loading the BERT Tokenizer
    bert_tokenizer = CustomTokenizer()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Parse FASTA files
    aa_records = {record.id: str(record.seq) for record in SeqIO.parse(input, "fasta")}
    ss_records = {record.id: str(record.seq) for record in SeqIO.parse(colabfold, "fasta")}
    logger.info(f"Loaded {len(aa_records)} sequences from {input}")

    # Check if headers match
    if aa_records.keys() != ss_records.keys():
        logger.warning("Headers in input and colabfold do not match!")
        sys.exit()
    else:
        logger.info("Headers match successfully.")
    
    # Load ProstT5 model - needed for embedding generation
    prost_model_name = "Rostlab/ProstT5"
    prost_tokenizer = T5Tokenizer.from_pretrained(prost_model_name)
    prost_model = T5EncoderModel.from_pretrained(prost_model_name).eval().to(device)

    logger.info(f"Starting Computing ProstT5 embeddings for {len(aa_records)} sequences from {input}")

    # reead in the ProstT5 CNN
    repo_root = Path(__file__).parent.resolve()
    CNN_DIR = repo_root / "cnn/"    
    cnn_checkpoint_path = Path(CNN_DIR) / "cnn_chkpnt" / "model.pt"

    train_set = ProteinDataset(aa_records, ss_records, prost_model, prost_tokenizer, bert_tokenizer, cnn_checkpoint_path, max_length)
    train_set.process_and_save(precompute_path) # dataset.h5

    logger.info(f"Finished Computing ProstT5 embeddings for {len(aa_records)} sequences from {input}")
    logger.info(f"Saved to {precompute_path}")


@main_cli.command()
@click.help_option("--help", "-h")
@click.pass_context
@click.option(
    "-d",
    "--directory",
    help="Directory containing hdf5 files created with distill_prostt5 precompute. Suffix MUST be .h5 for all. Will automatically detect and merge all.",
    type=click.Path(),
    required=True,
)
@click.option(
    "-p",
    "--precompute_path",
    help="Path to output file where you want to save combined hdf5 embeddings and other data required for the distillation",
    type=click.Path(),
    required=True,
)
def merge(
    ctx,
    directory,
    precompute_path,
    **kwargs,
):
    """merges precomputes embeddings and tokenised input for distillation"""

    # a format change is required to merge and lookup the data efficiently
    # saving each protein and a group in the hdf5 fdile is very inefficient, and observed it struggles above 2.88 M proteins
    # It is like making 17M files in a filesystem 
    # https://stackoverflow.com/questions/35321093/limit-on-number-of-hdf5-datasets
    # therefore, better to store as 4 datasets (one for input ids, labels, attention mask and target)
    # with 17M entries - like and array much cheaper to look up
    # I should have written precompute like this too, but have already computed the 17M embeddings
    # so will modify PrecomputedProteinDataset instead

    logger.info(f"Finding all .h5 files in {directory} to merge")
    file_paths = glob.glob(os.path.join(directory, '**', '*.h5'), recursive=True)

    logger.info(f"Found {len(file_paths)} .h5 files in {directory}")
    logger.info(f"There are {file_paths}")
    logger.info(f"Starting merging into {precompute_path}")

    total_groups = 0
    for file_path in file_paths:
        with h5py.File(file_path, "r") as f:
            total_groups += len(f.keys())  # Assuming each top-level group represents a dataset entry

    with h5py.File(precompute_path, "w") as merged_file:
        current_index = 0
        merged_file.create_dataset('input_ids', (total_groups,), dtype=h5py.special_dtype(vlen=np.int32))
        merged_file.create_dataset('labels', (total_groups,), dtype=h5py.special_dtype(vlen=np.int32))
        merged_file.create_dataset('attention_mask', (total_groups,), dtype=h5py.special_dtype(vlen=np.int32))
        merged_file.create_dataset('target', (total_groups, 512), dtype=h5py.special_dtype(vlen=np.float32))

    # Iterate over each HDF5 file
        for file_path in file_paths:
            with h5py.File(file_path, "r") as f:
            # Iterate over the groups in the current file
                for group_name in f.keys():
                    print(current_index)
                    group = f[group_name]
                # Iterate over the datasets in the group and save them individually
                    for name, data in group.items():
                    # Create dataset under the 'proteins' group with unique names
                        merged_file[name][current_index] = data
                    current_index += 1

    logger.info(f"Finished merging into {precompute_path}")


    


@main_cli.command()
@click.help_option("--help", "-h")
@click.pass_context
@click.option(
    "-p",
    "--train_path",
    help="Path to .h5 file containing training data processed with the precompute subcommand ",
    type=click.Path(),
    required=True,
)
@click.option(
    "-e",
    "--eval_path",
    help="Path to .h5 file containing evaluation data processed with the precompute subcommand ",
    type=click.Path(),
    required=True,
)
@click.option(
    "-o",
    "--output_dir",
    help="Output directory where checkpoints will be saved ",
    type=click.Path(),
    required=True,
)
@click.option(
    "-m",
    "--model_ckpt",
    help="Model checkpoint directory (to restart training from here) ",
    type=click.Path()
)
@click.option(
    "-b",
    "--batch_size",
    help="Batch size per device - 192 can fit in MI250 GPU memory",
    type=int,
    default=192
)
@click.option(
    "--epochs",
    help="Epochs",
    type=int,
    default=50
)
@click.option(
    "-a",
    "--alpha",
    help="Weighted contribution of Colabfold CE loss to total loss",
    type=float,
    default=0.3
)
@click.option(
    "--activation",
    help="activation type - choose gelu or swiglu, defaults to swiglu",
    type=str,
    default='swiglu'
)
@click.option(
    "--num_layers",
    help="Number of layers (default to 6)",
    type=int,
    default=6
)
@click.option(
    "--num_heads",
    help="Number of attention heads (default to 8)",
    type=int,
    default=8,
)
@click.option(
    "--hidden_size",
    help="Hidden size (default to 512)",
    type=int,
    default=512,
)
@click.option(
    "--learning_rate",
    help="learning rate (default to 3e-4)",
    type=float,
    default=3e-4,
)
@click.option(
    "--save_steps",
    help="Save checkpoint this many steps (default to 1000)",
    type=int,
    default=1000,
)
@click.option(
    "--logging_eval_steps",
    help="Eval and log at this many steps (default to 25)",
    type=int,
    default=25,
)
def train(
    ctx,
    train_path,
    eval_path,
    output_dir,
    model_ckpt,
    batch_size,
    epochs,
    alpha,
    activation,
    num_layers,
    num_heads,
    hidden_size,
    learning_rate,
    save_steps,
    logging_eval_steps,
    **kwargs,
):
    """Trains distilled Mini ProstT5 model"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # get training dataset
    train_set = PrecomputedProteinDataset(train_path)  # dataset.h5
    eval_set = PrecomputedProteinDataset(eval_path)  # dataset.h5

    # Initialize Mini ProstT5 Model
    model = MPROSTT5(hidden_size=hidden_size, num_layers=num_layers,num_heads=num_heads, alpha=alpha, activation=activation).to(device)
    # Print number of trainable parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    total_params = sum(p.numel() for p in model_parameters)
    logger.info(f"Mini ProstT5 Total Trainable Parameters: {total_params}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        save_strategy="steps",
        logging_strategy="steps",
        evaluation_strategy="steps",
        eval_steps=logging_eval_steps,
        save_steps=save_steps,     
        logging_steps=logging_eval_steps,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        per_device_train_batch_size=batch_size, # batch size
        gradient_accumulation_steps=1, 
        num_train_epochs=epochs,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=eval_set
    )

    # Train the model
    if model_ckpt:
        trainer.train(resume_from_checkpoint=model_ckpt)
    else:
        trainer.train()
    




def main():
    main_cli()


if __name__ == "__main__":
    main()
