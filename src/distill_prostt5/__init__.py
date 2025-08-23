#!/usr/bin/env python3

import click
import os
import torch
import numpy as np
import sys
from Bio import SeqIO
from typing import Any, Dict, List, Optional, Tuple, Union
from safetensors.torch import load_file
from Bio.SeqFeature import FeatureLocation, SeqFeature
import glob
import torch.nn as nn
from pathlib import Path
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import TrainingArguments, Trainer, T5Tokenizer, T5EncoderModel, set_seed
from tqdm import tqdm
#from MPROSTT5_bert import MPROSTT5, CustomTokenizer  # Import the mini ProstT5 model
import h5py
from Bio import SeqIO
from  tqdm import tqdm
from loguru import logger
import random
import shutil

seed = 30
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
set_seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from distill_prostt5.classes.MPROSTT5_bert import MPROSTT5, CustomTokenizer
from distill_prostt5.classes.datasets import ProteinDataset, PrecomputedProteinDataset, ProteinDatasetNoLogits
from distill_prostt5.utils.inference import write_predictions, toCPU, write_probs
from distill_prostt5.utils.initialisation import  init_large_from_base




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
@click.option(
    "--no_logits",
    help="Only tokenize & randomly crop sequences, do not embed and calculate logits.",
    is_flag=True,
)
def precompute(
    ctx,
    input,
    colabfold,
    precompute_path,
    max_length,
    no_logits,
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
    logger.info(f"Loaded {len(aa_records)} AA sequences from {input}")
    logger.info(f"Loaded {len(ss_records)} 3Di sequences from {input}")

    # Check if headers match
    if aa_records.keys() != ss_records.keys():
        logger.warning("Headers in input and colabfold do not match!")
        sys.exit()
    else:
        logger.info("Headers match successfully.")

    if no_logits:
        train_set = ProteinDatasetNoLogits(aa_records, ss_records, bert_tokenizer, max_length)
        train_set.process_and_save(precompute_path) # dataset.h5
        logger.info(f"Finished Tokenising and randomly cropping sequences for {len(aa_records)} sequences from {input}")


    else:
        # Load ProstT5 model - needed for embedding generation
        prost_model_name = "Rostlab/ProstT5"
        prost_tokenizer = T5Tokenizer.from_pretrained(prost_model_name)
        prost_model = T5EncoderModel.from_pretrained(prost_model_name).eval().to(device)

        logger.info(f"Starting Computing ProstT5 embeddings for {len(aa_records)} sequences from {input}")

        # reead in the ProstT5 CNN
        repo_root = Path(__file__).parent.resolve()
        CNN_DIR = repo_root / "cnn/"    
        cnn_checkpoint_path = Path(CNN_DIR) / "cnn_chkpnt" / "model.pt"

        train_set = ProteinDataset(aa_records, ss_records, prost_model, prost_tokenizer, bert_tokenizer, cnn_checkpoint_path, max_length, no_logits)
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
                    group = f[group_name]
                # Iterate over the datasets in the group and save them individually
                    for name, data in group.items():
                    # Create dataset under the 'proteins' group with unique names
                        merged_file[name][current_index] = data
                    current_index += 1

    logger.info(f"Finished merging into {precompute_path}")

"""
LR scheduler options
"""
# Define the allowed learning rate scheduler options
LR_SCHEDULER_CHOICES = [
    "linear",
    "cosine",
    "cosine_with_restarts",
    "polynomial",
    "constant",
    "constant_with_warmup",
    "inverse_sqrt",
    "reduce_lr_on_plateau",
    "cosine_with_min_lr",
    "warmup_stable_decay"
]

    


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
    "--intermediate_size",
    help="Intermediate size size (default to 512)",
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
@click.option(
    "--num_workers",
    help="Number of workers for dataloader (default to 1)",
    type=int,
    default=1,
)
@click.option(
    "--warmup_ratio",
    help="warmup ratio",
    type=float,
    default=0.1,
)
@click.option(
    "--no_logits",
    help="Only tokenize & randomly crop sequences, do not embed and calculate logits.",
    is_flag=True,
)
@click.option(
    "--lr_scheduler_type",
    type=click.Choice(LR_SCHEDULER_CHOICES, case_sensitive=False),
    default="linear",
    show_default=True,
    help="Type of learning rate scheduler to use."
)
@click.option(
    "--initialise",
    help="Use a smaller model to initialise the weights of the larger model",
    is_flag=True,
)
@click.option(
    "--base_path",
    help="Base model path",
    type=click.Path()
)
@click.option(
    "--base_activation",
    help="base model activation type - choose gelu or swiglu, defaults to swiglu",
    type=str,
    default='swiglu'
)
@click.option(
    "--base_num_layers",
    help="Number of layers (default to 6)",
    type=int,
    default=6
)
@click.option(
    "--base_num_heads",
    help="Number of attention heads (default to 8)",
    type=int,
    default=8,
)
@click.option(
    "--base_hidden_size",
    help="Hidden size (default to 512)",
    type=int,
    default=512,
)
@click.option(
    "--base_intermediate_size",
    help="Base model Intermediate size size (default to 512)",
    type=int,
    default=512,
)
@click.option(
    "--restart",
    help="Restart training from a checkpoint but different training args",
    is_flag=True,
)
@click.option(
    "--restart_path",
    help="Restart model path",
    type=click.Path()
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
    intermediate_size,
    learning_rate,
    warmup_ratio,
    save_steps,
    logging_eval_steps,
    num_workers,
    no_logits,
    lr_scheduler_type,
    initialise,
    base_path,
    base_activation,
    base_num_layers,
    base_num_heads,
    base_hidden_size,
    base_intermediate_size,
    restart,
    restart_path,
    **kwargs,
):
    """Trains distilled Mini ProstT5 model"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # get training dataset
    train_set = PrecomputedProteinDataset(train_path)  
    eval_set = PrecomputedProteinDataset(eval_path)  

    # Initialize Mini ProstT5 Model
    model = MPROSTT5(hidden_size=hidden_size, intermediate_size=intermediate_size,  num_layers=num_layers, num_heads=num_heads, alpha=alpha, activation=activation, no_logits=no_logits).to('cpu')
    # Print number of trainable parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    total_params = sum(p.numel() for p in model_parameters)
    logger.info(f"Mini ProstT5 Total Trainable Parameters: {total_params}")

    if initialise:

        #base_model_config = MPROSTT5(hidden_size=base_hidden_size, num_layers=base_num_layers, num_heads=base_num_heads, alpha=alpha, activation=base_activation).to(device)
        #base_model = base_model_config.from_pretrained(base_path)
        #base_model = MPROSTT5.from_pretrained(base_path).to(device)

        # base model config (alpha doesn't matter)
        # instantiate model weights on cpu
        base_model = MPROSTT5(
            hidden_size=base_hidden_size,
            intermediate_size=base_intermediate_size,
            num_layers=base_num_layers,
            num_heads=base_num_heads,
            alpha=alpha,
            activation=base_activation,
            no_logits=no_logits
        ).to("cpu")

        # SLoad weights 
        state_dict = load_file(f"{base_path}/model.safetensors")

        # Load into model
        missing, unexpected = base_model.load_state_dict(state_dict, strict=False)

        # (Optional) Print issues
        print("Missing base model keys:", missing)
        print("Unexpected base model keys:", unexpected)
        
        for name, param in model.state_dict().items():
            print(name, param.shape)
            if name == "projection.weight":
                print(param.shape)
        base_model.load_state_dict(state_dict, strict=True)
        base_model = base_model.to("cpu")
        #for name, param in model.state_dict().items():
        #    print(name, param)
        model = init_large_from_base(base_model, model)
        #print(model) 
        #for name, param in base_model.state_dict().items():
        #    print(name, param)
        #for name, param in model.state_dict().items():
        #    print(name, param)
        # put on gpu
        model = model.to(device)

    if restart:

        # SLoad weights 
        state_dict = load_file(f"{restart_path}/model.safetensors")

        # Load into model
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        # (Optional) Print issues
        print("Missing base model keys:", missing)
        print("Unexpected base model keys:", unexpected)
        
        # put on gpu
        model = model.to(device)



    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        save_strategy="steps",
        logging_strategy="steps",
        eval_strategy="steps",
        eval_steps=logging_eval_steps,
        save_steps=save_steps,     
        logging_steps=logging_eval_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        per_device_train_batch_size=batch_size, # batch size
        gradient_accumulation_steps=1, 
        num_train_epochs=epochs,
        dataloader_num_workers=num_workers, 
        dataloader_pin_memory=True,  # Optimizes performance on GPU
        lr_scheduler_type=lr_scheduler_type
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
    

@main_cli.command()
@click.help_option("--help", "-h")
@click.pass_context
@click.option(
    "-i",
    "--input",
    help="Input FASTA",
    type=click.Path(),
    required=True,
)
@click.option(
    "-o",
    "--output_dir",
    help="Output directory",
    type=click.Path(),
    required=True,
)
@click.option(
    "-m",
    "--model_ckpt",
    help="Model checkpoint directory (to predict 3Di using this) ",
    type=click.Path()
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
    "--intermediate_size",
    help="Intermediate size size (default to 512)",
    type=int,
    default=512,
)
@click.option(
    "--mask_threshold",
    help="Mask residues below this confidence threshold - between 0 and 100",
    type=int,
    default=0,
)
@click.option(
            "--cpu",
            is_flag=True,
            help="Use cpus only.",
)
@click.option(
            "--phold",
            is_flag=True,
            help="Phold output format.",
)
def infer(
    ctx,
    input,
    output_dir,
    model_ckpt,
    num_layers,
    num_heads,
    hidden_size,
    intermediate_size,
    cpu,
    mask_threshold,
    phold,
    **kwargs,
):
    """Infers 3Di from input AA FASTA"""

    if cpu:
        device = 'cpu'
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if phold:
        output_3di: Path = Path(output_dir) / "phold_3di.fasta"
        output_path_mean: Path = Path(output_dir) / "phold_prostT5_3di_mean_probabilities.csv"
        output_aa: Path = Path(output_dir) / "phold_aa.fasta"
        print(f"Copying {input} to {output_aa}")
        shutil.copy2(input, output_aa)

    else:
        output_3di: Path = Path(output_dir) / "output_3di.fasta"
        output_path_mean: Path = Path(output_dir) / "3di_mean_probabilities.csv"

    # get training dataset

    # Dictionary to store the records
    cds_dict = {}
    # need a dummmy nested dict
    cds_dict["proteins"] = {}

    with open(input, "rt") as handle:  # handles gzip too
        records = list(SeqIO.parse(handle, "fasta"))
        if not records:
            logger.warning(f"No proteins were found in your input file {input}.")
            logger.error(
                f"Your input file {input} is likely not a amino acid FASTA file. Please check this."
            )
        for record in records:
            prot_id = record.id
            feature_location = FeatureLocation(0, len(record.seq))
            # Seq needs to be saved as the first element in list hence the closed brackets [str(record.seq)]
            seq_feature = SeqFeature(
                feature_location,
                type="CDS",
                qualifiers={
                    "ID": record.id,
                    "description": record.description,
                    "translation": str(record.seq),
                },
            )

            cds_dict["proteins"][prot_id] = seq_feature

    if not cds_dict:
        logger.error(f"Error: no AA protein sequences found in {input} file")

    # only use swiglu

    model = MPROSTT5(hidden_size=hidden_size, num_layers=num_layers,num_heads=num_heads)
    model = MPROSTT5(hidden_size=hidden_size, intermediate_size=intermediate_size,  num_layers=num_layers, num_heads=num_heads).to('cpu')
    state_dict = load_file(f"{model_ckpt}/model.safetensors")
    model.load_state_dict(state_dict)
    tokenizer = CustomTokenizer()

    model.to(device)
    model.eval()

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    total_params = sum(p.numel() for p in model_parameters)
    logger.info(f"Mini ProstT5 Total Parameters: {total_params}")

    predictions = {}
    # taken from Phold, just for ease, definitely dont need to extra nesting level of the dictionary
    for record_id, cds_records in cds_dict.items():
            # instantiate the nested dict
            predictions[record_id] = {}
            seq_record_dict = cds_dict[record_id]
            seq_dict = {}

            # gets the seq_dict with key for id and the translation
            for key, seq_feature in seq_record_dict.items():
                # get the protein seq for normal
                seq_dict[key] = seq_feature.qualifiers["translation"]

            # sort sequences by length to trigger OOM at the beginning
            seq_dict = dict(
                sorted(seq_dict.items(), key=lambda kv: len(kv[1][0]), reverse=True)
            )

            batch = list()
            max_batch = 5
            max_residues = 100000
            max_seq_len = 10000

            for seq_idx, (pdb_id, seq) in tqdm(enumerate(seq_dict.items(), 1), total=len(seq_dict), desc="Processing Sequences"):
                # replace non-standard AAs
                seq = seq.replace("U", "X").replace("Z", "X").replace("O", "X")
                seq_len = len(seq)
                batch.append((pdb_id, seq, seq_len))

                # count residues in current batch and add the last sequence length to
                # avoid that batches with (n_res_batch > max_residues) get processed
                n_res_batch = sum([s_len for _, _, s_len in batch]) + seq_len
                if (
                    len(batch) >= max_batch
                    or n_res_batch >= max_residues
                    or seq_idx == len(seq_dict)
                    or seq_len > max_seq_len
                ):
                    pdb_ids, seqs, seq_lens = zip(*batch)
                    batch = list()

                    inputs = tokenizer.batch_encode_plus(
                        seqs,
                        add_special_tokens=True,
                        padding="longest",
                        return_tensors="pt",
                    ).to(device)
                    inputs = {k: v for k, v in inputs.items() if k != "token_type_ids"}
                    #inputs = {k: v for k, v in inputs.items() if k != "token_type_ids"}
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    try:
                        with torch.no_grad():
                            outputs = model(**inputs)
                    except RuntimeError:
                        logger.warning(f" number of residues in batch {n_res_batch}")
                        logger.warning(f" seq length is {seq_len}")
                        logger.warning(f" ids are {pdb_ids}")
                        logger.warning(
                            "RuntimeError during embedding for {} (L={})".format(
                                pdb_id, seq_len
                            )
                        )
                        continue

            
                    try:
                        logits = outputs.logits
                        #probabilities = torch.nn.functional.softmax(logits, dim=-1)
                        probabilities = toCPU(
                            torch.max(F.softmax(logits, dim=-1), dim=-1, keepdim=True).values
                        )
                        # batch-size x seq_len x embedding_dim
                        # extra token is added at the end of the seq
                        for batch_idx, identifier in enumerate(pdb_ids):
                            s_len = seq_lens[batch_idx]

                             # slice off padding 
                            pred = logits[batch_idx, 0:s_len, :].squeeze()

                            pred = toCPU(
                                torch.argmax(pred, dim=1, keepdim=True)
                            ).astype(np.byte)

                            # doubles the length of time taken
                            mean_prob = round(100 * probabilities[batch_idx, 0:s_len].mean().item(), 2)
                            all_prob = probabilities[batch_idx, 0:s_len]

                            # predictions[record_id][identifier] = pred
                            predictions[record_id][identifier] = (
                                pred,
                                mean_prob,
                                all_prob,
                            )

                        

                    except IndexError:
                        logger.warning(
                            "Index error during prediction for {} (L={})".format(
                                pdb_id, seq_len
                            )
                        )


    write_predictions(predictions, output_3di, mask_threshold)
    write_probs(predictions,output_path_mean)
    




def main():
    main_cli()


if __name__ == "__main__":
    main()
