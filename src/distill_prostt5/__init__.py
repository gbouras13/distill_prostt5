#!/usr/bin/env python3

import click
import os
import csv
import math
import time
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
from collections import defaultdict

seed = 30
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
set_seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from distill_prostt5.classes.MPROSTT5_bert import MPROSTT5, CustomTokenizer
from distill_prostt5.classes.datasets import ProteinDataset, PrecomputedProteinDataset, ProteinDatasetNoLogits, ProteinDatasetPlddt, PrecomputedProteinDatasetPlddt
from distill_prostt5.utils.inference import write_predictions, toCPU, write_probs, write_plddt
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
@click.option(
    "--use_focal",
    help="Use focal loss",
    is_flag=True,
)
@click.option(
    "--gamma",
    help="Focal loss gamma - controls how much you down weight easy samples. Gamma = 0 is cross entropy loss ",
    type=float,
    default=2.0,
)
@click.option(
    "--no_reweight",
    help="No rewighting classes for focal loss",
    is_flag=True,
)
@click.option(
    "--step_down",
    help="Changes single layer projection (hidden_size, 20) to a 2-layer step down with SWIglu activation and intermediate dimension hidden_size // step_down_ratio ",
    is_flag=True,
)
@click.option(
    "--step_down_ratio",
    help="Controls the intermediate dimension in the 2-layer step down intermediate dimension hidden_size // step_down_ratio  ",
    type=int,
    default=4,
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
    use_focal,
    gamma,
    no_reweight,
    step_down,
    step_down_ratio,
    **kwargs,
):
    """Trains distilled Mini ProstT5 model"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

   

    # get training dataset
    train_set = PrecomputedProteinDataset(train_path)  
    eval_set = PrecomputedProteinDataset(eval_path)  

    # Initialize Mini ProstT5 Model
    model = MPROSTT5(hidden_size=hidden_size, 
                     intermediate_size=intermediate_size,  
                     num_layers=num_layers, 
                     num_heads=num_heads, 
                     alpha=alpha, activation=activation, 
                     no_logits=no_logits,
                     use_focal=use_focal,
                     gamma=gamma,
                     no_reweight=no_reweight,
                     step_down=step_down,
                     step_down_ratio=step_down_ratio).to('cpu')
    
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
            no_logits=no_logits,
            use_focal=use_focal,
            gamma=gamma,
            no_reweight=no_reweight,
            step_down=step_down,
            step_down_ratio=step_down_ratio
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
@click.option(
    "--step_down",
    help="Changes single layer projection (hidden_size, 20) to a 2-layer step down with SWIglu activation and intermediate dimension hidden_size // step_down_ratio ",
    is_flag=True,
)
@click.option(
    "--step_down_ratio",
    help="Controls the intermediate dimension in the 2-layer step down intermediate dimension hidden_size // step_down_ratio  ",
    type=int,
    default=4,
)
@click.option(
    "--plddt_head",
    help="Infer with plddt head",
    is_flag=True,
)
@click.option(
    "--batch_size",
    help="Controls inference batchsize  ",
    type=int,
    default=5,
)
@click.option(
    "--max_residues",
    help="Max residues per batch",
    type=int,
    default=50000,
)
@click.option(
    "--half",
    help="Use half precision",
    is_flag=True,
)
@click.option(
    "--fast",
    help="try fast inference",
    is_flag=True,
)
@click.option(
    "--autobatch",
    help="Autobatch",
    is_flag=True,
)
@click.option(
    "--step",
    help="step size for autobatch",
    type=int,
    default=25,
)
@click.option(
    "--max_batch",
    help="max batch size for autobatch",
    type=int,
    default=750,
)
@click.option(
    "--sample_seqs",
    help="number of seqs to subsample for autobatch",
    type=int,
    default=5000,
)
@click.option(
    "--chunk_len",
    help="chunk length",
    type=int,
    default=50000,
)
@click.option(
    "--threads",
    help="number of threads (only for cpu mode)",
    type=int,
    default=1,
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
    step_down,
    step_down_ratio,
    plddt_head,
    batch_size,
    max_residues,
    half,
    fast,
    autobatch,
    step,
    max_batch,
    sample_seqs,
    chunk_len,
    threads,
    **kwargs,
):
    """Infers 3Di from input AA FASTA"""

    def chunk_sequence(seq, max_len):
        """
        Yield (start, subseq) splitting seq into `max_len` nearly equal chunks.
        No overlap. Preserves order.
        """
        L = len(seq)
        n_chunks = math.ceil(L / max_len)
        base = L // n_chunks
        remainder = L % n_chunks

        start = 0
        for i in range(n_chunks):
            # distribute the extra 1 from remainder to the first `remainder` chunks
            size = base + (1 if i < remainder else 0)
            end = start + size
            yield start, seq[start:end]
            start = end


    # used if cpu
    torch.set_num_threads(threads)

    if cpu:
        device = 'cpu'
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_3di: Path = Path(output_dir) / "phold_3di.fasta"
    output_path_mean: Path = Path(output_dir) / "phold_prostT5_3di_mean_probabilities.csv"
    output_path_plddt: Path = Path(output_dir) / "output_plddt.json"
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

    model = MPROSTT5(hidden_size=hidden_size, 
    intermediate_size=intermediate_size,  
    num_layers=num_layers, num_heads=num_heads, 
    step_down=step_down, 
    step_down_ratio=step_down_ratio,
    plddt_head_flag=plddt_head)

    state_dict = load_file(f"{model_ckpt}/model.safetensors")
    model.load_state_dict(state_dict)
    tokenizer = CustomTokenizer()

    model.to(device)
    # half precision
    if half:
        model.half() 
        model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Mini ProstT5 Total Parameters: {total_params}")

    predictions = {}
    batch_predictions = {}

    fail_ids = []

    #max_residues = 100000 passed as CLI
    max_seq_len = 100000

    if autobatch:

        seqs = []
        for feat in cds_dict["proteins"].values():
            v = feat.qualifiers.get("translation")
            if v and isinstance(v, str):
                seqs.append(v)

        logger.info("Beginning batch size tuning")
        logger.info(f"Using minimum batch size of 1 and maximum batch size of {max_batch}")
        # define the sampling

        def sample_probe_sequences(seqs, n=5000, seed=0):
            rng = random.Random(seed)

            if n >= len(seqs):
                sampled = list(seqs)
            else:
                sampled = rng.sample(seqs, n)

            # sort by sequence length
            sampled.sort(key=len, reverse=True)

            return sampled

        # auto tune

        def autotune_batching_real_data(
            model,
            tokenizer,
            device,
            probe_seqs,
            start_bs=1,
            max_bs=max_batch,
            step=step # step size
            # safety=0.8,
            # max_res_cap=100000,
        ):
            model.eval()
            model.half()

            bs = start_bs
            results = []

            while bs <= max_bs:
                try:
                    
                    # seqs = probe_seqs
                    n_tokens = sum(len(s) for s in probe_seqs)

                    logger.info(f"Running with batch size {bs}")

                    model.eval()

                    total_tokens = 0
                    total_time = 0.0
                    batches = 0

                    # iterate over real sequences in batches
                    for i in range(0, len(probe_seqs), bs):
                        batch_seqs = probe_seqs[i : i + bs]
                        n_tokens = sum(len(s) for s in batch_seqs)
                        total_tokens += n_tokens

                        inputs = tokenizer(
                            batch_seqs,
                            padding=True,
                            return_tensors="pt",
                        )
                        inputs.pop("token_type_ids", None)
                        inputs = {k: v.to(device) for k, v in inputs.items()}

                        # timing
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        with torch.no_grad():
                            _ = model(**inputs)
                        torch.cuda.synchronize()

                        total_time += time.perf_counter() - t0
                        
                        batches += 1

                    time_per_token = total_time / total_tokens


                    token_per_batch = math.floor(total_tokens / batches)

                
                    results.append({
                        "bs": bs,
                        "tokens_per_batch": token_per_batch,
                        "time": total_time,
                        "time_per_token": time_per_token,
                    })

                    logger.info(f"Time elapsed {total_time}")
                    logger.info(f"Tokens per batch {token_per_batch}")

                    bs += step

                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    break

            
            if not results:
                raise RuntimeError("No batch size fits on this GPU")

            best_entry = min(results, key=lambda x: x["time_per_token"])

            best_bs = best_entry["bs"]
            best_residues = best_entry["tokens_per_batch"]
            # best_tpt = best_bs["time_per_token"]

            logger.info(f"best batch size: {best_bs}")
            logger.info(f"best max residues: {best_residues}")

            return best_bs, best_residues

        probe_seqs = sample_probe_sequences(seqs, n=sample_seqs)

        batch_size, max_residues = autotune_batching_real_data(
            model,
            tokenizer,
            device,
            probe_seqs,
        )

        logger.info(f"Using batch size {batch_size}")
        logger.info(f"Using max residues {max_residues}")

    if fast:

        

        # --- build + validate sequences in one pass ---
        for record_id, seq_record_dict in cds_dict.items():
            batch_predictions = {}
            chunk_store = defaultdict(dict)

            seq_items = []
            for k, feat in seq_record_dict.items():
                v = feat.qualifiers.get("translation")
                if v and isinstance(v, str):
                    seq = v.replace("U", "X").replace("Z", "X").replace("O", "X")
                    seq_items.append((k, seq, len(seq)))
                else:
                    logger.info(f"Protein header {k} is corrupt. It will be saved in fails.tsv")
                    fail_ids.append(k)

            # --- keep original order ---
            original_keys = list(seq_record_dict.keys())

            # --- sort once (longest first) ---
            seq_items.sort(key=lambda x: x[2], reverse=True)

            batch = []
            res_batch = 0

            for idx, (pid, seq, slen) in enumerate(
                tqdm(seq_items, desc="Processing Sequences"), 1
            ):

                if slen > chunk_len:
                    for chunk_idx, (start, subseq) in enumerate(chunk_sequence(seq, chunk_len)):
                        chunk_pid = f"{pid}__chunk{chunk_idx}"
                        batch.append((chunk_pid, subseq, len(subseq)))
                        res_batch += len(subseq)
                else:
                    batch.append((pid, seq, slen))
                    res_batch += slen

                if (
                    len(batch) >= batch_size
                    or res_batch >= max_residues
                    or slen > max_seq_len
                    or idx == len(seq_items)
                ):
                    pdb_ids, seqs, seq_lens = zip(*batch)
                    batch.clear()
                    res_batch = 0

                    inputs = tokenizer(
                        seqs,
                        add_special_tokens=True,
                        padding=True,
                        return_tensors="pt",
                    )
                    inputs.pop("token_type_ids", None)
                    inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

                    try:
                        with torch.no_grad():
                            outputs = model(**inputs)
                    except RuntimeError:
                        logger.warning(f"OOM / RuntimeError, ids={pdb_ids}")
                        fail_ids.extend(pdb_ids)
                        continue

                    logits = outputs.logits  # [B, L, C]
                    pred_ids = torch.argmax(logits, dim=-1)

                    store_probs = True
                    if store_probs:
                        probs = torch.softmax(logits, dim=-1).max(dim=-1).values

                    if plddt_head:
                        plddt = outputs.plddt_pred

                    pred_ids = pred_ids.cpu().numpy().astype(np.int8)
                    if store_probs:
                        probs = probs.cpu().numpy()
                    if plddt_head:
                        plddt = plddt.cpu().numpy()

                    for i, pid_out in enumerate(pdb_ids):
                        L = seq_lens[i]
                        pred = pred_ids[i, :L]

                        if store_probs:
                            mean_prob = round(100 * probs[i, :L].mean(), 2)
                            all_prob = probs[i, :L]
                        else:
                            mean_prob = None
                            all_prob = None

                        if "__chunk" in pid_out:
                            base_id, chunk_tag = pid_out.split("__chunk")
                            chunk_idx = int(chunk_tag)


                            if plddt_head:
                                chunk_store[base_id][chunk_idx] = (
                                    pred,
                                    all_prob,
                                    plddt[i, :L] if plddt_head else None,
                                )
                            else:

                                chunk_store[base_id][chunk_idx] = (
                                    pred,
                                    all_prob)

                        else:
                            if plddt_head:
                                batch_predictions[pid_out] = (
                                    pred,
                                    mean_prob,
                                    all_prob,
                                    plddt[i, :L] 
                                )
                            else:
                                batch_predictions[pid_out] = (
                                    pred,
                                    mean_prob,
                                    all_prob
                                )


            # --- recombine chunked sequences ---
            for pid, chunks in chunk_store.items():
                preds = []
                probs_all = []
                plddts = []

                for idx in sorted(chunks):

                    if plddt_head:

                        pred, prob, plddt = chunks[idx]
                    else:
                        pred, prob = chunks[idx]


                    preds.append(pred)
                    if prob is not None:
                        probs_all.append(prob)
                    if plddt_head:
                        plddts.append(plddt)

                if plddt_head:
                    plddt_full = np.concatenate(plddts)

                pred_full = np.concatenate(preds)

                if probs_all:
                    probs_full = np.concatenate(probs_all)
                    mean_prob = round(100 * probs_full.mean(), 2)
                else:
                    probs_full = None
                    mean_prob = None

                if plddt_head:
                    plddt_full = np.concatenate(plddts)
                    batch_predictions[pid] = (
                        pred_full,
                        mean_prob,
                        probs_full,
                        plddt_full,
                    )
                else:
                    batch_predictions[pid] = (
                        pred_full,
                        mean_prob,
                        probs_full,
                    )

            # --- reorder to match original FASTA ---
            predictions[record_id] = {}
            for k in original_keys:
                if k in batch_predictions:
                    predictions[record_id][k] = batch_predictions[k]


    else:

        
        for record_id, cds_records in cds_dict.items():
                # instantiate the nested dict
                predictions[record_id] = {}
                seq_record_dict = cds_dict[record_id]
                seq_dict = {}

                # gets the seq_dict with key for id and the translation
                for key, seq_feature in seq_record_dict.items():
                    # get the protein seq for normal
                    seq_dict[key] = seq_feature.qualifiers["translation"]

                # for k, v in seq_dict.items():
                #     if not v or len(v) == 0:
                #         logger.info("Empty value for key:", k)
                #     elif not isinstance(v[0], str):
                #         logger.info("Unexpected type for key:", k, "->", type(v[0]))
                #         fail_ids.append(k)


                # Filter out entries that are empty or malformed
                # for logan - some entries are missing sequences (because of the lack of \n I think)
                # e.g
                # after seqkit
                # grep ERR11457585_70038_2  nonhuman-complete.fa.zst.split/nonhuman-complete.part_016.fa
                # >ERR11457585_70038_2 # 661 # 1926 # 1 # ID=67343_2;partial=00;start_tyDKNDMEKEIGALKKAEDAIYIDSTNMTIEEVVNKVIETIKEKM*

                # zstdcat nonhuman-complete.fa.zst | grep -A10  ERR11457585_70038_2

                # >ERR11457585_70038_2 # 661 # 1926 # 1 # ID=67343_2;partial=00;start_tyDKNDMEKEIGALKKAEDAIYIDSTNMTIEEVVNKVIETIKEKM*
                #    
                # >ERR11457585_71594_3 # 1765 # 3144 # -1 # ID=68839_3;partial=00;start_type=ATG;rbs_motif=GGAG/GAGG;rbs_spacer=5-10bp;gc_cont=0.590
                # MELIRGLKNGMVLQRDMGTNACKITISLRGVQHPQPSLGKLEHLGGERYRLTGIPVGGPYALTLADGTRRLEFADLWVGDVWLLGGQSNMEGWGERGEAELRYDEAPLQKIRAFYLDDHWESARSQLHLPWTNHDTALAEKFLAGRGLTLAQRDCLTLADAGVGPGLFIGQYLCEQSGVPQGLIPCAFGGTCMQDWLPENLTPTSQYRHTLRRFWEIGGNVRGMFWYQGESDLNWLCAAKLHDRMEHMVAAFRKDFDLPELPFVQVQIGRTQGCDDCQLDRIAAWHKIRCLQAEMRFPLFATVSAANATYQDTIHLDTPSQRCIGKAAAMQMCSLLGREELANPVLKHIEIRQTNGHLPTNKTSVVLTYDHVIGELRADGAPSGFSVTLFDEIPYLFPNKLIHHVVLRGNQVEIVTGYSAEQLAHAFVWHGAGPNALCNVHDAEGRALLAMGPMPVCTV*
                # >ERR11457585_71610_4 # 2018 # 2461 # -1 # ID=68854_4;partial=00;start_type=ATG;rbs_motif=GGAGG;rbs_spacer=5-10bp;gc_cont=0.473
                # MDNTAYKNRLNAYISHLEQDEKSRATIAQYRRDIICFFEYLGSAELTKEAVLAYKRQLELKYMPVSVNAKLSALNSFFSFAGRADLELKLLKIQKRAYCPAERELSKEEYFRLVKAAGRRRNRPPFADFTDDLRHWNKGFGAEIYYR*

                # zstdcat nonhuman-complete.fa.zst | grep -A10  ERR11457432_144795_1
                # >ERR11457432_144795_1 # 102 # 575 # -1 # ID=137618_1;partial=00;start_type=ATG;rbs_motif=TAA;rDYIYVNTLKHLIADPVRTSIRWSSSHGDRFRRAGIDWEISQSGFQYAHIQ

                # >ERR11457432_154121_3 # 1749 # 1883 # -1 # ID=146012_3;partial=00;start_type=ATG;rbs_motif=GGAG/GAGG;rbs_spacer=5-10bp;gc_cont=0.467
                # MEEQQDDFFSPENIAELERRIKRLRSGESKLTERDLINPDDEKD*

                valid_seq_dict = {}
                for k, v in seq_dict.items():
                    if v and len(v) > 0 and isinstance(v[0], str):
                        valid_seq_dict[k] = v
                    else:
                        logger.info(f"Protein header {k} is corrupt. It will be saved in fails.tsv")
                        fail_ids.append(k)

                # Sort only the valid ones
                seq_dict = dict(
                    sorted(valid_seq_dict.items(), key=lambda kv: len(kv[1][0]), reverse=True)
                )

                batch = list()
                max_batch = batch_size
                max_residues = 100000
                max_seq_len = 100000

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
                            for id in pdb_ids:
                                fail_ids.append(id)
                            continue

                
                        try:
                            logits = outputs.logits

                            if plddt_head:
                                plddt_pred = outputs.plddt_pred


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

                                if plddt_head:

                                    plddt_slice = toCPU(plddt_pred[batch_idx, 0:s_len])

                                # doubles the length of time taken
                                mean_prob = round(100 * probabilities[batch_idx, 0:s_len].mean().item(), 2)
                                all_prob = probabilities[batch_idx, 0:s_len]


                                if plddt_head:

                                
                                

                                    # predictions[record_id][identifier] = pred
                                    predictions[record_id][identifier] = (
                                        pred,
                                        mean_prob,
                                        all_prob,
                                        plddt_slice
                                    )
                                else:
                                    # predictions[record_id][identifier] = pred
                                    predictions[record_id][identifier] = (
                                        pred,
                                        mean_prob,
                                        all_prob
                                    )

                            

                        except IndexError:
                            logger.warning(
                                "Index error during prediction for {} (L={})".format(
                                    pdb_id, seq_len
                                )
                            )

                            for id in pdb_ids:
                                fail_ids.append(id)
                            
                            continue



    # write list of fails if length > 0
    if len(fail_ids) > 0:
        fail_tsv: Path = Path(output_dir) / "fails.tsv"

        # Convert the list to a list of lists
        data_as_list_of_lists = [[str(item)] for item in fail_ids]

        # Write the list to a TSV file
        with open(fail_tsv, "w", newline="") as file:
            tsv_writer = csv.writer(file, delimiter="\t")
            tsv_writer.writerows(data_as_list_of_lists)

    write_predictions(predictions, output_3di, mask_threshold, plddt_head)
    write_probs(predictions,output_path_mean, plddt_head)
    if plddt_head:
        write_plddt(predictions,output_path_plddt)
    



"""
precompute_plddt command
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
    "--plddt_dir",
    help="Path to directory where plddt jsons (from AFDB) are stored",
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
def precompute_plddt(
    ctx,
    input,
    colabfold,
    plddt_dir,
    precompute_path,
    max_length,
    **kwargs,
):
    """precomputes ProstT5 3Di tokens and pLDDT scores and tokenises input"""

    logger.info("Beginning precomputation of plddt dataset")

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

    train_set = ProteinDatasetPlddt(aa_records, ss_records, plddt_dir, bert_tokenizer, max_length)
    train_set.process_and_save(precompute_path) # dataset.h5
    logger.info(f"Finished processing and randomly cropping sequences for {len(aa_records)} sequences from {input}")


    logger.info(f"Saved to {precompute_path}")




@main_cli.command()
@click.help_option("--help", "-h")
@click.pass_context
@click.option(
    "-p",
    "--train_path",
    help="Path to .h5 file containing training data processed with the precompute-plddt subcommand ",
    type=click.Path(),
    required=True,
)
@click.option(
    "-e",
    "--eval_path",
    help="Path to .h5 file containing evaluation data processed with the precompute-plddt subcommand ",
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
    "--lr_scheduler_type",
    type=click.Choice(LR_SCHEDULER_CHOICES, case_sensitive=False),
    default="linear",
    show_default=True,
    help="Type of learning rate scheduler to use."
)
@click.option(
    "--frozen_path",
    help="Frozen model path",
    type=click.Path()
)
@click.option(
    "--step_down",
    help="Changes single layer projection (hidden_size, 20) to a 2-layer step down with SWIglu activation and intermediate dimension hidden_size // step_down_ratio ",
    is_flag=True,
)
@click.option(
    "--step_down_ratio",
    help="Controls the intermediate dimension in the 2-layer step down intermediate dimension hidden_size // step_down_ratio  ",
    type=int,
    default=4,
)
@click.option(
    "-m",
    "--model_ckpt",
    help="Model checkpoint directory (to restart training from here) ",
    type=click.Path()
)
def train_plddt(
    ctx,
    train_path,
    eval_path,
    output_dir,
    batch_size,
    epochs,
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
    lr_scheduler_type,
    frozen_path,
    step_down,
    step_down_ratio,
    model_ckpt,
    **kwargs,
):
    """Trains distilled Mini ProstT5 model plddt adapter"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # get training dataset
    train_set = PrecomputedProteinDatasetPlddt(train_path)  
    eval_set = PrecomputedProteinDatasetPlddt(eval_path)  

    # Initialize Mini ProstT5 Model
    model = MPROSTT5(hidden_size=hidden_size, 
                     intermediate_size=intermediate_size,  
                     num_layers=num_layers, 
                     num_heads=num_heads, 
                     activation=activation, 
                     step_down=step_down,
                     step_down_ratio=step_down_ratio,
                     plddt_head_flag=True).to('cpu')
    
    # Print number of parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    total_params = sum(p.numel() for p in model_parameters)
    logger.info(f"Mini ProstT5 Total Parameters: {total_params}")
    


    # Load weights 
    if frozen_path:
        state_dict = load_file(f"{frozen_path}/model.safetensors")

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
        lr_scheduler_type=lr_scheduler_type,
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





def main():
    main_cli()


if __name__ == "__main__":
    main()
