
"""
Define the dataset
"""

import torch.nn as nn
import torch
import h5py
from .CNN import CNN
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
from random import randint
import os
import json
import sys

"""
pads plddt to max length and converts to torch tensor
"""

def pad_plddt(scores, max_length=512, pad_value=0.0):
    # convert list to tensor
    tensor = torch.tensor(scores, dtype=torch.float32)

    # pad at the end
    pad_len = max_length - tensor.size(0)
    tensor = torch.cat([tensor, torch.full((pad_len,), pad_value)])

    # add batch dimension -> [1, 512]
    tensor = tensor.unsqueeze(0)

    return tensor

"""
processes the colabfold input for labels - these are the 20 class outputs
"""

def process_amino_acid_sequence(seq: str, max_length=int):
    ss_mapping = {
        0: "A", 1: "C", 2: "D", 3: "E", 4: "F", 5: "G", 6: "H", 7: "I", 8: "K",
        9: "L", 10: "M", 11: "N", 12: "P", 13: "Q", 14: "R", 15: "S", 16: "T", 17: "V",
        18: "W", 19: "Y"
    }
    inverse_mapping = {v: k for k, v in ss_mapping.items()}  # Reverse mapping for encoding
    
    # Truncate if necessary
    seq = seq[:max_length]
    
    # Map sequence to indices
    mapped_seq = [inverse_mapping.get(aa, -100) for aa in seq]
    
    # Pad sequence to length max_length
    mapped_seq += [-100] * (max_length - len(mapped_seq))
    
    # Convert to torch tensor
    return torch.tensor(mapped_seq, dtype=torch.long).unsqueeze(0)


class ProteinDataset(Dataset):
    def __init__(self, aa_records, ss_records, prost_model, prost_tokenizer,bert_tokenizer, cnn_checkpoint, max_length):
        self.aa_records = aa_records
        self.ss_records = ss_records
        self.prost_model = prost_model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.prost_tokenizer = prost_tokenizer
        self.bert_tokenizer = bert_tokenizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cnn_checkpoint = cnn_checkpoint
        self.max_length = max_length
        self.data = []

    def __len__(self):
        return len(self.aa_records)  # Return the number of items in your dataset

    def process_and_save(self, save_path):
        with h5py.File(save_path, "w") as h5f:
            i = 0 
            #for key  in self.aa_records.keys():
            for key in tqdm(self.aa_records.keys(), desc="Processing sequences"):

                # generate tokens for ProstT5 embedding generation
                aa_seq = self.aa_records[key]
                prostt5_prefix = "<AA2fold>"
                aa_seq_pref = prostt5_prefix + " " + " ".join(aa_seq)
                prost_tokens = self.prost_tokenizer(aa_seq_pref, return_tensors="pt", padding='max_length', truncation=True,  max_length=self.max_length+1) # max_length +1 to let us strip off the prostt5 prefix and keep the same size
                prost_input_ids = prost_tokens.input_ids.to(self.device)
                prost_attention_mask = prost_tokens.attention_mask.to(self.device)

                # print("prostT5 tokens")
                # print(prost_input_ids)
                # print(prost_input_ids.shape)

                # generate tokens for mini ProstT5 (bert0)

                bert_tokens = self.bert_tokenizer(aa_seq, return_tensors="pt", padding='max_length', truncation=True,  max_length=self.max_length)
                bert_input_ids = bert_tokens.input_ids.to(self.device)
                bert_attention_mask = bert_tokens.attention_mask.to(self.device)


                # labels for the real colabfold predictions

                ss_seq = self.ss_records[key]
                colabfold_labels = process_amino_acid_sequence(ss_seq, self.max_length)

                # print("bert tokens")
                # print(bert_input_ids)
                # print(bert_input_ids.shape)

                # to generate the ProstT5 logits

                with torch.no_grad():
                    # follows translate.py/phold
                    residue_embedding = self.prost_model.encoder(prost_input_ids, attention_mask=prost_attention_mask).last_hidden_state
                    residue_embedding = ( # mask out padded elements in the attention output (can be non-zero) for further processing/prediction
                        residue_embedding
                        *prost_attention_mask.unsqueeze(dim=-1)
                    )
                    residue_embedding = residue_embedding[:, 1:] # strip off the AA2fold token
                    predictor = CNN().to(self.device)
                    state = torch.load(self.cnn_checkpoint, map_location=self.device)
                    predictor.load_state_dict(state["state_dict"])
                    prediction = predictor(residue_embedding.to(self.device))
                    logits = prediction.transpose(1, 2) 

                # Save tensors to HDF5
                grp = h5f.create_group(str(i))
                grp.create_dataset("input_ids", data=bert_input_ids.cpu().numpy())
                grp.create_dataset("labels", data=colabfold_labels.cpu().numpy())
                grp.create_dataset("attention_mask", data=bert_attention_mask.cpu().numpy())
                grp.create_dataset("target", data=logits.cpu().numpy())
                i += 1

        print(f"Dataset saved to {save_path}")


"""
For no Logits
"""

class ProteinDatasetNoLogits(Dataset):
    def __init__(self, aa_records, ss_records, bert_tokenizer,  max_length):
        self.aa_records = aa_records
        self.ss_records = ss_records
        self.bert_tokenizer = bert_tokenizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = max_length
        self.data = []

    def __len__(self):
        return len(self.aa_records)  # Return the number of items in your dataset

    def process_and_save(self, save_path):
        with h5py.File(save_path, "w") as h5f:
            index = 0 
            total_groups = len(self.aa_records)
            h5f.create_dataset('input_ids', (total_groups,), dtype=h5py.special_dtype(vlen=np.int32))
            h5f.create_dataset('labels', (total_groups,), dtype=h5py.special_dtype(vlen=np.int32))
            h5f.create_dataset('attention_mask', (total_groups,), dtype=h5py.special_dtype(vlen=np.int32))
            h5f.create_dataset('target', (total_groups,), dtype=h5py.special_dtype(vlen=np.int32))

            #for key  in self.aa_records.keys():
            i = 0
            for key in tqdm(self.aa_records.keys(), desc="Processing sequences"):


                # generate tokens for ProstT5 embedding generation
                aa_seq = self.aa_records[key]
                ss_seq = self.ss_records[key]

                # Apply random cropping
                if len(aa_seq) > self.max_length:
                    start = randint(0, len(aa_seq) - self.max_length)
                    aa_seq = aa_seq[start:start + self.max_length]
                    ss_seq = ss_seq[start:start + self.max_length]

                bert_tokens = self.bert_tokenizer(aa_seq, return_tensors="pt", 
                                                    padding='max_length', truncation=True,  
                                                    max_length=self.max_length)
                bert_input_ids = bert_tokens.input_ids.to(self.device)
                bert_attention_mask = bert_tokens.attention_mask.to(self.device)

                # labels for the real colabfold predictions

                colabfold_labels = process_amino_acid_sequence(ss_seq, self.max_length)

                # print("bert tokens")
                # print(bert_input_ids)
                # print(bert_input_ids.shape)

                # to generate the ProstT5 logits

                # Save tensors to HDF5
                h5f["input_ids"][index] = bert_input_ids.cpu().numpy()
                h5f["labels"][index] = colabfold_labels.cpu().numpy()
                h5f["attention_mask"][index] = bert_attention_mask.cpu().numpy()
                h5f["target"][index] = colabfold_labels.cpu().numpy()
                index += 1


        print(f"Dataset saved to {save_path}")   


"""
Define reading dataset once precomputed and merged
Merging changes the 

Don't need --no_logits as regardless the 4 items returned are the same (target just differs in what it is)
"""

class PrecomputedProteinDataset(Dataset):
    def __init__(self, hdf5_path):
        self.hdf5_path = hdf5_path
        self.h5f = h5py.File(self.hdf5_path, "r")

    def __len__(self):
        return self.h5f['input_ids'].shape[0]

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.h5f['input_ids'][idx], dtype=torch.long)
        labels = torch.tensor(self.h5f['labels'][idx], dtype=torch.long)
        attention_mask = torch.tensor(self.h5f['attention_mask'][idx], dtype=torch.long)
        target = torch.tensor( np.array(self.h5f['target'][idx].tolist(), dtype=np.float32), dtype=torch.float)

        return {
            "input_ids": input_ids.squeeze(0),
            "labels": labels.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "target": target.squeeze(0)
        }

"""
plddt
"""


class ProteinDatasetPlddt(Dataset):
    def __init__(self, aa_records, ss_records, plddt_dir, bert_tokenizer,  max_length):
        """
        aa_records: dict of sequence_id -> amino acid sequence
        ss_records: dict of sequence_id -> colabfold labels
        plddt_dir: directory of AFDB pLDDT jsons
        """
        self.aa_records = aa_records
        self.ss_records = ss_records
        self.plddt_dir = plddt_dir
        self.bert_tokenizer = bert_tokenizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = max_length
        self.data = []

    def __len__(self):
        return len(self.aa_records)  # Return the number of items in your dataset

    def process_and_save(self, save_path):
        with h5py.File(save_path, "w") as h5f:
            index = 0 
            total_groups = len(self.aa_records)
            h5f.create_dataset('input_ids', (total_groups,), dtype=h5py.special_dtype(vlen=np.int32))
            h5f.create_dataset('labels', (total_groups,), dtype=h5py.special_dtype(vlen=np.int32))
            h5f.create_dataset('attention_mask', (total_groups,), dtype=h5py.special_dtype(vlen=np.int32))
            h5f.create_dataset('target', (total_groups,), dtype=h5py.special_dtype(vlen=np.int32))

            #for key  in self.aa_records.keys():
            i = 0
            for key in tqdm(self.aa_records.keys(), desc="Processing sequences"):

                # generate tokens for ProstT5 embedding generation
                aa_seq = self.aa_records[key]
                ss_seq = self.ss_records[key]

                # now amino acids start with AF-A0A0B8XXX7-F1-model_v4
                # json is AF-A0A009EC30-F1-confidence_v4.json

                plddt_filename = key.replace("model", "confidence") + ".json"

                json_path = os.path.join(self.plddt_dir, plddt_filename)

                try:
                    with open(json_path) as f:
                        plddt_data = json.load(f)
                except FileNotFoundError:
                    print(f"[ERROR] File not found: {json_path}")
                    plddt_data = None
                    sys.exit()
                except json.JSONDecodeError:
                    print(f"[ERROR] Failed to decode JSON in file: {json_path}")
                    plddt_data = None
                    sys.exit()

                with open(json_path) as f:
                    plddt_data = json.load(f)

                confidence_scores = plddt_data["confidenceScore"]

                # Apply random cropping
                if len(aa_seq) > self.max_length:
                    start = randint(0, len(aa_seq) - self.max_length)
                    aa_seq = aa_seq[start:start + self.max_length]
                    ss_seq = ss_seq[start:start + self.max_length]
                    confidence_scores = confidence_scores[start:start + self.max_length]

                bert_tokens = self.bert_tokenizer(aa_seq, return_tensors="pt", 
                                                    padding='max_length', truncation=True,  
                                                    max_length=self.max_length)
                bert_input_ids = bert_tokens.input_ids.to(self.device)
                bert_attention_mask = bert_tokens.attention_mask.to(self.device)

                # labels for the real colabfold predictions

                colabfold_labels = process_amino_acid_sequence(ss_seq, self.max_length)

                plddt_targets = pad_plddt(confidence_scores, self.max_length)

                # print("bert tokens")
                # print(bert_input_ids)
                # print(bert_input_ids.shape)

                # print(f"[DEBUG] index={index}")
                # print(f"  input_ids shape: {bert_input_ids.shape}")
                # print(f"  labels shape: {colabfold_labels.shape}")
                # print(f"  attention_mask shape: {bert_attention_mask.shape}")
                # print(f"  confidence_scores type: {plddt_targets.shape}")

                # print(plddt_targets)

                # Save tensors to HDF5
                h5f["input_ids"][index] = bert_input_ids.cpu().numpy()
                h5f["labels"][index] = colabfold_labels.cpu().numpy()
                h5f["attention_mask"][index] = bert_attention_mask.cpu().numpy()
                h5f["target"][index] = plddt_targets.cpu().numpy()
                index += 1


        print(f"Dataset saved to {save_path}")   




class PrecomputedProteinDatasetPlddt(Dataset):
    def __init__(self, hdf5_path):
        self.hdf5_path = hdf5_path
        self.h5f = h5py.File(self.hdf5_path, "r")

    def __len__(self):
        return self.h5f['input_ids'].shape[0]

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.h5f['input_ids'][idx], dtype=torch.long)
        labels = torch.tensor(self.h5f['labels'][idx], dtype=torch.long)
        attention_mask = torch.tensor(self.h5f['attention_mask'][idx], dtype=torch.long)
        target = torch.tensor( np.array(self.h5f['target'][idx].tolist(), dtype=np.float32), dtype=torch.float)
        
        return {
            "input_ids": input_ids.squeeze(0),
            "labels": labels.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "target": target.squeeze(0)
        }

"""
distill_prostt5 train-plddt -p test_plddt.h5 -e test_plddt.h5  -o test_plddt_output_modernbert --epochs 12 -b $BATCH_SIZE --logging_eval_steps 100  --num_workers 8 --num_heads $NUM_HEADS --num_layers $NUM_LAYERS --hidden_size $HIDDEN_DIM --intermediate_size $INTERMEDIATE_DIM --save_steps 1000 --frozen_path checkpoint-320784/
"""