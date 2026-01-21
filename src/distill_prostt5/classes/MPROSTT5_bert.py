import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import TokenClassifierOutput
import math
from transformers import PreTrainedTokenizer, ModernBertModel, ModernBertConfig
import re
from transformers.utils import ModelOutput
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class ProstT5Output(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None   # token classification logits
    plddt_pred: Optional[torch.FloatTensor] = None  # [B, L] pLDDT values
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


# def cosine_similarity_token_composition(pred_tokens: torch.Tensor, label_tokens: torch.Tensor, vocab_size: int) -> torch.Tensor:
#     """
#     https://github.com/nayoung10/ASSD/blob/e4b9fd9b7c58fa3bb27231f2cf42aa02758f469b/src/modules/metrics.py#L76
#     Computes the cosine similarity between the token composition of predicted and label sequences.

#     Args:
#         pred_tokens (torch.Tensor): Tensor of predicted token indices (batch_size, seq_len).
#         label_tokens (torch.Tensor): Tensor of ground-truth token indices (batch_size, seq_len).
#         vocab_size (int): Total number of tokens in the vocabulary.

#     Returns:
#         torch.Tensor: Cosine similarity scores with shape (batch_size, seq_len, vocab_size).
#     """
#     def get_token_distribution(tokens: torch.Tensor, vocab_size: int):
#         """Computes the normalized token frequency distribution."""
#         batch_size = tokens.shape[0]
#         token_counts = torch.zeros((batch_size, vocab_size), device=tokens.device)
#         for i in range(batch_size):
#             token_counts[i].scatter_add_(0, tokens[i], torch.ones_like(tokens[i], dtype=torch.float))
#         return F.normalize(token_counts, p=2, dim=1)  # L2 normalize

#     # Compute token distributions
#     pred_dist = get_token_distribution(pred_tokens, vocab_size)
#     label_dist = get_token_distribution(label_tokens, vocab_size)

#     # Compute cosine similarity
#     cos_sim = (pred_dist * label_dist).sum(dim=1)  # Cosine similarity per batch
#     return cos_sim

"""
Focal Loss
"""

# def focal_loss(logits, labels, alpha=1.0, gamma=2.0, reduction="mean"):
def focal_loss(logits, labels,  gamma=2.0, reduction="mean", no_reweight=False):
    """
    logits: [N, C] unnormalized scores
    labels: [N] ground-truth labels
    """

    # 3Di dataset frequencies
    freqs = {
        "A": 0.0283, "C": 0.0290, "D": 0.2436, "E": 0.0128,
        "F": 0.0189, "G": 0.0219, "H": 0.0228, "I": 0.0191,
        "K": 0.0168, "L": 0.0630, "M": 0.0068, "N": 0.0229,
        "P": 0.1059, "Q": 0.0404, "R": 0.0248, "S": 0.0583,
        "T": 0.0157, "V": 0.2156, "W": 0.0194, "Y": 0.0140
    }

    vocab = {
            "A": 0, "C": 1, "D": 2, "E": 3, "F": 4, "G": 5, "H": 6, 
            "I": 7, "K": 8, "L": 9, "M": 10, "N": 11, "P": 12, "Q": 13, 
            "R": 14, "S": 15, "T": 16, "V": 17, "W": 18, "Y": 19 
        }

    # Build freq tensor aligned to vocab indices
    # freq_tensor = torch.ones(len(vocab))  # default weight = 1
    # for aa, idx in vocab.items():
    #     if aa in freqs:
    #         freq_tensor[idx] = freqs[aa]
    #     else:
    #         freq_tensor[idx] = 10000.0  # no loss contribution to specials (won't matter anyway)

    # everything is in order anyway, and only 20 output tokens allowed

    # Convert to tensor in class index order
    freq_tensor = torch.tensor(list(freqs.values()))

    # Inverse frequency weighting
    if no_reweight:
        alpha = None
    else:
        alpha = 1.0 / freq_tensor
        alpha = alpha / alpha.sum()  # normalize so sum=1
        
    # log probs
    log_probs = F.log_softmax(logits, dim=-1) 
    probs = torch.exp(log_probs)  

    labels = labels.long()

    log_p = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1) 



    if alpha is not None:
        # alpha should be tensor of shape [num_classes]
        alpha = alpha.to(labels.device)
        at = alpha[labels]  # pick weight per label
    else:
        at = 1.0
    
    # print(freq_tensor)
    # print(at)
    # print(logits)
    # print(pt)
    # print(log_p)
    # print(labels)

    focal_loss = at * -(1 - pt) ** gamma * log_p

    # print(focal_loss)

    if reduction == "mean":
        return focal_loss.mean()
    elif reduction == "sum":
        return focal_loss.sum()
    return focal_loss


"""
Define the tokenizer for Mini ProstT5 - amino acids + 3 special tokens. 
We don't need to tokenize 3Di as they will never be input (this is explicitly AA -> 3Di convertor), so vocab size is only 28
The tokenisation matches ProstT5 just for ease 
"""

class CustomTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab=None, unk_token="<unk>", pad_token="<pad>", eos_token="</s>"):
        
        # Define vocabulary mapping amino acids & special tokens
        self.vocab = {
            "A": 3, "L": 4, "G": 5, "V": 6, "S": 7, "R": 8, "E": 9, "D": 10,
            "T": 11, "I": 12, "P": 13, "K": 14, "F": 15, "Q": 16, "N": 17,
            "Y": 18, "M": 19, "H": 20, "W": 21, "C": 22, "X": 23, "B": 24,
            "O": 25, "U": 26, "Z": 27, pad_token: 0, eos_token: 1, unk_token: 2
        }

        # Reverse vocabulary for decoding
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

        # Set special tokens explicitly
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.eos_token = eos_token


        # Initialize parent class properly
        super().__init__(
            unk_token=unk_token, 
            pad_token=pad_token, 
            eos_token=eos_token,
            vocab =vocab
        )

    def get_vocab(self):
        """ Returns the vocabulary dictionary. """
        return self.vocab 

    def _tokenize(self, text):
        return list(text)  
    
    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab.get(self.unk_token))  
    
    def _convert_id_to_token(self, index):
        # Reverse the vocab to convert id back to token
        reverse_vocab = {v: k for k, v in self.vocab.items()}
        return reverse_vocab.get(index, self.unk_token)

    def save_vocabulary(self, save_directory):
        # Optionally save your vocab to a file
        pass

    def encode(self, text, add_special_tokens=True):
        # Optionally add special tokens like [PAD], [EOS]
        return [self.convert_token_to_id(token) for token in self._tokenize(text)]
    
    def decode(self, token_ids, skip_special_tokens=False):
        # Optionally decode the token IDs back to string
        return ''.join([self.convert_id_to_token(id) for id in token_ids])

"""
Define the Mini ProstT5 - model
11M param modernBert https://huggingface.co/docs/transformers/en/model_doc/modernbert#transformers.ModernBertModel
Can play around with the size
"""

# https://github.com/lucidrains/PaLM-pytorch/blob/7164d13d5a831647edb5838544017f387130f987/palm_pytorch/palm_pytorch.py#L61C1-L64C32

class SwiGLU(nn.Module):
    def forward(self, x, gate):
        return F.silu(gate) * x

"""
Need to modify the MLP to pass the gate to SwiGLU in the forward pass
"""

class ModernBertMLPSwiGLU(nn.Module):
    """
    Applies the SwiGLU at the end of each ModernBERT layer.
    """

    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config
        self.Wi = nn.Linear(config.hidden_size, int(config.intermediate_size) * 2, bias=config.mlp_bias)
        self.act = SwiGLU()
        self.drop = nn.Dropout(config.mlp_dropout)
        self.Wo = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input, gate = self.Wi(hidden_states).chunk(2, dim=-1) 
        return self.Wo(self.drop(self.act(input, gate)))

"""
To add a swiglu activation and a 2 level projection - v0.5.0

"""


class ProjectionSwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, bias=True):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        # Project to two parallel streams
        self.w = nn.Linear(in_features, hidden_features, bias=bias)
        self.v = nn.Linear(in_features, hidden_features, bias=bias)

        # Project back down
        self.proj = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        return self.proj(F.silu(self.v(x)) * self.w(x))


# Example step-down projection
class StepDownProjection(nn.Module):
    def __init__(self, 
                 hidden_size=512, 
                 d_mid=256, 
                 out_dim=20):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, d_mid, bias=False)
        self.swiglu = ProjectionSwiGLU(d_mid)
        self.fc2 = nn.Linear(d_mid, out_dim, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.swiglu(x)
        x = self.fc2(x)
        return x

"""
full model
"""

class MPROSTT5(nn.Module):
    def __init__(
            self,
            pad_token_id=0,
            num_layers=6,
            hidden_size=512,
            intermediate_size=512, # dunno maybe can play with this
            num_heads=8,
            alpha=0.3, # contribution of colabfold Cross Entropy loss 
            activation='swiglu', # gelu or swiglu,
            no_logits=False, # no logits or not
            use_focal=False, # Use Focal loss 
            gamma=2.0, # gamma for focal loss
            no_reweight=False, # doesn't reweight classes for focal loss
            step_down=False, # 2 layer stepdown - implemented in v0.5.0
            step_down_ratio=4, # d_mid = hidden_size // step_down_ratio - make sure it is divisible by 4
            plddt_head_flag=False # plddt head flag

    ):
        super(MPROSTT5, self).__init__()

        self.tokenizer = CustomTokenizer()
        self.alpha = alpha
        self.no_logits = no_logits
        self.use_focal = use_focal
        self.gamma = gamma
        self.no_reweight = no_reweight
        self.plddt_head_flag = plddt_head_flag

        print(f"--use_focal is {use_focal}")
        if use_focal:
            print(f"--gamma is {gamma}")
            print(f"--no_reweight is {no_reweight}")

        # https://huggingface.co/docs/transformers/en/model_doc/modernbert#transformers.ModernBertModel

        self.configuration = ModernBertConfig( # these are mostly defaults other than the size and attention heads
            vocab_size = 28, # 28 now
            hidden_size = hidden_size, 
            intermediate_size = intermediate_size,
            num_hidden_layers = num_layers,
            num_attention_heads = num_heads,
            hidden_activation = 'gelu',
            max_position_embeddings = 8192,
            initializer_range = 0.02,
            initializer_cutoff_factor = 2.0,
            norm_eps = 1e-05,
            norm_bias = False,
            pad_token_id = pad_token_id,
            eos_token_id = 1,
            bos_token_id = 50281,
            cls_token_id = 50281,
            sep_token_id = 50282,
            global_rope_theta = 160000.0,
            attention_bias = False,
            attention_dropout = 0.0,
            global_attn_every_n_layers = 3,
            local_attention = 128,
            local_rope_theta = 10000.0,
            embedding_dropout = 0.0,
            mlp_bias = False,
            mlp_dropout = 0.0,
            decoder_bias = True,
            classifier_dropout = 0.0,
            classifier_bias = False,
            classifier_activation = 'gelu',
            deterministic_flash_attn = False,
            sparse_prediction = False,
            sparse_pred_ignore_index = -100,
            reference_compile = None,

        )

       
        self.model = ModernBertModel(self.configuration)
        
        # Replace activation function

        if activation == 'swiglu':
            for layer in self.model.layers:
                layer.mlp = ModernBertMLPSwiGLU(self.configuration)
        
        # 2 layer step down
        if step_down:
            if hidden_size % step_down_ratio != 0:
                raise ValueError(
                    f"hidden_size={hidden_size} must be divisible by step_down_ratio={step_down_ratio}"
                )
            self.projection = StepDownProjection(hidden_size, d_mid=hidden_size // step_down_ratio, out_dim=20)
        else:
            self.projection = nn.Linear(hidden_size, 20, bias=False)  # Project to ProstT5-CNN output dimension (20 states)

        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

        # train plddt head
        if self.plddt_head_flag:
            self.plddt_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.ReLU(),
                nn.Linear(hidden_size // 4, 1)
            )
            self.mse_loss = nn.MSELoss(reduction='none')

        # freeze for training plddt head

        if self.plddt_head_flag:
            for param in self.model.parameters():
                param.requires_grad = False

            # freeze projection so only pLDDT head trains
            for param in self.projection.parameters():
                param.requires_grad = False


    def forward(self, input_ids=None, labels=None, attention_mask=None, target=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state 
        logits = self.projection(last_hidden_states)  # projection to ProstT5 size # B  x seq_len x embedding dim (20)
        loss = None
        if self.plddt_head_flag:
            plddt_logits = self.plddt_head(last_hidden_states)  # [B, L, 1]
            plddt_pred = torch.sigmoid(plddt_logits).squeeze(-1) * 100.0  # [B, L]
        else:
            plddt_pred = None
        # to train the plddt head


        if self.plddt_head_flag and target is not None:
            # Mask out padded residues
            mask = (labels != -100)  # [B, L]

            # Mask prediction and target
            masked_pred = plddt_pred[mask]
            masked_target = target[mask]  # ensure float

            # Compute loss
            loss = self.mse_loss(masked_pred, masked_target).mean()

        else:

            if target is not None:

                # Create a mask where labels != -100 - -100 is pad in the colabfold labels
                mask = (labels != -100)

                # Apply the mask to logits and target before computing loss - mask will not calc loss for padding
                masked_logits = logits[mask]
                masked_target = target[mask]
                masked_labels = labels[mask]

                # print(labels)
                # print(mask)
                # print(masked_logits)

                # Compute softmax and log-softmax only on the masked values
                output = F.log_softmax(masked_logits, dim=1)

                if self.no_logits is False:

                    target_probs = F.softmax(masked_target, dim=1)

                    # Compute KL loss
                    kl_loss = self.kl_loss(output, target_probs)

                if self.use_focal:
                    f_or_ce_loss = focal_loss(masked_logits, masked_labels, gamma=self.gamma, reduction="mean", no_reweight = self.no_reweight)
                else:
                    # Cross-Entropy Loss
                    f_or_ce_loss = F.cross_entropy(masked_logits, masked_labels, reduction="mean")


                

                # Combined Loss
                # alpha is the amount of colabfold loss here

                if self.no_logits is False:
                
                    #loss = (1-self.alpha)* kl_loss + self.alpha * ce_loss  # Adjust weight as needed
                    loss = (1-self.alpha)* kl_loss + self.alpha * f_or_ce_loss 

                else:
                    #loss = ce_loss
                    loss = f_or_ce_loss


                predicted_classes = torch.argmax(masked_logits, dim=1)  
                # print("pred")
                # print(predicted_classes)
                if self.no_logits is False:
                    target_classes = torch.argmax(masked_target, dim=1)  
                else:
                    target_classes = masked_target


                # print("vanilla")
                # print(target_classes)

                # print("colabfold")
                # print(masked_labels)
                
                accuracy = (predicted_classes == target_classes).float().mean().item() * 100
                print(f"mini vs vanilla ProstT5 Accuracy: {accuracy:.2f}%")

                accuracy = (predicted_classes == masked_labels).float().mean().item() * 100
                print(f"mini vs colabfold Accuracy: {accuracy:.2f}%")

                accuracy = (target_classes == masked_labels).float().mean().item() * 100
                print(f"vanilla vs colabfold Accuracy: {accuracy:.2f}%")


        return ProstT5Output(
            loss=loss,
            logits=logits,         # always return token logits
            plddt_pred=plddt_pred, # only non-None if head is enabled
            hidden_states=last_hidden_states,
        )

    def tokenize_input(self, sequences):
        return self.tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

class MPROSTT5_PSSM(nn.Module):
    def __init__(self, hidden_size=512, intermediate_size=512, num_layers=6, num_heads=8, activation='swiglu'):
        super().__init__()
        self.config = ModernBertConfig(
            vocab_size=28,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            hidden_activation='gelu',
            pad_token_id=0,
        )
        self.model = ModernBertModel(self.config)
        if activation == 'swiglu':
            for layer in self.model.layers:
                layer.mlp = ModernBertMLPSwiGLU(self.config)
        self.projection = nn.Linear(hidden_size, 20, bias=False)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = output.last_hidden_state
        logits = torch.softmax(self.projection(hidden), dim=2)  # shape (B, L, 20)
        if labels is not None:
            pred = logits.flatten(end_dim = 1)
            target = labels.flatten(end_dim = 1)
            
            pred_mask = attention_mask.flatten(end_dim=1)
            target_mask = ~torch.any(target == -100, dim=1)
            
            pred = pred[pred_mask.bool()]
            target = target[target_mask.bool()]
            loss = self.kl_loss(torch.log(pred), target)
        else:
            loss = None

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden
        )

"""
These are the 3Di tokens corresponding to the predicted classes (by the CNN)

ss_mapping = {
        0: "A",
        1: "C",
        2: "D",
        3: "E",
        4: "F",
        5: "G",
        6: "H",
        7: "I",
        8: "K",
        9: "L",
        10: "M",
        11: "N",
        12: "P",
        13: "Q",
        14: "R",
        15: "S",
        16: "T",
        17: "V",
        18: "W",
        19: "Y"}

"""

