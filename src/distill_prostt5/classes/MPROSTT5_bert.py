import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import TokenClassifierOutput
import math
from transformers import PreTrainedTokenizer, ModernBertModel, ModernBertConfig
import re




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
            no_logits=False # no logits or not
    ):
        super(MPROSTT5, self).__init__()

        self.tokenizer = CustomTokenizer()
        self.alpha = alpha
        self.no_logits = no_logits

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
        
        self.projection = nn.Linear(hidden_size, 20, bias=False)  # Project to ProstT5-CNN output dimension (20 states)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, input_ids=None, labels=None, attention_mask=None, target=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state 
        logits = self.projection(last_hidden_states)  # projection to ProstT5 size # B  x seq_len x embedding dim (20)
        loss = None
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

            # Cross-Entropy Loss
            ce_loss = F.cross_entropy(masked_logits, masked_labels, reduction="mean")

            # Combined Loss
            # alpha is the amount of colabfold loss here

            if self.no_logits is False:
            
                loss = (1-self.alpha)* kl_loss + self.alpha * ce_loss  # Adjust weight as needed

            else:
                loss = ce_loss


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

            # some class balance code - didn't make much of a difference but not 100% sure is correct
        
            #  L=−∑logπ(at∣st)Rt
            # where the reward function Rt is the cosine distance
            # logπ(at∣st) is the log probability of token a sampled given the state s at step t. 

            # m = torch.distributions.Categorical(logits=masked_logits)
            # sampled_actions = m.sample()
            # cosim_reward = cosine_similarity_token_composition(sampled_actions, masked_labels, vocab_size=20)
            # print("cosine similarity",cosim_reward.mean())
            # policy_loss = -m.log_prob(sampled_actions) * cosim_reward
            # policy_loss = policy_loss.mean()
            # print("cosine reinforce loss", policy_loss)

            # beta = 1 # just try it out

            # loss = (1-alpha)* kl_loss + alpha * ce_loss + beta * policy_loss  # Adjust weight as needed


            # cosim_reward = cosine_similarity_token_composition(predicted_classes, masked_labels, vocab_size=28)

            # def cosine_distance_loss(cos_sim):
            #     return ((1 - cos_sim) ** 2).mean() 

            # cos_loss = cosine_distance_loss(cosim_reward)
            # # print(cos_loss)

            # beta = 1 # just try it out

            # loss = (1-alpha)* kl_loss + alpha * ce_loss + beta * cos_loss  # Adjust weight as needed

        

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=last_hidden_states
        )

    def tokenize_input(self, sequences):
        return self.tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

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

