
"""
functions to initialise a larger model using a smaller one

    # helpful explanation of what this is doing


    Imagine the small model’s weights like a small piece of patterned fabric. Now imagine the larger model needs a bigger piece of fabric.

    You repeat  the smaller fabric in both directions (up/down, left/right) to cover the whole larger fabric.
    You wrap around the edges — like how wallpaper or video game backgrounds repeat.
    So if the small model has a weight matrix that’s 768×768 and the large one needs 1024×1024:

    You fill the 1024×1024 matrix by repeating pieces of the 768×768 one until it’s full.

    Instead of starting the tiled pattern from the top-left corner, you:

    Put the small matrix in the center of the larger one.
    Then fill the edges around it by repeating (tiling) parts of the small one.
    This “centered wraparound tiling” was found to work best in the Phi and ModernBERT papers.

    If the small model has 12 layers, and the large one has 24:

    You just repeat the 12 layers twice to fill up the 24.
    Each of the 24 layers will still have useful, pretrained weights — rather than starting from scratch.

    If the embedding size (hidden dimension) increases (say, from 768 to 1024):

    You repeat the small embedding values across the new dimension.
    So each row in the embedding table is tiled left to right until it fills the bigger space.

    most code taken from 
    https://github.com/AnswerDotAI/ModernBERT/blob/8c57a0f01c12c4953ead53d398a36f81a4ba9e38/src/bert_layers/initialization.py
    and 
    https://github.com/AnswerDotAI/ModernBERT/blob/8c57a0f01c12c4953ead53d398a36f81a4ba9e38/src/bert_layers/model.py#L1502


    need to look at varying intermediate size too
"""


import math
from typing import Optional, Union

import torch
import torch.nn as nn
from enum import Enum

class StrEnum(str, Enum):
    """
    This is equivalent to Python's :class:`enum.StrEnum` since version 3.11.
    We include this here for compatibility with older version of Python.
    """

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"'{str(self)}'"



class TileMode(StrEnum):
    center_weights = "center_weights"
    tile_weights_from_edge = "tile_weights_from_edge"
    tile_weights_from_middle = "tile_weights_from_middle"


def tile_weight(
    pretrained_weights: torch.Tensor,
    new_weights: torch.Tensor,
    mode: Union[str, TileMode] = TileMode.tile_weights_from_middle,
) -> torch.Tensor:
    """
    Tile or center an input tensor to a larger desired size. Works for both 2D and 1D tensors.

    Args:
    pretrained_weights (torch.Tensor): The input tensor to be tiled or centered (1D or 2D).
    new_weights (torch.Tensor): The tensor with the desired size.
    mode (Union[str, TileMode]): 'center_weights', 'tile_weights_from_edge', or 'tile_weights_from_middle'

    Returns:
    torch.Tensor: The resulting tensor of the desired size.
    """
    assert pretrained_weights.dim() in (1, 2), "Input tensor must be 1-dimensional or 2-dimensional"
    if isinstance(mode, str):
        mode = TileMode(mode)

    pretrained_weights = pretrained_weights.clone()

    if pretrained_weights.dim() == 1:
        return _tile_1d(pretrained_weights, new_weights, mode)
    else:
        return _tile_2d(pretrained_weights, new_weights, mode)


def _tile_1d(pretrained_weights: torch.Tensor, new_weights: torch.Tensor, mode: TileMode) -> torch.Tensor:
    assert pretrained_weights.dim() == 1, "Input tensor must be 1-dimensional"
    input_size = pretrained_weights.shape[0]
    new_size = new_weights.shape[0]
    assert new_size >= input_size, "Desired size must be greater than or equal to input size"

    if mode == TileMode.center_weights:
        offset = (new_size - input_size) // 2
        new_weights[offset : offset + input_size] = pretrained_weights
        return new_weights.clone()
    elif mode == TileMode.tile_weights_from_edge:
        repeat_count = (new_size + input_size - 1) // input_size
        tiled_tensor = pretrained_weights.repeat(repeat_count)
        return tiled_tensor[:new_size].clone()
    elif mode == TileMode.tile_weights_from_middle:
        # Calculate offsets to center the original tensor
        offset = (new_size - input_size) // 2

        # Create a new tensor with the desired size
        result = torch.zeros(new_size, dtype=pretrained_weights.dtype, device=pretrained_weights.device)

        # Place the original tensor in the center
        result[offset : offset + input_size] = pretrained_weights

        # Tile the left and right sides
        for i in range(offset):
            result[offset - 1 - i] = pretrained_weights[input_size - 1 - (i % input_size)]
        for i in range(offset + input_size, new_size):
            result[i] = pretrained_weights[(i - offset) % input_size]
        return result.clone()


def _tile_2d(pretrained_weights: torch.Tensor, new_weights: torch.Tensor, mode: TileMode) -> torch.Tensor:
    assert pretrained_weights.dim() == 2, "Input tensor must be 2-dimensional"
    input_height, input_width = pretrained_weights.shape
    new_height, new_width = new_weights.shape
    assert new_height >= input_height, "Desired height must be greater than or equal to input height"
    assert new_width >= input_width, "Desired width must be greater than or equal to input width"

    if mode == TileMode.center_weights:
        height_offset = (new_height - input_height) // 2
        width_offset = (new_width - input_width) // 2
        new_weights[height_offset : height_offset + input_height, width_offset : width_offset + input_width] = pretrained_weights  # fmt: skip
        return new_weights.clone()
    elif mode == TileMode.tile_weights_from_edge:
        repeat_height = (new_height + input_height - 1) // input_height
        repeat_width = (new_width + input_width - 1) // input_width
        tiled_tensor = pretrained_weights.repeat(repeat_height, repeat_width)
        return tiled_tensor[:new_height, :new_width].clone()
    elif mode == TileMode.tile_weights_from_middle:
        # Calculate offsets to center the original tensor
        height_offset = (new_height - input_height) // 2
        width_offset = (new_width - input_width) // 2

        # Create a new tensor with the desired width and input height
        horizontal_tiled = torch.zeros(
            input_height, new_width, dtype=pretrained_weights.dtype, device=pretrained_weights.device
        )

        # Place the original tensor in the center horizontally
        horizontal_tiled[:, width_offset : width_offset + input_width] = pretrained_weights

        # Tile the left and right sides
        for i in range(width_offset):
            horizontal_tiled[:, i] = horizontal_tiled[
                :, width_offset + input_width - 1 - (width_offset - i - 1) % input_width
            ]
        for i in range(width_offset + input_width, new_width):
            horizontal_tiled[:, i] = horizontal_tiled[:, width_offset + (i - width_offset) % input_width]

        # Now tile vertically
        result = torch.zeros(new_height, new_width, dtype=pretrained_weights.dtype, device=pretrained_weights.device)
        result[height_offset : height_offset + input_height, :] = horizontal_tiled

        # Tile top
        for i in range(height_offset):
            row_to_copy = (input_height - 1) - (i % input_height)
            result[height_offset - 1 - i, :] = horizontal_tiled[row_to_copy, :]

        # Tile bottom
        for i in range(height_offset + input_height, new_height):
            row_to_copy = (i - height_offset) % input_height
            result[i, :] = horizontal_tiled[row_to_copy, :]
        return result.clone()


def tile_fused_qkv(
    pretrained_qkv_weight: torch.Tensor,
    new_qkv_weight: torch.Tensor,
    mode: Union[str, TileMode] = TileMode.tile_weights_from_middle,
):
    """
    Tile the weights of a fused pretrained QKV layer to a new, larger QKV dimension.

    Args:
        pretrained_qkv_weight (torch.Tensor): The original fused QKV layer
        new_qkv_weight (torch.Tensor): The new fused QKV layer with larger linear_dim
        mode (Union[str, TileMode]): The tiling mode to use
    Returns:
        torch.Tensor: The new fused QKV layer with tiled weights
    """
    # Split QKV, assume new_q, new_k, new_v are the same shape
    pretrained_q, pretrained_k, pretrained_v = pretrained_qkv_weight.chunk(3, dim=0)
    new_q, new_k, new_v = new_qkv_weight.chunk(3, dim=0)

    # Tile Q, K, V separately
    new_q = tile_weight(pretrained_q, new_q, mode=mode)
    new_k = tile_weight(pretrained_k, new_k, mode=mode)
    new_v = tile_weight(pretrained_v, new_v, mode=mode)

    # Concatenate tiled Q, K, V
    return torch.cat([new_q, new_k, new_v], dim=0)


def tile_fused_glu(
    pretrained_glu_weight: torch.Tensor,
    new_glu_weight: torch.Tensor,
    mode: Union[str, TileMode] = TileMode.tile_weights_from_middle,
):
    """
    Tile the weights of a fused pretrained GLU layer to a new, larger GLU dimension.

    Args:
        pretrained_glu_weight (torch.Tensor): The original fused GLU layer
        new_glu_weight (torch.Tensor): The new fused GLU layer with larger linear_dim
        mode (Union[str, TileMode]): The tiling mode to use
    Returns:
        torch.Tensor: The new fused GLU layer with tiled weights
    """
    # Split GLU, assume new_glu_wi, new_glu_wg are the same shape
    pretrained_glu_wi, pretrained_glu_wg = pretrained_glu_weight.chunk(2, dim=0)
    new_glu_wi, new_glu_wg = new_glu_weight.chunk(2, dim=0)

    # Tile GLU separately
    new_glu_wi = tile_weight(pretrained_glu_wi, new_glu_wi, mode=mode)
    new_glu_wg = tile_weight(pretrained_glu_wg, new_glu_wg, mode=mode)

    # Concatenate tiled GLU
    return torch.cat([new_glu_wi, new_glu_wg], dim=0)


def tile_fused_qkvff(
    pretrained_qkvff_weight: torch.Tensor,
    new_qkvff_weight: torch.Tensor,
    pretrained_attn_size: int,
    pretrained_mlp_size: int,
    new_attn_size: int,
    new_mlp_size: int,
    is_glu: bool = False,
    mode: Union[str, TileMode] = TileMode.tile_weights_from_middle,
):
    """
    Tile the weights of a fused pretrained QKVFF layer to a new, larger QKVFF dimension.

    Args:
        pretrained_qkvff_weight (torch.Tensor): The original fused QKVFF layer
        new_qkvff_weight (torch.Tensor): The new fused QKVFF layer with larger linear_dim
        pretrained_attn_size (int): The attention size of the pretrained fused QKVFF layer
        pretrained_mlp_size (int): The mlp size of the pretrained fused QKVFF layer
        new_attn_size (int): The attention size of the new fused QKVFF layer
        new_mlp_size (int): The mlp size of the new fused QKVFF layer
        is_glu (bool): Whether the QKVFF layer is a GLU layer
        mode (Union[str, TileMode]): The tiling mode to use
    Returns:
        torch.Tensor: The new fused QKVFF layer with tiled weights
    """
    # Split QKVFF
    pretrained_qkv, pretrained_ff = pretrained_qkvff_weight.split([pretrained_attn_size, pretrained_mlp_size], dim=0)
    new_qkv, new_ff = new_qkvff_weight.split([new_attn_size, new_mlp_size], dim=0)

    # Tile QKVFF separately
    new_qkv = tile_fused_qkv(pretrained_qkv, new_qkv, mode=mode)
    if is_glu:
        new_ff = tile_fused_glu(pretrained_ff, new_ff, mode=mode)
    else:
        new_ff = tile_weight(pretrained_ff, new_ff, mode=mode)

    # Concatenate tiled QKVFF
    return torch.cat([new_qkv, new_ff], dim=0)


class TileLinear(StrEnum):
    wqkv = "wqkv"
    glu = "glu"
    wqkvff = "wqkvff"
    default = "default"


def tile_linear(
    pretrained_linear: nn.Linear,
    new_linear: nn.Linear,
    linear_type: Union[str, TileLinear] = TileLinear.default,
    mode: Union[str, TileMode] = TileMode.tile_weights_from_middle,
    pretrained_attn_size: Optional[int] = None,
    pretrained_mlp_size: Optional[int] = None,
    new_attn_size: Optional[int] = None,
    new_mlp_size: Optional[int] = None,
    wqkvff_is_glu: Optional[bool] = None,
    bias_only: Optional[bool] = False,
):
    """
    Tile the weights of a linear layer to a new, larger linear dimension.

    Args:
        pretrained_linear (nn.Linear): The original linear layer
        new_linear (nn.Linear): The new linear layer with larger linear_dim
        linear_type (Union[str, TileLinear]): The type of linear layer to tile
        mode (Union[str, TileMode]): The tiling mode to use
        pretrained_attn_size (int): The attention size of the pretrained linear layer. Only used if linear_type is wqkvff.
        pretrained_mlp_size (int): The mlp size of the pretrained linear layer. Only used if linear_type is wqkvff.
        new_attn_size (int): The attention size of the new linear layer. Only used if linear_type is wqkvff.
        new_mlp_size (int): The mlp size of the new linear layer. Only used if linear_type is wqkvff.
        wqkvff_is_glu (bool): Whether the wqkvff layer is a GLU layer. Only used if linear_type is wqkvff.
        bias_only (bool): Whether to only tile the bias. Only used if tiling weight tied decoder.
    """
    if isinstance(linear_type, str):
        linear_type = TileLinear(linear_type)
    if isinstance(mode, str):
        mode = TileMode(mode)

    with torch.no_grad():
        if linear_type == TileLinear.wqkv:
            if not bias_only:
                new_linear.weight = nn.Parameter(
                    tile_fused_qkv(pretrained_linear.weight, new_linear.weight, mode=mode),
                    requires_grad=new_linear.weight.requires_grad,
                )
            if pretrained_linear.bias is not None:
                new_linear.bias = nn.Parameter(
                    tile_fused_qkv(pretrained_linear.bias, new_linear.bias, mode=mode),
                    requires_grad=new_linear.bias.requires_grad,
                )
        elif linear_type == TileLinear.glu:
            if not bias_only:
                new_linear.weight = nn.Parameter(
                    tile_fused_glu(pretrained_linear.weight, new_linear.weight, mode=mode),
                    requires_grad=new_linear.weight.requires_grad,
                )
            if pretrained_linear.bias is not None:
                new_linear.bias = nn.Parameter(
                    tile_fused_glu(pretrained_linear.bias, new_linear.bias, mode=mode),
                    requires_grad=new_linear.bias.requires_grad,
                )
        elif linear_type == TileLinear.wqkvff:
            if not bias_only:
                new_linear.weight = nn.Parameter(
                    tile_fused_qkvff(
                        pretrained_linear.weight,
                        new_linear.weight,
                        pretrained_attn_size,
                        pretrained_mlp_size,
                        new_attn_size,
                        new_mlp_size,
                        wqkvff_is_glu,
                        mode=mode,
                    ),
                    requires_grad=new_linear.weight.requires_grad,
                )
            if pretrained_linear.bias is not None:
                new_linear.bias = nn.Parameter(
                    tile_fused_qkvff(
                        pretrained_linear.bias,
                        new_linear.bias,
                        pretrained_attn_size,
                        pretrained_mlp_size,
                        new_attn_size,
                        new_mlp_size,
                        wqkvff_is_glu,
                        mode=mode,
                    ),
                    requires_grad=new_linear.bias.requires_grad,
                )
        else:
            if not bias_only:
                new_linear.weight = nn.Parameter(
                    tile_weight(pretrained_linear.weight, new_linear.weight, mode=mode),
                    requires_grad=new_linear.weight.requires_grad,
                )
            if pretrained_linear.bias is not None:
                new_linear.bias = nn.Parameter(
                    tile_weight(pretrained_linear.bias, new_linear.bias, mode=mode),
                    requires_grad=new_linear.bias.requires_grad,
                )


def tile_norm(
    pretrained_norm: Union[nn.LayerNorm, nn.Identity],
    new_norm: Union[nn.LayerNorm, nn.Identity],
    mode: Union[str, TileMode] = TileMode.tile_weights_from_middle,
):
    """
    Tile the weights of a pretrained norm layer to a new, larger layer norm dimension.

    Args:
        pretrained_norm (Union[nn.LayerNorm, nn.Identity]): The original norm layer
        new_norm (Union[nn.LayerNorm, nn.Identity]): The new norm layer with larger layer norm dimension
        mode (Union[str, TileMode]): The Phi-style weight tiling mode to use
    """
    if isinstance(pretrained_norm, nn.Identity):
        return
    if isinstance(mode, str):
        mode = TileMode(mode)

    with torch.no_grad():
        new_norm.weight.data = nn.Parameter(
            tile_weight(pretrained_norm.weight, new_norm.weight, mode=mode),
            requires_grad=new_norm.weight.requires_grad,
        )
        if hasattr(pretrained_norm, "bias") and pretrained_norm.bias is not None:
            new_norm.bias.data = nn.Parameter(
                tile_weight(pretrained_norm.bias, new_norm.bias, mode=mode),
                requires_grad=new_norm.bias.requires_grad,
            )


def tile_embedding(
    pretrained_embedding: nn.Embedding,
    new_embedding: nn.Embedding,
    mode: Union[str, TileMode] = TileMode.tile_weights_from_middle,
) -> nn.Embedding:
    """
    Tile the weights of an embedding layer to a new, larger embedding dimension.

    Args:
    pretrained_embedding (nn.Embedding): The original embedding layer
    new_embedding (nn.Embedding): The new embedding layer with larger embedding_dim
    tile_mode (Union[str, TileMode]): The Phi-style weight tiling mode to use

    Returns:
    nn.Embedding: The new embedding layer with tiled weights
    """
    with torch.no_grad():
        # Ensure vocabulary size remains the same
        if pretrained_embedding.num_embeddings != new_embedding.num_embeddings:
            raise ValueError("Vocabulary size (num_embeddings) must remain constant")

        # Ensure new embedding dimension is larger
        if new_embedding.embedding_dim <= pretrained_embedding.embedding_dim:
            raise ValueError("New embedding_dim must be larger than the old embedding_dim")

        # Tile the weights
        new_embedding.weight.data = nn.Parameter(
            tile_weight(pretrained_embedding.weight, new_embedding.weight, mode=mode),
            requires_grad=new_embedding.weight.requires_grad,
        )

        # Handle padding_idx if it exists
        if pretrained_embedding.padding_idx is not None:
            if new_embedding.padding_idx is None:
                new_embedding.padding_idx = pretrained_embedding.padding_idx
            else:
                assert new_embedding.padding_idx == pretrained_embedding.padding_idx, "padding_idx must remain the same"

def init_large_from_base(
    pretrained_model,
    new_model,
    mode: Union[str, TileMode] = TileMode.tile_weights_from_middle,
):
    """
    Initialize the new model from the pretrained model.

    This method uses Gopher layer scaling and Phi-style weight tiling as selected by `mode`.
    The new model must have the same or more layers and the same or larger dimensions than the pretrained model.

    Args:
        pretrained_model (FlexBertModel): The smaller, pre-trained model
        new_model (FlexBertModel): The larger model to be initialized
        mode (Union[str, TileMode]): The Phi-style weight tiling mode to use

    This function assumes that the new_model has more layers and a larger hidden size
    than the pretrained_model, but the same vocabulary size.
    """


    tile_embedding(pretrained_model.model.embeddings.tok_embeddings, new_model.model.embeddings.tok_embeddings, mode=mode)

    if hasattr(pretrained_model.model.embeddings, "norm"):
        tile_norm(pretrained_model.model.embeddings.norm, new_model.model.embeddings.norm, mode=mode)

    # Calculate the layer mapping
    pretrained_layers = len(pretrained_model.model.layers)
    new_layers = len(new_model.model.layers)

    print(f"small layer count {pretrained_layers}")
    print(f"new layer count {new_layers}")
    #layer_mapping = [round(i * pretrained_layers / new_layers) for i in range(new_layers)]
    # as HF is 0 indexed
    layer_mapping = [round(i * (pretrained_layers - 1) / (new_layers - 1)) for i in range(new_layers)]
    print(f"layer mapping {layer_mapping}")

    # Initialize layers
    for new_model_idx, pretrained_idx in enumerate(layer_mapping):

        new_model_layer = new_model.model.layers[new_model_idx]
        pretrained_layer = pretrained_model.model.layers[pretrained_idx]

        # Then tile the attention & mlp layers
        tile_linear(pretrained_layer.attn.Wqkv, new_model_layer.attn.Wqkv, linear_type=TileLinear.wqkv, mode=mode)
        # finally, tile the attention output layer
        tile_linear(pretrained_layer.attn.Wo, new_model_layer.attn.Wo, linear_type=TileLinear.default, mode=mode)
        # mlp_norm 
        tile_norm(pretrained_layer.mlp_norm, new_model_layer.mlp_norm, mode=mode)
        # mlp.Wi
        tile_linear(pretrained_layer.mlp.Wi, new_model_layer.mlp.Wi, linear_type=TileLinear.default, mode=mode)
        # mpl.Wo
        tile_linear(pretrained_layer.mlp.Wo, new_model_layer.mlp.Wo, linear_type=TileLinear.default, mode=mode)

    return new_model


# model.layers.0.attn.Wqkv.weight
# torch.Size([1680, 560])
# torch.Size([2880, 960])
# 3
# model.layers.0.attn.Wo.weight
# torch.Size([560, 560])
# torch.Size([960, 960])
# 4
# model.layers.0.mlp_norm.weight
# torch.Size([560])
# torch.Size([960])
# 5
# model.layers.0.mlp.Wi.weight
# torch.Size([1024, 560])
# torch.Size([1024, 960])
# 6
# model.layers.0.mlp.Wo.weight
# torch.Size([560, 512])
# torch.Size([960, 512])



