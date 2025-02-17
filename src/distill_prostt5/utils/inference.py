
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from pathlib import Path
from loguru import logger

def write_predictions(
    predictions: Dict[str, Dict[str, Tuple[List[str], Any, Any]]],
    out_path: Path
) -> None:
    """
    Write predictions to an output file.

    Args:
        predictions (Dict[str, Dict[str, Tuple[List[str], Any, Any]]]): Predictions dictionary containing contig IDs, sequence IDs, predictions, and additional information.
        out_path (Path): Path to the output file.
        proteins_flag (bool): Flag indicating whether the predictions are in proteins mode or not.
        mask_threshold (float): between 0 and 100 - below this ProstT5 confidence, 3Di predictions are masked

    Returns:
        None
    """
    # same as CNN
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
        19: "Y"
    }
    


    with open(out_path, "w+") as out_f:
        for contig_id, rest in predictions.items():
            prediction_contig_dict = predictions[contig_id]

            # Filter out entries where the length of the value is 0
            # Issue #47

            prediction_contig_dict = {
                k: v for k, v in prediction_contig_dict.items() if len(v) > 0
            }



            # no contig_id
            out_f.write(
                "".join(
                    [
                        ">{}\n{}\n".format(
                            f"{seq_id}",
                            "".join(
                                list(map(lambda yhat: ss_mapping[int(yhat)], yhats))
                            ),
                        )
                        for seq_id, yhats in prediction_contig_dict.items()
                    ]
                )
            )
    logger.info(f"Finished writing results to {out_path}")
    return None

def toCPU(tensor: torch.Tensor) -> np.ndarray:
    """
    Move a tensor to CPU and convert it to a NumPy array.

    Args:
        tensor (torch.Tensor): Input tensor.

    Returns:
        np.ndarray: NumPy array.
    """
    if len(tensor.shape) > 1:
        return tensor.detach().cpu().squeeze(dim=-1).numpy()
    else:
        return tensor.detach().cpu().numpy()
