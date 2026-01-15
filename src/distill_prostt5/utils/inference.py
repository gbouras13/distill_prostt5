
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from pathlib import Path
from loguru import logger
import json

def write_predictions(
    predictions: Dict[str, Dict[str, Tuple[List[str], Any, Any]]],
    out_path: Path,
    mask_prop_threshold: float,
    plddt_head_flag: bool
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
    # same as CNN with extra 1 for 'X' masking
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
        19: "Y",
        20: "X"
    }
    


    with open(out_path, "w+") as out_f:

        for contig_id, rest in predictions.items():
            prediction_contig_dict = predictions[contig_id]

            # Filter out entries where the length of the value is 0
            # Issue #47

            prediction_contig_dict = {
                k: v for k, v in prediction_contig_dict.items() if len(v) > 0
            }

            if plddt_head_flag:

                for key, (pred, mean_prob, all_prob, all_plddts) in prediction_contig_dict.items():
                    # masking with plddt
                    for i in range(len(pred)):
                        if all_plddts[i] < mask_prop_threshold:
                            pred[i] = 20

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
                            for seq_id, (yhats, _, _, _) in prediction_contig_dict.items()
                        ]
                    )
                )
            
            else:

                for key, value in prediction_contig_dict.items():
                    pred = value[0]
                    mean_prob = value[1] if len(value) > 1 else None
                    all_prob = value[2] if len(value) > 2 else None
                    plddt = value[3] if len(value) > 3 else None
                    
                    all_prob = all_prob * 100
                    # masking with probs
                    for i in range(len(pred)):
                        if all_prob[i] < mask_prop_threshold:
                            pred[i] = 20


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
                            for seq_id, (yhats, _, _) in prediction_contig_dict.items()
                        ]
                    )
                )

    logger.info(f"Finished writing results to {out_path}")
    return None

def write_probs(
    predictions: Dict[str, Dict[str, Tuple[int, float, Union[int, np.ndarray]]]],
    output_path_mean: Path,
    plddt_head_flag: bool
) -> None:
    """
    Write all ProstT5 encoder + CNN probabilities and mean probabilities to output files.

    Args:
        predictions (Dict[str, Dict[str, Tuple[int, float, Union[int, np.ndarray]]]]):
            Predictions dictionary containing contig IDs, sequence IDs, probabilities, and additional information.
        output_path_mean (str): Path to the output file for mean probabilities.

    Returns:
        None
    """

    if plddt_head_flag:

        with open(output_path_mean, "w+") as out_f:
            for contig_id, rest in predictions.items():
                prediction_contig_dict = predictions[contig_id]

                for seq_id, (N, mean_prob, N, N) in prediction_contig_dict.items():
                    out_f.write("{},{}\n".format(seq_id, mean_prob))

    else:

        with open(output_path_mean, "w+") as out_f:
            for contig_id, rest in predictions.items():
                prediction_contig_dict = predictions[contig_id]

                for seq_id, (N, mean_prob, N) in prediction_contig_dict.items():
                    out_f.write("{},{}\n".format(seq_id, mean_prob))


def write_plddt(
    predictions: Dict[str, Dict[str, Tuple[int, float, Union[int, np.ndarray]]]],
    output_path_plddt: Path,
) -> None:
    """
    Write all encoder pLDDT

    Args:
        predictions (Dict[str, Dict[str, Tuple[int, float, Union[int, np.ndarray]]]]):
            Predictions dictionary containing contig IDs, sequence IDs, probabilities, and additional information.
        output_path_plddt (str): Path to the output file for plddts.

    Returns:
        None
    """
    with open(output_path_plddt, "w+") as out_f:
        for contig_id, rest in predictions.items():
            prediction_contig_dict = predictions[contig_id]

            for seq_id, (N,N, N, plddt_slice) in prediction_contig_dict.items():

                plddt_list = (
                        plddt_slice.flatten().tolist()
                        if isinstance(plddt_slice, np.ndarray)
                        else plddt_slice
                    )

                rounded_list = [round(num, 2) for num in plddt_list]


                per_residue_plddt = {"seq_id": seq_id, "probability": rounded_list}


                json_data = json.dumps(per_residue_plddt)
                

                # Write the JSON string to the file
                out_f.write(json_data + "\n")  # Add a newline after each JSON object


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
