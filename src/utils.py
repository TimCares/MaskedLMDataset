"""
Module for utility functions.
"""
import torch
import numpy as np
from typing import List, Any, Dict, Union, Iterable

def check_int_str_list_type(l: List[Any]) -> Any:
    """Check if all elements in a list are either of type int or str.

    Args:
        l (List[Any]): The list to check.

    Returns:
        Any: The type of the elements in the list, either int, str. If the list contains elements of both types,
            or all elements or of another type (e.g. float), returns Any.
    """        
    if all(isinstance(item, int) for item in l):
        return int
    elif all(isinstance(item, str) for item in l):
        return str
    else:
        return Any

def collater(samples:List[Dict[str, Union[torch.Tensor, np.ndarray, Iterable]]]) -> Dict[str, torch.Tensor]:
    """Batches a set of items, where each item is a dictionary of tensors, numpy arrays, or iterable objects.
    The result is a dictionary of tensors, with the same keys as the input items.
    All items are expected to have the exact same(!) keys, and the values associated
    with each key are stacked in a new tensor.

    Args:
        samples (List[Dict[str, Union[torch.Tensor, np.ndarray, Iterable]]]): A list of items to batch.
            Each item must be a dictionary.

    Returns:
        Dict[str, torch.Tensor]: The batched items.
    """    
    batch_tensors = {}
    for tensor_key in samples[0]: # iterate over all keys
        if isinstance(samples[0][tensor_key], torch.Tensor):
            batch_tensors[tensor_key] = torch.stack([d[tensor_key] for d in samples])
        elif isinstance(samples[0][tensor_key], np.ndarray):
            batch_tensors[tensor_key] = torch.from_numpy(np.stack([d[tensor_key] for d in samples]))
        else:
            batch_tensors[tensor_key] = torch.tensor([d[tensor_key] for d in samples], dtype=torch.long)

    return batch_tensors
