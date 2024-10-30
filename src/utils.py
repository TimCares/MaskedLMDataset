"""
Module for utility functions.
"""
import torch
import numpy as np
from typing import List, Any, Dict, Union, Iterable, Tuple
from multiprocessing import shared_memory

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

class SharedMemoryArray:
    """
    Wrapper around NumPy arrays using multiprocessing.shared_memory for shared memory.
    See: https://docs.python.org/3/library/multiprocessing.shared_memory.html
    """

    def __init__(self, array:np.ndarray):
        self.shape = array.shape
        self.dtype = array.dtype
        self.shm = shared_memory.SharedMemory(create=True, size=array.nbytes)
        self.array = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)
        self.array[:] = array[:]

    def __getstate__(self) -> Tuple[Tuple[int, ...], np.dtype, str]:
        """Prepare the object's state for pickling. Useful for multiprocessing when
        we do not want to copy the entire object to each worker process.

        Returns:
            Tuple[Tuple[int, ...], np.dtype, str]: A tuple containing the shape of the array (tuple[0]),
                the dtype (tuple[1]) of the array, and the name of the shared memory block (tuple[2]).
        """
        return (self.shape, self.dtype, self.shm.name)

    def __setstate__(self, state:Tuple[Tuple[int, ...], np.dtype, str]) -> None:
        """Restore the object's state from the unpickled state. This is executed in every worker process
        when using multiprocessing.

        Args:
            state (Tuple[Tuple[int, ...], np.dtype, str]): The state of the object to be restored.
                A tuple containing the shape of the array (tuple[0]), the dtype (tuple[1]) of the array,
                and the name of the shared memory block (tuple[2]).
        """
        self.shape, self.dtype, shm_name = state
        # get shared memory by name
        self.shm = shared_memory.SharedMemory(name=shm_name)
        # create array from shared memory, this is the reconstructed array
        self.array = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)

    def __del__(self) -> None:
        """Clean up shared memory resources."""
        try:
            self.shm.close()
            self.shm.unlink()
        except FileNotFoundError:
            pass  # The shared memory block might have been already unlinked.
