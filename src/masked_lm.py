"""
This module defines a memory efficient dataset for training models on raw unstructured text data.
"""
import os
import mmap
import numpy as np
import multiprocessing
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
from typing import Union, Tuple, Dict, Any, List
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from .base_datasets import BaseDataset
from .mixins import TextMixin
from multiprocessing import shared_memory

def init_worker(tokenizer_path:os.PathLike) -> None:
    """Initialize the tokenizer in the worker process. Each worker process will have its own tokenizer instance.

    Args:
        tokenizer_path (os.PathLike): The path to the tokenizer to be used.
    """    
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

def tokenize_line(line:str) -> List[int]:
    """Tokenize a single line of text using the global tokenizer instance.

    Args:
        line (str): The text to be tokenized.

    Returns:
        List[int]: The tokenized text as a list of token ids.
    """    
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(line.strip()))


class MaskedLMDataset(TextMixin, BaseDataset):
    def __init__(
        self,
        name:str,
        data_path:os.PathLike,
        split:str,
        text_file:os.PathLike,
        tokenizer:Union[PreTrainedTokenizer, PreTrainedTokenizerFast]=None,
        mask_prob:float=0.0,
        block_size:int=512,
    ):
        """A universal dataset for raw unstructured text data that supports masked language modeling.

        Args:
            name (str): The name of the dataset, used for saving the index files.
            data_path (os.PathLike): The path where the data is stored.
            split (str): The split of the data. One of 'train', 'val', or 'test'.
            text_file (os.PathLike): The path to the text file containing the raw text data.
            tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast], optional): A tokenizer class that implements
                the Huggingface tokenizer API. Used to tokenize text data. Defaults to None.
                If None, then the BERT base uncased tokenizer will be used by default:
                BertTokenizer.from_pretrained("bert-base-uncased").
            mask_prob (float, optional): The probability of masking a token in the text data. Defaults to 0.0, so no masking is done.
            block_size (int, optional): How many tokens should be in one block. One block is equal to one training example, i.e.
                a single text sequence. This is the maximum sequence length. The text file will be sliced into chunks
                of <block_size> tokens. Defaults to 512.
        """
        # set before calling super().__init__ as the constructor of
        # BaseDataset uses property "data_dir", and therefore "name" must be set before calling super().__init__
        self.name = name
        super().__init__(
            data_path=data_path,
            split=split,
            tokenizer=tokenizer,
            max_seq_len=block_size,
            mlm_probability=mask_prob,
        )
        self.text_file = text_file
        # Subtract 2 for CLS and SEP tokens -> each slice of the text data will be block_size-2 tokens long
        # if we then add 2 tokens for CLS and SEP tokens, the total length will be block_size
        self.block_size = block_size - 2
        self.token_file = os.path.join(self.path_to_split, f'mlm_{self.name}_{self.split}.bin')
        self.index_file = os.path.join(self.path_to_split, f'mlm_{self.name}_{self.split}.idx')
        self.index_entry_size = 16  # Each index entry has two int64 (offset and length)

        if not self.index_exists():
            self.preprocess()

    @property
    def data_dir(self) -> str:
        """
        Name of the directory in self.data_path where the data is stored.
        """        
        return self.name

    def __getstate__(self):
        """
        Exclude non-picklable objects from the pickled state.
        This is necessary for multiprocessing environments, as is the case when using pytorch dataloaders.
        """
        state = self.__dict__.copy()
        state.pop('fp', None)  # Remove the file pointer
        state.pop('mmap_file', None)  # Remove the mmap object
        return state

    def __setstate__(self, state):
        """
        Reinitialize the non-picklable objects after unpickling.
        Each worker process will call this method when it is created, and therefore has its own file pointer and mmap object.
        """
        self.__dict__.update(state)
        # Reopen the file and recreate the mmap object
        self.fp = open(self.token_file, 'rb')
        self.mmap_file = mmap.mmap(self.fp.fileno(), 0, access=mmap.ACCESS_READ)

    def load(self) -> None:
        self.build_sequences()

    def preprocess(self) -> None:
        """
        Tokenize the text file and store tokenized data into a binary mmap file.
        Create an index file that stores the offset and length of each tokenized sequence/line.
        """
        n_unk_tokens = 0
        n_total_tokens = 0
        # Calculate total lines for progress bar
        total_lines = sum(1 for _ in open(self.text_file, 'r', encoding='utf-8'))
        with open(self.text_file, 'r', encoding='utf-8') as f_in, \
            open(self.token_file, 'wb') as f_out, \
            open(self.index_file, 'wb') as f_idx:
            
            offset = 0
            batch_size = 10000 # works well for large files

            # Initialize multiprocessing pool
            pool = multiprocessing.Pool(initializer=init_worker, initargs=(self.tokenizer_path,))
            pbar = tqdm(total=total_lines, desc="Tokenizing")

            lines = []
            for line in f_in:
                lines.append(line)
                if len(lines) >= batch_size: # collect lines until batch_size is reached
                    # Tokenize the batch in parallel
                    tokenized_lines = pool.map(tokenize_line, lines)
                    # Write tokenized data to file and update index
                    for tokens in tokenized_lines:
                        length = len(tokens)
                        tokens = np.array(tokens, dtype=np.int32)
                        n_unk_tokens += (tokens == self.tokenizer.unk_token_id).sum() # count unknown tokens
                        n_total_tokens += length
                        tokens.tofile(f_out)
                        # Write offset (current position) and length as int64 to index file
                        f_idx.write(np.array([offset, length], dtype=np.int64).tobytes())
                        # Tokens(!) are stored in int32, so each token is 4 bytes
                        # -> advance offset by "number of tokens in the current line" * 4 bytes
                        offset += length * 4
                        pbar.update(1) # one line processed, update progress bar
                    lines = [] # clear the batch of lines

            # Process remaining lines
            if lines:
                tokenized_lines = pool.map(tokenize_line, lines)
                for tokens in tokenized_lines:
                    length = len(tokens)
                    tokens = np.array(tokens, dtype=np.int32)
                    n_unk_tokens += (tokens == self.tokenizer.unk_token_id).sum()
                    n_total_tokens += length
                    tokens.tofile(f_out)
                    f_idx.write(np.array([offset, length], dtype=np.int64).tobytes())
                    offset += length * 4
                    pbar.update(1)

            pbar.close()
            pool.close()
            pool.join()

        self.log(f'Preprocessing complete. Processed {n_total_tokens} tokens, '
                 f'found {n_unk_tokens}({n_unk_tokens/n_total_tokens*100:.05f}%) unknown tokens.')

    def build_sequences(self) -> None:
        """
        Slice the tokenized data into blocks of a specific size.
        This means potentially concatenating multiple lines of the tokenized text file to form a sequence of tokens (if 
        they together have not more than block_size tokens).
        This allows for less padding, more efficient memory usage and richer context for the model.
        """
        # Memory-map the index file
        self.index_fp = open(self.index_file, 'rb')
        self.index_mmap = mmap.mmap(self.index_fp.fileno(), 0, access=mmap.ACCESS_READ)

        items = []
        current_offset = 0
        current_length = 0

        num_lines = self.get_num_lines()

        for idx in tqdm(range(num_lines), desc="Building sequences"):
            offset, length = self.get_index_entry(idx) # get position of one line in the tokenized text file
            if current_length == 0:
                current_offset = offset

            # Handle lines longer than block_size
            if length >= self.block_size:
                if current_length > 0:
                    # if we encounter a line that is longer than block_size, and we are currently in the middle of a chunk,
                    # then we need to stop the current chunk
                    items.append([current_offset, current_length])

                # Split the line into chunks of block_size
                num_splits = (length + self.block_size - 1) // self.block_size
                for i in range(num_splits):
                    split_offset = offset + i * self.block_size * 4  # 4 bytes per int32 token
                    split_length = min(self.block_size, length - i * self.block_size)
                    items.append([split_offset, split_length])
                
                current_length = 0 # reset current_length, that means we start a new chunk
            else:
                if current_length + length <= self.block_size:
                    # if there is still space in the current chunk, add the current line to it
                    current_length += length
                else:
                    # if there is no space in the current chunk, save it as a sequence
                    items.append([current_offset, current_length])
                    # ... and start a new chunk with the current line (that is too long for the previous chunk)
                    current_offset = offset
                    current_length = length

        # Add any remaining tokens as the last sequence
        if current_length > 0:
            items.append([current_offset, current_length])

        # wrap in shared memory for multiprocessing
        self._items = SharedMemoryArray(np.array(items, dtype=np.int64))

        # Close the index mmap and file since it's no longer needed
        self.index_mmap.close()
        self.index_fp.close()
        del self.index_mmap
        del self.index_fp

    @property
    def items(self):
        """
        Wrapper for easy access.
        """
        return self._items.array

    def get_num_lines(self) -> int:
        """Calculate the number of lines in the index file.

        Returns:
            int: The number of lines in the index file.
        """
        index_file_size = os.path.getsize(self.index_file) # get the size of the index file in bytes
        num_lines = index_file_size // self.index_entry_size  # Each index entry has two int64 (offset [8 byte] and length [8 byte]) -> 16 bytes
        return num_lines

    def get_index_entry(self, idx:int) -> Tuple[int, int]:
        """Retrieve an index entry (offset and length) from the mmap index file.

        Args:
            idx (int): The current index in the index file. Can also be seen as the line number of the lines in the text file.

        Returns:
            Tuple[int, int]: The offset, at which position in the tokenized text file the sentence is located,
                and length of the sequence (in bytes).
        """
        start = idx * self.index_entry_size
        end = start + self.index_entry_size
        data = self.index_mmap[start:end]
        offset, length = np.frombuffer(data, dtype=np.int64)
        return offset, length

    def __getitem__(self, idx:int) -> Dict[str, Any]:
        # Ensure that the mmap file is initialized -> not the case if not using multiprocessing
        if not hasattr(self, 'mmap_file'):
            self.fp = open(self.token_file, 'rb')
            self.mmap_file = mmap.mmap(self.fp.fileno(), 0, access=mmap.ACCESS_READ)

        result_dict = dict()
        # each items stores the position and length of one block/sequence in the tokenized text file
        offset, length = self.items[idx]
        # sequence length is in number of tokens, each token is 4 bytes (length * 4 = length in bytes)
        tokens = np.frombuffer(self.mmap_file[offset:offset + length * 4], dtype=np.int32).tolist()

        text_dict = self.get_text(tokens)
        result_dict.update(text_dict)
        result_dict["id"] = idx

        return result_dict

    def get_index_files(self) -> Tuple[str]:
        """Returns a tuple of strings, where each string is the name of an index file containing the data
        for the split of the dataset.

        Returns:
            Tuple[str]: Tuple of strings, where each string is the name of an index file containing the data.
        """
        token_file_ = os.path.join(self.split, f'mlm_{self.name}_{self.split}.bin')
        index_file_ = os.path.join(self.split, f'mlm_{self.name}_{self.split}.idx')
        # output used in "index_exists()" of supertype BaseDataset
        # does "os.path.join(self.path_to_data, index_file)" for each index file
        # since we have the index file in the split directory, we need to include the split directory in the index file paths
        return (token_file_, index_file_)
    
    def collater(self, samples:List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_tensors = super().collater(samples)
        batch_tensors = self.apply_mask(batch_tensors)
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
