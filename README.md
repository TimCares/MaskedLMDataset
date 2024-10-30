# MaskedLMDataset
This repository provides a generic implementation of a masked language modeling dataset, used for training language models, in native pytorch.

It uses mmap files and python's shared memory for memory efficiency during training, and is able to handle large text files.

**Note:** I currently do not plan to create a pip package from this repository, as it is unclear whether I will continuously maintain this repository.
Future improvements may include support for sharded/multiple text files and increased memory efficiency.

## Supported Functionality
- Masking of tokens, either purely token-based or whole word mask
- Concatenates multiple lines of text (from the text file) to minimize padding
- Memory efficiency by using mmap for the tokenized data, and sharing the index with multiple processes (in a multiprocessing env) via python's multiprocessing.shared_memory module
- Different sequence lengths without re-tokenizing the data


## Usage
Create a single text file of raw unstructured text. This is the data that will be used for training the language model. Multiple
GBs are not a problem due to the usage of mmap files.

```python
dataset = MaskedLMDataset(
    name='my_dataset', # how the dataset should be called, used when saving the tokenized data
    data_path='path/to/the/data', # where the tokenized data should be persisted
    split='train', # for which split the data should be used, useful to differentiate multiple datasets for different splits
    text_file='path/to/the/text_file.txt', # where the file with the text, used for training, is stored
    tokenizer=None, # tokenizer implementing the api from HuggingFace tokenizers, used for tokenizing and encoding the text
    mask_prob=0.15, # the masking probability, BERT uses 0.15
    whole_word_mask=True, # if whole words should be masked at a time, this is encouraged
    block_size=512, # how many tokens should be in each block, this corresponds to the maximum sequence length used during training
)

dataset.load() # after this, the dataset is ready to use
```

An [example](example.txt) text file to use is provided in the root of this repository.
Creating the dataset automatically tokenizes and saves the data. Depending on the amount of text data this can take some time.

Calling the `load` method creates the index that defines the individual training examples. Here, multiple sentences/lines in the
original text file (path/to/the/text_file.txt) will be concatenated if the number of tokens is still below the block size (maximum
sequence length).

For example, the following lines in a text file together have fewer tokens than 512 (example used above), so they will be concatenated
to one sequence, and are therefore part of the same training example.

```txt
Actual Leading Causes of Death
An unhealthy diet is actually a leading cause of death in the United States.
```

Since this process does not alter the data itself, one can create multiple `MaskedLMDataset` datasets with different block sizes
from the same text file.

```python
dataset_1 = MaskedLMDataset(
    name='my_dataset',
    data_path='path/to/the/data',
    split='train',
    text_file='path/to/the/text_file.txt',
    tokenizer=None,
    mask_prob=0.15,
    whole_word_mask=True,
    block_size=512, # <----------------------
)
dataset_1.load() # generate sequences of length 512

dataset_2 = MaskedLMDataset(
    name='my_dataset',
    data_path='path/to/the/data',
    split='train',
    text_file='path/to/the/text_file.txt',
    tokenizer=None,
    mask_prob=0.15,
    whole_word_mask=True,
    block_size=1024, # <----------------------
)
dataset_2.load() # generate sequences of length 1024
```

Even though we have two datasets, they are based on the same **name**, the same **data path**,
the same **split**, and the same **text file**, making them identical. The tokenized data will only be created once,
but `dataset_2` will feature longer sequences of the same data.

After calling `load`, the dataset can easily be used with a torch DataLoader:

```python
# create the dataset as seen above
dataset = ...

from torch.utils.data import DataLoader

# example dataloader args
batch_size = 256
num_workers = 8
shuffle = True
drop_last = True

dataloader = DataLoader(
    dataset,
    collate_fn=dataset.collater,
    batch_size=batch_size,
    num_workers=num_workers,
    sampler=None,
    shuffle=shuffle,
    drop_last=drop_last
)

for i, batch in enumerate(dataloader):
    print(batch)
    if i == 10:
        break

```

Each batch will be a dictionary of tensors, which represent the batched data.
Keys include:

| Key | Description | Shape | Optional |
|:----------------------------------------|:----------:|:-----:|:-----:|
| text | The tokenized, encoded, padded, and batched text data. If masking is used, some tokens will be masked | (B, T) | ❌ |
| padding_mask | The padding mask. 1 indicates padding, 0 actual tokens. | (B, T) | ❌ |
| id | The id of each training example. | (B,) | ❌ |
| mask_labels | Whether token should be masked. 1 for masking, 0 for no masking. | (B, T) | ✅ |
| labels | The token id of the tokens that are masked, -100 for others (useful for argument `ignore_index` in [torch.nn.functional.cross_entropy](https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html)).  | (B, T) | ✅ |

If the argument `mask_prob` is 0 (mask_prob=0.0), then the optional keys will not be available, as masking is disabled.
