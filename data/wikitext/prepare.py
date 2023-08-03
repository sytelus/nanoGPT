# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
from tqdm import tqdm
import numpy as np
import tiktoken
import datasets
from datasets import load_dataset # huggingface datasets

from functools import partial

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc

if __name__ == '__main__':
    data_root = os.environ.get('DATA_ROOT', os.path.dirname(__file__))

    dataset_name = "wikitext-103-raw-v1" #"wikitext-2-raw-v1"

    dataset = load_dataset("wikitext", name=dataset_name)
    print(f'loaded dataset {dataset_name}, {len(dataset["train"])} rows')

    # we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
    enc = tiktoken.get_encoding("gpt2")
    def process(enc, example):
        ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
        # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
        out = {'ids': ids, 'len': len(ids)}
        return out

    process_enc = partial(process, enc)
    # tokenize the dataset
    tokenized = dataset.map(
        process_enc,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    tokenized_path = os.path.join(data_root,'tokenized', dataset_name, 'tiktoken')
    os.makedirs(tokenized_path, exist_ok=True)

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(tokenized_path, f'{split}.bin')
        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

    # train.bin is ~17GB, val.bin ~8.5MB
    # train has ~9B tokens (9,035,582,198)
    # val has ~4M tokens (4,434,897)

    # to read the bin files later, e.g. with numpy:
    # m = np.memmap('train.bin', dtype=np.uint16, mode='r')
