from typing import List, Union
import os
import sys
from datasets import load_dataset, Dataset,DatasetDict
from transformers import AutoTokenizer, PreTrainedTokenizerFast, AutoTokenizer
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
    ByteLevelBPETokenizer,
    SentencePieceBPETokenizer,
    SentencePieceUnigramTokenizer
)


unknown_token = "<unk>"
pad_token = "<pad>"
bos_token = "</s>"
eos_token = "</s>"
cls_token = "<cls>"
sep_token = "<sep>"
mask_token = "<mask>"
special_tokens_dict = {
    # key name must match PreTrainedTokenizer params
    "unk_token": unknown_token,
    "pad_token": pad_token,
    "bos_token": bos_token,
    "eos_token": eos_token,
    "cls_token": cls_token,
    "sep_token": sep_token,
    "mask_token": mask_token
}


def get_training_corpus(dataset, batch_size):
    return (
        dataset["train"][i : i + batch_size]["text"]
        for i in range(0, len(dataset["train"]), batch_size)
    )

def unique_dict_vals(tokens_dict):
    return list(set(tokens_dict.values()))

def train_bbpe_tokenizer(dataset_or_files, save_path=None,
                        batch_size=1000, vocab_size=50257,
                        special_tokens_dict=special_tokens_dict,
                        min_frequency = 2, # min freq for a token to be added to vocab
                        dropout=None,
                        add_prefix_space=False, # add space to first word of each sentence
                        ):
    # ByteLevelBPETokenizer is wrapper over models.BPE() for byte level encoding
    # this is better to use instead of example on HF website because it takes care of
    # many other things like adding default alphabets
    special_tokens = unique_dict_vals(special_tokens_dict)
    tokenizer = ByteLevelBPETokenizer(dropout=dropout, add_prefix_space=add_prefix_space)

    if isinstance(dataset_or_files, list):
        tokenizer.train(files=dataset_or_files,
                        vocab_size=vocab_size, special_tokens=special_tokens,
                        min_frequency=min_frequency, show_progress=True)
    else:
        tokenizer.train_from_iterator(get_training_corpus(dataset, batch_size),
                                        vocab_size=vocab_size, special_tokens=special_tokens,
                                        min_frequency=min_frequency, show_progress=True)

    if save_path:
        pttokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer,
                                              padding_size="left", truncation_side="left",
                                              **special_tokens_dict)
        pttokenizer.save_pretrained(save_path)

    return tokenizer

def train_sentencepiece_bpe_tokenizer(dataset_or_files, save_path=None,
                        batch_size=1000, vocab_size=50257,
                        special_tokens_dict=special_tokens_dict,
                        unknown_token=unknown_token,
                        min_frequency=2, dropout=None,
                        add_prefix_space=False, # add space to first word of each sentence
                        replacement="\u2581", # replacement for space
                        fuse_unk=False, # fuse unknown token
                        limit_alphabet=1000,
                        initial_alphabet=[]
                        ):
    special_tokens = unique_dict_vals(special_tokens_dict)
    tokenizer = SentencePieceBPETokenizer(dropout=dropout, unk_token=unknown_token,
                                          add_prefix_space=add_prefix_space,
                                          replacement=replacement, fuse_unk=fuse_unk)

    if isinstance(dataset_or_files, list):
        tokenizer.train(files=dataset_or_files,
                        vocab_size=vocab_size,
                        special_tokens=special_tokens,
                        min_frequency=min_frequency,
                        limit_alphabet=limit_alphabet,
                        initial_alphabet=initial_alphabet, show_progress=True)
    else:
        tokenizer.train_from_iterator(get_training_corpus(dataset, batch_size),
                                        vocab_size=vocab_size,
                                        special_tokens=special_tokens,
                                        min_frequency=min_frequency,
                                        limit_alphabet=limit_alphabet,
                                        initial_alphabet=initial_alphabet, show_progress=True)

    if save_path:
        pttokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer,
                                              padding_size="left", truncation_side="left",
                                              **special_tokens_dict)
        pttokenizer.save_pretrained(save_path)

    return tokenizer

def train_sentencepiece_unigram_tokenizer(dataset_or_files, save_path=None, batch_size=1000, vocab_size=50257,
                        special_tokens_dict=special_tokens_dict,
                        unknown_token=unknown_token,
                        add_prefix_space=False, # add space to first word of each sentence
                        replacement="\u2581", # replacement for space
                        initial_alphabet=[] # pre_tokenizers.ByteLevel.alphabet()
                        ):
    special_tokens = unique_dict_vals(special_tokens_dict)
    tokenizer = SentencePieceUnigramTokenizer(add_prefix_space=add_prefix_space, replacement=replacement)

    if isinstance(dataset_or_files, list):
        tokenizer.train(files=dataset_or_files,
                        vocab_size=vocab_size,
                        special_tokens=special_tokens,
                        unk_token= unknown_token,
                        initial_alphabet=initial_alphabet, show_progress=True)
    else:
        tokenizer.train_from_iterator(get_training_corpus(dataset, batch_size),
                                        vocab_size=vocab_size,
                                        special_tokens=special_tokens,
                                        unk_token= unknown_token,
                                        initial_alphabet=initial_alphabet, show_progress=True)

    if save_path:
        pttokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer,
                                              padding_size="left", truncation_side="left",
                                              **special_tokens_dict)
        pttokenizer.save_pretrained(save_path)

    return tokenizer

def load_tokenizer(path):
    os.makedirs(path, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(path)
    return tokenizer

if __name__ == "__main__":
    # Simple test
    dataset_name = "wikitext-103-raw-v1" #"wikitext-2-raw-v1"

    dataset = load_dataset("wikitext", name=dataset_name)
    test_str = "Hi ~x 😀 1 123"
    base_save_path = os.path.join(r"e:\tokenizers", dataset_name)

    save_path=os.path.join(base_save_path, "bbpe")
    tokenizer = train_bbpe_tokenizer(dataset, save_path=save_path)
    tokenizer = load_tokenizer(save_path)
    r = tokenizer.encode(test_str)
    print(len(test_str), len(r), r)
    r = tokenizer.decode(r)
    print(r==test_str, len(r), r)

    save_path=os.path.join(base_save_path, "sentencepiece_bpe")
    tokenizer = train_sentencepiece_bpe_tokenizer(dataset, save_path=save_path)
    tokenizer = load_tokenizer(save_path)
    r = tokenizer.encode(test_str)
    print(len(test_str), len(r), r)
    r = tokenizer.decode(r)
    print(r==test_str, len(r), r)

    save_path=os.path.join(base_save_path, "sentencepiece_unigram")
    tokenizer = train_sentencepiece_unigram_tokenizer(dataset, save_path=save_path)
    tokenizer = load_tokenizer(save_path)
    r = tokenizer.encode(test_str)
    print(len(test_str), len(r), r)
    r = tokenizer.decode(r)
    print(r==test_str, len(r), r)


