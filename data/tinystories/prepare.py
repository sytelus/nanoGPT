import os
import sys
if os.path.abspath('data/') not in sys.path:
    sys.path.append(os.path.abspath('data/'))
import hf_prepare

if __name__ == '__main__':
    hf_prepare.prepare("roneneldan/TinyStories", "tinystories_v1")
    hf_prepare.prepare("roneneldan/TinyStories", "tinystories_v2", data_files={
        "train": "TinyStoriesV2-GPT4-train.txt",
        "validation": "TinyStoriesV2-GPT4-valid.txt",
    })