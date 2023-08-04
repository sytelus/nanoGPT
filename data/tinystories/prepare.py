import nanogpt_common.hf_data_prepare as hf_data_prepare

if __name__ == '__main__':
    hf_data_prepare.prepare("roneneldan/TinyStories", "tinystories_v1")
    hf_data_prepare.prepare("roneneldan/TinyStories", "tinystories_v2", data_files={
        "train": "TinyStoriesV2-GPT4-train.txt",
        "validation": "TinyStoriesV2-GPT4-valid.txt",
    })