import nanogpt_common.hf_data_prepare as hf_data_prepare

if __name__ == '__main__':
    hf_data_prepare.prepare("wikitext", "wikitext-103-raw-v1")
    hf_data_prepare.prepare("wikitext", "wikitext-2-raw-v1")
