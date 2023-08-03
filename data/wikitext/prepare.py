import os
import sys
if os.path.abspath('data/') not in sys.path:
    sys.path.append(os.path.abspath('data/'))
import hf_prepare

if __name__ == '__main__':
    #hf_prepare.prepare("wikitext", "wikitext-103-raw-v1")
    hf_prepare.prepare("wikitext", "wikitext-2-raw-v1")
