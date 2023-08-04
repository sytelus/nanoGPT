# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
import os

wandb_log = True
wandb_project = 'nanogpt-wikitext103'
wandb_run_name=None

out_dir = os.path.join(os.environ.get('DATA_ROOT', 'out'), 'gpt2-124M-wikitext103')

dataset = 'wikitext-103-raw-v1'
data_dir = os.path.join(os.environ.get('DATA_ROOT', '/data'), 'tokenized', dataset, 'tiktoken')
train_file = os.path.join(data_dir, 'train.bin')
val_file = os.path.join(data_dir, 'validation.bin')
test_file = os.path.join(data_dir, 'test.bin')

# model
# 355M params https://arxiv.org/pdf/1909.08053.pdf
n_layer = 24
n_head = 16
n_embd = 1024

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 16
learning_rate = 6e-4
block_size = 1024
gradient_accumulation_steps = 4 * 8

learning_rate = 1.5e-4
min_lr = 1e-5
max_iters = 300000
lr_decay_iters = 300000
warmup_iters = 3000

# eval stuff
eval_interval = 1000
log_interval = 10

