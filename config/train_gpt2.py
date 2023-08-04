# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = 'owt'
wandb_run_name='gpt2-124M'

# these make the total batch size be ~0.5M tokens/batch
# global_batch_size = local_batch_size * gpu_count * grad_acc_steps = 12 * 8 * 5 = 480
# tokens_per_iter = global_batch_size * context_length = 480 * 1024 = 491,520 tokens/iteration
# OpenAI used global_batch_size = 512
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8

# total_training_tokens = tokens_per_iter * max_iters = 491,520 * 600,000 = 294,912,000,000 # ~300B
# OWT has 9B tokens, so this is 300B/9B = ~33 epochs
# OpenAI did 800K iterations so they did 800K * 512 * 1024 = 409B tokens
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1
