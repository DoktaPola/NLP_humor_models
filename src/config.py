import torch

VALUE_TO_FILL_NA = ''
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed_val = 42
warmup_steps = 1e2
sample_every = 100