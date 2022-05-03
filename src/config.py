import torch

VALUE_TO_FILL_NA = ''
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
