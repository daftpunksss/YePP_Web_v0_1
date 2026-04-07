import torch
from hybrid import HybridNet

checkpoint_path = "../uas_hybridnet_oracle.ckpt"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = HybridNet().to(device)
print(model)
checkpoint = torch.load(checkpoint_path, map_location=device)
if 'state_dict' in checkpoint:
    state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
else:
    model.load_state_dict(checkpoint)
model.eval()