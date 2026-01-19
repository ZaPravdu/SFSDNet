import torch
from ..config import cfg
from model.VIC import Video_Counter


def get_model(config):
    state = torch.load(config.model_path)
    new_state = {}
    for k, v in state.items():
        name = k[7:] if k.startswith('module.') else k
        new_state[name] = v

    model = Video_Counter(cfg, config.cfg_data)
    model.load_state_dict(new_state, strict=True)
    model.eval()
    model = model.to(config.device)
    state = torch.load('./sdnet.pth')
    new_state = {}
    for k, v in state.items():
        name = k[7:] if k.startswith('module.') else k
        new_state[name] = v

    return model