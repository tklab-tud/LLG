import torch
import numpy as np

def dlg(model, train_dataset, parameter, device):
    dummy_data = torch.randn(np.prod(parameter["shape_img"])).to(device).requires_grad_(True)
    dummy_label = torch.randn( parameter["num_classes"]).to(device).requires_grad_(True)

    return (dummy_label, dummy_data)