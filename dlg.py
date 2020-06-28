import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from result import Result

""" 
The DLG and iDLG attack were taken and adapted from the repository linked in the original paper:
"iDLG: Improved Deep Leakage from Gradients" by Zhao et Al.
https://arxiv.org/pdf/2001.02610.pdf
https://github.com/PatrickZH/Improved-Deep-Leakage-from-Gradients
"""


def attack(model, train_dataset, parameter, device, improved):
    # select attacked ids
    ids = np.random.permutation(len(train_dataset))[:parameter["batch_size"]]

    # prepare attacked batch (orig)
    orig_data = torch.Tensor(parameter["batch_size"], 1, parameter["shape_img"][0], parameter["shape_img"][1]).to(
        device)
    orig_label = torch.Tensor(parameter["batch_size"])
    orig_label = orig_label.long().to(device)
    orig_label = orig_label.view(parameter["batch_size"], )

    for index, id in enumerate(ids):
        orig_data[index] = transforms.ToTensor()(train_dataset[id][0])
        orig_label[index] = torch.Tensor([train_dataset[id][1]]).long().to(device).view(1, )

    # calculate orig gradients
    orig_out = model(orig_data)
    criterion = nn.CrossEntropyLoss().to(device)
    y = criterion(orig_out, orig_label)
    gradient = torch.autograd.grad(y, model.parameters())
    gradient = list((_.detach().clone() for _ in gradient))

    # prepare dummy data
    dummy_data = torch.rand((parameter["batch_size"], 1, parameter["shape_img"][0], parameter["shape_img"][1])).to(
        device).requires_grad_(True)
    dummy_label = torch.randn((parameter["batch_size"], parameter["num_classes"])).to(device).requires_grad_(True)

    # optimizer setup
    if not improved:
        optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=parameter["dlg_lr"])
    else:
        optimizer = torch.optim.LBFGS([dummy_data, ], lr=parameter["dlg_lr"])
        # predict label of dummy gradient
        idlg_pred = torch.argmin(torch.sum(gradient[-2], dim=-1), dim=-1).detach().reshape((1,)).requires_grad_(
            False)

    res = Result(parameter)
    res.set_origin(orig_data.cpu().detach(), orig_label)

    for iteration in range(parameter["dlg_iterations"]):

        # clears gradients, computes loss, returns loss
        def closure():
            optimizer.zero_grad()
            pred = model(dummy_data)
            if not improved:
                dummy_loss = - torch.mean(
                    torch.sum(torch.softmax(dummy_label, -1) * torch.log(torch.softmax(pred, -1)), dim=-1))
            else:
                dummy_loss = criterion(pred, idlg_pred)

            dummy_gradient = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

            grad_diff = 0
            for gx, gy in zip(dummy_gradient, gradient):
                grad_diff += ((gx - gy) ** 2).sum()
            grad_diff.backward()
            return grad_diff

        optimizer.step(closure)

        if iteration % parameter["log_interval"] == 0:
            res.add_snapshot(dummy_data.cpu().detach())

    return res


def idlg(model, train_dataset, parameter, device):
    return attack(model, train_dataset, parameter, device, True)


def dlg(model, train_dataset, parameter, device):
    return attack(model, train_dataset, parameter, device, False)
