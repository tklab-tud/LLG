import torch
import numpy as np
from torchvision import datasets, transforms
import torch.nn as nn
from PIL import Image

def closure():
    print("")

def attack(model, train_dataset, parameter, device, improved):
    dummy_data = torch.rand((parameter["batch_size"], 1 ,parameter["shape_img"][0] ,parameter["shape_img"][1] )).to(device).requires_grad_(True)
    dummy_label = torch.randn((parameter["batch_size"], parameter["num_classes"])).to(device).requires_grad_(True)

    ids= np.random.permutation(len(train_dataset))[:parameter["batch_size"]]


    # prepare attacked batch (orig)
    orig_data = torch.Tensor(parameter["batch_size"],1,parameter["shape_img"][0] ,parameter["shape_img"][1]).to(device)
    orig_label = torch.Tensor(parameter["batch_size"])
    orig_label = orig_label.long().to(device)
    orig_label = orig_label.view(parameter["batch_size"],)

    for index, id in enumerate(ids):
        orig_data[index] = transforms.ToTensor()(train_dataset[id][0])
        orig_label[index] = torch.Tensor([train_dataset[id][1]]).long().to(device).view(1,)


    # calculate orig gradients
    orig_out = model(orig_data)
    criterion = nn.CrossEntropyLoss().to(device)
    y = criterion(orig_out, orig_label)
    gradient = torch.autograd.grad(y, model.parameters())
    gradient = list((_.detach().clone() for _ in gradient))

    #optimizer setup
    if not improved:
        optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=parameter["dlg_lr"])
    else:
        optimizer = torch.optim.LBFGS([dummy_data, ], lr=parameter["dlg_lr"])
        # predict label of dummy gradient
        label_pred = torch.argmin(torch.sum(gradient[-2], dim=-1), dim=-1).detach().reshape((1,)).requires_grad_(
            False)


    for iteration in range(parameter["dlg_iterations"]):
        print("")
        """
        optimizer.step(closure)
        current_loss = closure().item()
        train_iters.append(iters)
        losses.append(current_loss)
        mses.append(torch.mean((dummy_data - gt_data) ** 2).item())
        """



def idlg(model, train_dataset, parameter, device):
    return attack(model, train_dataset, parameter,device, True)


def dlg(model, train_dataset, parameter, device):
    return attack(model, train_dataset, parameter,device, False)

