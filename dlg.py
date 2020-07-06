import numpy as np
import torch
import torch.nn as nn

from Result import Result
from prediction import prediction


def attack(setting):
    # some abbreviations
    parameter = setting.parameter
    targets = setting.target
    device = setting.device
    train_dataset = setting.train_dataset
    model = setting.model

    # prepare attacked batch's data and label on device
    orig_data = torch.Tensor(parameter["batch_size"], parameter["channel"], parameter["shape_img"][0],
                             parameter["shape_img"][1]).to(device)
    orig_label = torch.Tensor(parameter["batch_size"]).long().to(device).view(parameter["batch_size"], )

    # fill missing targets if underspecified
    for i in range(parameter["batch_size"] - len(targets)):
        targets.append(np.random.randint(0, parameter["num_classes"]))
    print("Attack target classes: ", targets[:parameter["batch_size"]])

    # get one example for target class and put it into the attacked batch
    ids = []
    for i_t, target in enumerate(targets[:parameter["batch_size"]]):
        # searching for a sample with target label
        for i_s, sample in enumerate(train_dataset):
            # does this sample have the right label and was not used before?
            if sample[1] == target and not ids.__contains__(i_s):
                ids.append(i_s)
                orig_data[i_t] = sample[0]
                orig_label[i_t] = sample[1]
                break

    # Check if enough sample were found and cut off sample if target list was to long
    if len(ids) < parameter["batch_size"]:
        exit("Could not find enough samples in the dataset")

    print("Attack target ids: ", ids)

    # calculate orig gradients
    orig_out = model(orig_data)
    criterion = nn.CrossEntropyLoss().to(device)
    y = criterion(orig_out, orig_label)
    gradient = torch.autograd.grad(y, model.parameters())
    gradient_list = list((_.detach().clone() for _ in gradient))

    # prepare dummy data
    dummy_data = torch.randn(
        (parameter["batch_size"], parameter["channel"], parameter["shape_img"][0], parameter["shape_img"][1])).to(
        device).requires_grad_(True)
    dummy_label = torch.randn((parameter["batch_size"], parameter["num_classes"])).to(device).requires_grad_(True)

    # optimizer setup
    if not parameter["improved"]:
        optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=parameter["dlg_lr"])
    else:
        optimizer = torch.optim.LBFGS([dummy_data, ], lr=parameter["dlg_lr"])
        # predict label of dummy gradient
        idlg_pred = prediction(parameter, gradient_list, model, orig_data, orig_label, device)

    # Prepare Result Object
    res = Result(parameter)
    res.set_origin(orig_data.cpu().detach().numpy(), orig_label.cpu().detach().numpy())

    for iteration in range(parameter["dlg_iterations"]):

        # clears gradients, computes loss, returns loss
        def closure():
            optimizer.zero_grad()
            dummy_pred = model(dummy_data)
            if not parameter["improved"]:
                dummy_loss = - torch.mean(
                    torch.sum(torch.softmax(dummy_label, -1) * torch.log(torch.softmax(dummy_pred, -1)), dim=-1))
            else:
                dummy_loss = criterion(dummy_pred, idlg_pred)

            dummy_gradient = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

            grad_diff = torch.Tensor([0]).to(device)
            for gx, gy in zip(dummy_gradient, gradient_list):
                grad_diff += ((gx - gy) ** 2).sum()
            grad_diff.backward()
            res.add_loss(grad_diff.item())
            return grad_diff

        optimizer.step(closure)
        current_loss = closure().item()

        if iteration % parameter["log_interval"] == 0:
            print('{: 3d} loss = {:1.8f}'.format(iteration, current_loss))
            res.add_snapshot(dummy_data.cpu().detach().numpy())

    return res
