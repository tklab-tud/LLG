import numpy as np
import torch
import torch.nn as nn
import torchvision

from Result import Result
from Prediction import Predictor


class Dlg:
    def __init__(self, setting):
        self.setting = setting
        self.gradient = None
        self.criterion = nn.CrossEntropyLoss().to(setting.device)

        self.orig_data = torch.Tensor(setting.parameter["batch_size"], setting.parameter["channel"], setting.parameter["shape_img"][0],
                                      setting.parameter["shape_img"][1]).to(setting.device)
        self.orig_label = torch.Tensor(setting.parameter["batch_size"]).long().to(setting.device).view(setting.parameter["batch_size"], )
        self.dummy_data = torch.randn(
            (setting.parameter["batch_size"], setting.parameter["channel"], setting.parameter["shape_img"][0], setting.parameter["shape_img"][1])).to(
            setting.device).requires_grad_(True)
        self.dummy_label = torch.randn((setting.parameter["batch_size"], setting.parameter["num_classes"])).to(setting.device).requires_grad_(True)

        self.setting.fill_targets()



    def victim_side(self):
        # abbreviations
        parameter = self.setting.parameter
        model = self.setting.model

        # load the data
        for i_i, id in enumerate(self.setting.ids[:parameter["batch_size"]]):
            self.orig_data[i_i] = self.setting.train_dataset[id][0]
            self.orig_label[i_i] = self.setting.train_dataset[id][1]

        # calculate orig gradients
        orig_out = model(self.orig_data)
        y = self.criterion(orig_out, self.orig_label)
        grad = torch.autograd.grad(y, model.parameters())
        self.gradient = list((_.detach().clone() for _ in grad))



    def attack(self):
        # abbreviations
        parameter = self.setting.parameter
        device = self.setting.device
        model = self.setting.model

        self.victim_side()

        # optimizer setup
        if not parameter["improved"]:
            optimizer = torch.optim.LBFGS([self.dummy_data, self.dummy_label], lr=parameter["dlg_lr"])
        else:
            optimizer = torch.optim.LBFGS([self.dummy_data, ], lr=parameter["dlg_lr"])
            # predict label of dummy gradient
            self.setting.predictor = Predictor(self.setting)

        # Prepare Result Object
        res = Result(self.setting)
        res.set_origin(self.orig_data.cpu().detach().numpy(), self.orig_label.cpu().detach().numpy())

        for iteration in range(parameter["dlg_iterations"]):

            # clears gradients, computes loss, returns loss
            def closure():
                optimizer.zero_grad()
                dummy_pred = model(self.dummy_data)
                if not parameter["improved"]:
                    dummy_loss = - torch.mean(
                        torch.sum(torch.softmax(self.dummy_label, -1) * torch.log(torch.softmax(dummy_pred, -1)), dim=-1))
                else:
                    dummy_loss = self.criterion(dummy_pred, torch.Tensor(self.setting.predict()).long().to(device))

                dummy_gradient = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

                grad_diff = torch.Tensor([0]).to(device)
                for gx, gy in zip(dummy_gradient, self.gradient):
                    grad_diff += ((gx - gy) ** 2).sum()
                grad_diff.backward()
                res.add_loss(grad_diff.item())
                return grad_diff

            optimizer.step(closure)
            current_loss = closure().item()

            if iteration % parameter["log_interval"] == 0:
                print('{: 3d} loss = {:1.8f}'.format(iteration, current_loss))
                res.add_snapshot(self.dummy_data.cpu().detach().numpy())

        self.setting.result = res
