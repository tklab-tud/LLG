import torch
import torch.nn as nn
from Result import Result

import aDPtorch.privacy_engine_xl as adp


class Dlg:
    def __init__(self, setting):
        self.setting = setting
        self.criterion = nn.CrossEntropyLoss().to(setting.device)
        self.gradient = None
        self.dummy_data = None
        self.dummy_label = None

    def victim_side(self):
         # calculate orig gradients
        self.setting.parameter["orig_data"], self.setting.parameter["orig_label"] = \
            self.setting.dataloader.get_batch(self.setting.parameter["dataset"], self.setting.parameter["targets"], self.setting.parameter["batch_size"])

        self.setting.parameter["orig_data"] = self.setting.parameter["orig_data"].to(self.setting.device)
        self.setting.parameter["orig_label"] = self.setting.parameter["orig_label"].to(self.setting.device)

        orig_out = self.setting.model(self.setting.parameter["orig_data"])
        y = self.criterion(orig_out, self.setting.parameter["orig_label"])
        grad = torch.autograd.grad(y, self.setting.model.parameters())

        # Noisy Gradients
        if self.setting.parameter["differential_privacy"]:
            adp.apply_noise(grad, self.setting.parameter["batch_size"], self.setting.parameter["max_norm"], self.setting.parameter["noise_multiplier"], self.setting.parameter["noise_type"], self.setting.device)

        # Gradient Compression
        if self.setting.parameter["compression"]:
            values = torch.sum(grad[-2], dim=-1).clone()
            magnitudes = [torch.abs(value) for value in values]
            magnitudes_sorted = sorted(magnitudes)

            threshold = int(len(magnitudes_sorted) * self.setting.parameter["threshold"]) - 1
            max_magnitude = magnitudes_sorted[threshold]
            max_mag_count = 1
            first_idx = threshold
            for i, mag in enumerate(magnitudes_sorted):
                if mag == max_magnitude:
                    first_idx = i
            max_mag_count = threshold - first_idx

            count = 0
            for magnitude, tens in zip(magnitudes, grad):
                if magnitude < max_magnitude:
                    tens.zero_()
                elif magnitude == max_magnitude:
                    if count <= max_mag_count:
                        tens.zero_()
                    else:
                        continue
                    count += 1
                elif magnitude > max_magnitude:
                    continue

        self.gradient = list((_.detach().clone() for _ in grad))


    def reconstruct(self):
        # abbreviations
        parameter = self.setting.parameter
        device = self.setting.device
        model = self.setting.model
        setting = self.setting


        self.dummy_data = torch.randn(
            (setting.parameter["batch_size"], setting.parameter["channel"], setting.parameter["shape_img"][0],
             setting.parameter["shape_img"][1])).to(
            setting.device).requires_grad_(True)
        self.dummy_label = torch.randn((setting.parameter["batch_size"], setting.parameter["num_classes"])).to(
            setting.device).requires_grad_(True)
        self.dummy_pred = None

        # optimizer setup
        if parameter["version"] == "dlg":
            optimizer = torch.optim.LBFGS([self.dummy_data, self.dummy_label], lr=parameter["dlg_lr"])
        else:
            optimizer = torch.optim.LBFGS([self.dummy_data, ], lr=parameter["dlg_lr"])
            # predict label of dummy gradient
            pred = torch.Tensor(self.setting.predictor.prediction).long().to(device).reshape(
                    (parameter["batch_size"],)).requires_grad_(False)

        # Prepare Result Object
        res = Result(self.setting)

        for iteration in range(parameter["dlg_iterations"]):

            # clears gradients, computes loss, returns loss
            def closure():
                optimizer.zero_grad()
                self.dummy_pred = model(self.dummy_data)
                if parameter["version"] == "dlg":
                    dummy_loss = - torch.mean(
                        torch.sum(torch.softmax(self.dummy_label, -1) * torch.log(torch.softmax(self.dummy_pred, -1)),
                                  dim=-1))
                else:
                    dummy_loss = self.criterion(self.dummy_pred, pred)

                dummy_gradient = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

                grad_diff = torch.Tensor([0]).to(device)
                for gx, gy in zip(dummy_gradient, self.gradient):
                    grad_diff += ((gx - gy) ** 2).sum()
                grad_diff.backward()
                res.add_loss(grad_diff.item())
                return grad_diff

            optimizer.step(closure)

            if iteration % parameter["log_interval"] == 0:
                res.add_snapshot(self.dummy_data.cpu().detach().numpy())

        if self.setting.parameter["version"] == "dlg":
            self.setting.predictor.prediction = [self.dummy_label[x].argmax().item() for x in range(parameter["batch_size"])]
            self.setting.predictor.update_accuracy()
            #res.update_figures()


        self.setting.result = res

