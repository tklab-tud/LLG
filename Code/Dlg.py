import copy
import torch
import torch.nn as nn
from Result import Result


class Dlg:
    def __init__(self, setting):
        self.setting = setting
        self.defenses = setting.defenses
        self.criterion = nn.CrossEntropyLoss().to(setting.device)
        self.gradient = None
        self.dummy_data = None
        self.dummy_label = None
        self.seperated_gradients = []

    def victim_side(self):
        para = self.setting.parameter

        para["orig_data"] = [None]*para["local_iterations"]
        para["orig_label"] = [None]*para["local_iterations"]

        self.seperated_gradients = []


        # calculate orig gradients
        for i in range(para["local_iterations"]):
            para["orig_data"][i], para["orig_label"][i] = \
                self.setting.dataloader.get_batch(para["dataset"], para["targets"], para["batch_size"])
            para["orig_data"][i] = para["orig_data"][i].to(self.setting.device)
            para["orig_label"][i] = para["orig_label"][i].to(self.setting.device)
            orig_out = self.setting.model(para["orig_data"][i])
            y = self.criterion(orig_out, para["orig_label"][i])
            grad = torch.autograd.grad(y, self.setting.model.parameters())

            # local train iteration
            if self.setting.parameter["local_training"]:
                self.setting.train(1, [para["orig_data"][i], para["orig_label"][i]], victim=True)

            self.seperated_gradients.append(list((_.detach().clone() for _ in grad)))

        # Copy the structure of a grad, but make it zeroes
        aggregated = list(x.zero_() for x in grad)

        # iterate over the gradients for each local iteration
        for grad in self.seperated_gradients:
            # there iterate through the gradients and add to the aggregator
            for i_g,g in enumerate(grad):
                aggregated[i_g] = torch.add(aggregated[i_g], g)

        self.defenses.apply(aggregated, para["num_users"]-1)

        self.gradient = list(torch.div(x, 1) for x in aggregated)
        #Might also take the average instead of the sum
        #self.gradient = list(torch.div(x, para["local_iterations"]) for x in aggregated)

        if para["differential_privacy"] or para["compression"]:
            self.defenses.inject(self.seperated_gradients, aggregated, self.setting.model)

    def reconstruct(self):
        # abbreviations
        parameter = self.setting.parameter
        device = self.setting.device
        model = self.setting.model
        setting = self.setting


        self.dummy_data = torch.randn(
            (parameter["batch_size"]*parameter["local_iterations"], setting.parameter["channel"], setting.parameter["shape_img"][0],
             setting.parameter["shape_img"][1])).to(
            setting.device).requires_grad_(True)
        self.dummy_label = torch.randn((parameter["batch_size"]*parameter["local_iterations"], setting.parameter["num_classes"])).to(
            setting.device).requires_grad_(True)
        self.dummy_pred = None

        # optimizer setup
        if parameter["version"] == "dlg":
            optimizer = torch.optim.LBFGS([self.dummy_data, self.dummy_label], lr=parameter["dlg_lr"])
        else:
            optimizer = torch.optim.LBFGS([self.dummy_data, ], lr=parameter["dlg_lr"])
            # predict label of dummy gradient
            pred = torch.Tensor(self.setting.predictor.prediction).long().to(device).reshape(
                    (parameter["batch_size"]*parameter["local_iterations"],)).requires_grad_(False)

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
            self.setting.predictor.prediction = [self.dummy_label[x].argmax().item() for x in range(parameter["batch_size"]*parameter["local_iterations"])]
            self.setting.predictor.update_accuracy()
            #res.update_figures(),


        self.setting.result = res

