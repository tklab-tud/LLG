import datetime
import json
import os
import time

import numpy as np
import torch

from Dataloader import Dataloader
from Dlg import Dlg
from Predictor import Predictor
from Result import Result
from model import Net1, weights_init
from train import train, test


class Setting:
    def __init__(self, dataloader, **kwargs):

        # Parameter
        self.parameter = {}
        self.parameter = {}
        self.restore_default_parameter()
        self.dataloader = dataloader
        self.predictor = None
        self.model = None
        self.result = None
        self.dlg = None
        self.device = None

        self.check_cuda()

        self.configure(**kwargs)




    # The setting can be changed during a run with configure()
    def configure(self, **kwargs):
        old_parameter = self.parameter.copy()
        self.update_parameter(**kwargs)

        # Invalidate old results
        self.predictor = Predictor(self)
        self.dlg = Dlg(self)
        self.result = Result(self)
        self.parameter["orig_data"] = []
        self.parameter["orig_label"] = []


        # Some arguments need some special treatment, everything else will be just an parameter update
        for key, value in kwargs.items():
            if key == "dataset" and value != old_parameter["dataset"]:
                if value == "MNIST":
                    self.parameter["shape_img"] = (28, 28)
                    self.parameter["num_classes"] = 10
                    self.parameter["channel"] = 1
                    self.parameter["hidden"] = 588
                elif value == 'CIFAR':
                    self.parameter["shape_img"] = (32, 32)
                    self.parameter["num_classes"] = 100
                    self.parameter["channel"] = 3
                    self.parameter["hidden"] = 768
                elif value == 'CIFAR-grey':
                    self.parameter["shape_img"] = (32, 32)
                    self.parameter["num_classes"] = 100
                    self.parameter["channel"] = 1
                    self.parameter["hidden"] = 768
                else:
                    print("Unsupported dataset '" + value + "'")
                    exit()

                self.model = self.load_model()
                self.dlg = Dlg(self)
                self.predictor = Predictor(self)

            if key == "result_path" and value is None:
                self.parameter["result_path"] = old_parameter["result_path"]

            if key == "model" and value != old_parameter["model"]:
                self.model = self.load_model()

        if self.model is None:
            self.model = self.load_model()




    def update_parameter(self, **kwargs):
        # update existing parameters
        for key, value in kwargs.items():
            if key == "batch_size" and value <= 0:
                exit("Batch_size must be > 0")

            if self.parameter.__contains__(key):
                self.parameter[key] = value
            else:
                exit("Unknown Parameter: " + key)

    def restore_default_parameter(self):
        self.parameter = {
            # General settings
            "dataset": "MNIST",
            "targets": [],
            "batch_size": 2,
            "model": 1,
            "log_interval": 10,
            "result_path": "results/{}/".format(str(datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S"))),
            "run_name": "",
            "set_size": None,

            # Attack settings
            "dlg_lr": 1,
            "dlg_iterations": 50,
            "version": "v2",
            "orig_data": [],
            "orig_label": [],
            "shape_img": (28, 28),
            "num_classes": 10,
            "channel": 1,
            "hidden": 588,
            "hidden2": 9216,

            # Train settings
            "test_size": 1000,
            "train_size": 1000,
            "train_lr": 0.1,
            "test_loss": -1,
            "test_acc": 0
        }

    def check_cuda(self):
        # Check CUDA
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def load_model(self):
        if self.parameter["model"] == 1:
            model = Net1(self.parameter)
            model.apply(weights_init)
        else:
            exit("No model found for: ", self.parameter["model"])

        return model.to(self.device)


    def attack(self, extent):
        self.dlg.victim_side()
        if extent == "victim_side":
            return

        if not self.parameter["version"] == "dlg":
            self.predictor.predict()

        if extent == "predict":
            return

        self.dlg.reconstruct()

    def copy(self):
        kwargs = {}
        kwargs.update(**self.parameter)
        kwargs.pop("orig_data", None)
        kwargs.pop("orig_label", None)
        kwargs.pop("targets", None)
        tmp_setting = Setting(self.dataloader, **kwargs)
        return tmp_setting

    def get_backup(self):

        tmp_parameter = self.parameter.copy()
        tmp_parameter.pop("orig_data", None)
        tmp_parameter["orig_label"] = self.parameter["orig_label"].cpu().detach().numpy().tolist()

        adjusted_gradients = self.dlg.gradient[-2].sum(-1) - self.predictor.offset
        adjusted_gradients = adjusted_gradients.cpu().detach().numpy().tolist()
        original_gradients = self.dlg.gradient[-2].sum(-1).cpu().detach().numpy().tolist()

        data_dic = {
            "parameter": tmp_parameter,
            "attack_results": {
                "losses": self.result.losses,
                "mses": self.result.mses.tolist(),
                # "snapshots": list(map(lambda x: x.tolist(), self.result.snapshots))
            },
            "prediction_results": {
                "correct": self.predictor.correct,
                "false": self.predictor.false,
                "accuracy": self.predictor.acc,
                "prediction": self.predictor.prediction,
                "impact": self.predictor.impact,
                "offset": self.predictor.offset.cpu().detach().numpy().tolist(),
                "original_gradients": original_gradients,
                "adjusted_gradients": adjusted_gradients}
        }

        return data_dic

    def reinit_weights(self):
        weights_init(self.model)

    def train(self, train_size):
        train(self, train_size)
        self.parameter["test_loss"], self.parameter["test_acc"] = test(self)
        print("Training finished, loss = {}, acc = {}".format(self.parameter["test_loss"], self.parameter["test_acc"]))
