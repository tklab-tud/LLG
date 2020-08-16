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
from net import Net1, weights_init
from train import train, test


class Setting:
    def __init__(self, dataloader="new", **kwargs):
        # Parameter
        self.parameter = {}
        self.restore_default_parameter()
        self.update_parameter(**kwargs)

        # Cuda
        self.device = None
        self.check_cuda()

        # Predictor
        self.predictor = Predictor(self)

        if dataloader == "new":
            self.dataloader = Dataloader(self, self.parameter["dataset"])
        else:
            self.dataloader = dataloader

        # Data
        if self.dataloader is not None:
            self.load_data()

        # Model
        self.model = self.load_model()

        # Results
        self.result = Result(self)

        # DLG
        self.dlg = Dlg(self)

    # The setting can be changed during a run with configure()
    # Some arguments need some special treatments, everything else will be just an parameter update
    def configure(self, **kwargs):
        self.update_parameter(**kwargs)

        # Reset old results
        self.predictor = Predictor(self)
        self.dlg = Dlg(self)
        self.result = Result(self)
        self.parameter["orig_data"] = []
        self.parameter["orig_label"] = []

        for key, value in kwargs.items():
            if key == "dataset":
                self.dataloader = Dataloader(self, value)
                if value == "MNIST":
                    self.parameter["shape_img"] = (28, 28)
                    self.parameter["num_classes"] = 10
                    self.parameter["channel"] = 1
                    self.parameter["hidden"] = 588
                    self.parameter["hidden2"] = 9216
                elif value == 'CIFAR':
                    self.parameter["shape_img"] = (32, 32)
                    self.parameter["num_classes"] = 100
                    self.parameter["channel"] = 3
                    self.parameter["hidden"] = 768
                    self.parameter["hidden2"] = 12544
                # changing dataset requires new model and new data
                self.model = self.load_model()
                self.parameter["orig_data"], self.parameter["orig_label"] = \
                    self.dataloader.get_batch(self)
            elif key == "model":
                self.model = self.load_model()
            elif key == "targets" or key == "batch_size":
                self.parameter["orig_data"], self.parameter["orig_label"] = \
                    self.dataloader.get_batch(self)

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
            "log_interval": 5,
            "result_path": "results/{}/".format(str(datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S"))),
            "run_name": "",

            # Attack settings
            "dlg_lr": 1,
            "dlg_iterations": 50,
            "prediction": "v2",
            "orig_data": [],
            "orig_label": [],
            "shape_img": (28, 28),
            "num_classes": 10,
            "channel": 1,
            "hidden": 588,
            "hidden2": 9216,
            "orig_gradient": None,
            "adjusted_gradient": None,

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

        return model.to(self.device)

    def load_data(self):
        self.parameter["orig_data"], self.parameter["orig_label"] = \
            self.dataloader.get_batch(self)

    def attack(self):
        self.dlg.attack()

    def predict(self, verbose=False):
        self.predictor.predict()
        if verbose:
            self.predictor.print_prediction()
        return self.predictor.prediction

    def copy(self):
        kwargs = {}
        kwargs.update(**self.parameter)
        kwargs.__delitem__("orig_data")
        kwargs.__delitem__("orig_label")
        kwargs.__delitem__("targets")
        tmp_setting = Setting(
            dataloader=self.dataloader,
            **kwargs
        )
        return tmp_setting

    def get_backup(self):

        tmp_parameter = self.parameter.copy()
        tmp_parameter.__delitem__("orig_data")
        # tmp_parameter.__delitem__("orig_label")
        tmp_parameter["orig_label"] = self.parameter["orig_label"].cpu().detach().numpy().tolist()

        adjusted_gradients = self.dlg.gradient[-2].sum(-1) - self.predictor.offset
        adjusted_gradients = adjusted_gradients.cpu().detach().numpy().tolist()


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
                "original_gradients": self.dlg.gradient[-2].sum(-1).cpu().detach().numpy().tolist(),
                "adjusted_gradients": adjusted_gradients}
        }

        return data_dic

    def reinit_weights(self):
        weights_init(self.model)

    def train(self, train_size):
        print("Training started")
        train(self, train_size)
        self.parameter["test_loss"], self.parameter["test_acc"] = test(self)
        print("Training finished, loss = {}".format(self.parameter["test_loss"]))
