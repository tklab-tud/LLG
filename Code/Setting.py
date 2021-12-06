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
from model import *
from train import train, test, train_federated
import copy

import defenses as defs


class Setting:
    def __init__(self, dataloader, **kwargs):
        # Parameter
        self.parameter = {}
        self.restore_default_parameter()
        self.dataloader = dataloader
        self.predictor = None
        self.model = None
        self.result = None
        self.dlg = None
        self.device = None
        self.model_backup = None
        self.defenses = None
        self.check_cuda()

        self.configure(**kwargs)

    # The setting can be changed during a run with configure()
    def configure(self, **kwargs):
        old_parameter = self.parameter.copy()
        self.update_parameter(**kwargs)
        self.check_cuda()

        # Invalidate old results
        self.predictor = Predictor(self)
        self.dlg = Dlg(self)
        self.result = Result(self)
        self.defenses = defs.Defenses(self)
        self.parameter["orig_data"] = [[]]
        self.parameter["orig_label"] = [[]]

        # Some arguments need some special treatment, everything else will be just an parameter update
        for key, value in kwargs.items():
            if key == "dataset" and value != old_parameter["dataset"]:
                if value == "MNIST":
                    self.parameter["shape_img"] = (28, 28)
                    self.parameter["num_classes"] = 10
                    self.parameter["channel"] = 1
                    self.parameter["hidden"] = 588
                elif value == "EMNIST":
                    self.parameter["shape_img"] = (28, 28)
                    self.parameter["num_classes"] = 62
                    self.parameter["channel"] = 1
                    self.parameter["hidden"] = 588
                elif value == "FEMNIST":
                    self.parameter["shape_img"] = (28, 28)
                    self.parameter["num_classes"] = 62
                    self.parameter["channel"] = 1
                    self.parameter["hidden"] = 588 # 7 * 7 * 64
                elif value == "FEMNIST-digits":
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
                elif value == 'CELEB-A':
                    self.parameter["shape_img"] = (218, 178)
                    self.parameter["num_classes"] = 10177
                    self.parameter["channel"] = 3
                    self.parameter["hidden"] = 29700
                elif value == 'CELEB-A-male':
                    self.parameter["shape_img"] = (218, 178)
                    self.parameter["num_classes"] = 2
                    self.parameter["channel"] = 3
                    self.parameter["hidden"] = 29700
                elif value == 'CELEB-A-hair':
                    self.parameter["shape_img"] = (218, 178)
                    self.parameter["num_classes"] = 5
                    self.parameter["channel"] = 3
                    self.parameter["hidden"] = 29700
                elif value == 'SVHN':
                    self.parameter["shape_img"] = (32, 32)
                    self.parameter["num_classes"] = 10
                    self.parameter["channel"] = 3
                    self.parameter["hidden"] = 12*8*8
                elif value in ['DUMMY-ONE', 'DUMMY-ZERO', 'DUMMY-RANDOM']:
                    continue
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
            "local_iterations": 1,
            "model": "LeNet",
            "log_interval": 10,
            "result_path": "results/{}/".format(str(datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S"))),
            "run_name": "",
            "set_size": None,
            "dummy" : False,
            "local_training": False,

            # device
            "cuda_id": 0,

            # Differential Privacy settings
            "differential_privacy": False,
            "alphas": [],
            "noise_multiplier": 1.0,
            # clip
            "max_norm": None,
            "noise_type": "gauss",

            # Defenses
            "dropout": 0.0,
            "compression": False,
            "threshold": 0.1,

            # Attack settings
            "dlg_lr": 1,
            "dlg_iterations": 50,
            "version": "v2",
            "orig_data": [[]],
            "orig_label": [[]],
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
            "test_acc": 0,

            # federated learning
            "federated": False,
            "num_users": 1
        }

    def check_cuda(self, verbose=False):
        # Check CUDA
        if torch.cuda.is_available():
            self.device = torch.device("cuda:{}".format(self.parameter["cuda_id"]))
            if verbose:
                print("using cuda gpu: " + str(torch.cuda.current_device()))
        else:
            self.device = torch.device("cpu")
            if verbose:
                print("cuda unavailable using cpu")

    def load_model(self):
        if self.parameter["model"] == "LeNet":
            model = Net1(self.parameter)
        elif self.parameter["model"] == "LeNetNew":
            model = LeNet(self.parameter)
        elif self.parameter["model"] == "NewNewLeNet":
            model = NewNewLeNet(self.parameter)
        elif self.parameter["model"] == "ResNet":
            model = resnet20(self.parameter)
        elif self.parameter["model"] == "MLP":
            len_in = self.parameter["channel"]
            for x in self.parameter["shape_img"]:
                len_in *= x
            model = MLP(len_in, self.parameter["hidden"], self.parameter["num_classes"], dropout=self.parameter["dropout"])
        else:
            exit("No model found for: ", self.parameter["model"])

        model.apply(weights_init)
        return model.to(self.device)

    def backup_model(self):
        self.model_backup = copy.deepcopy(self.model)

    def restore_model(self):
        self.model = self.model_backup

    def attack(self, extent, keep: bool=False):
        # Creating a model backup
        self.backup_model()

        # Victim side gradients will be calculated in any run
        self.dlg.victim_side()

        if not keep:
            self.restore_model()

        # End after this if extent is victim side
        if extent == "victim_side":
            return

        # In case of v1, v2, v3 ... do the prediction; dlg's prediction is done at reconstruction step
        if not self.parameter["version"] == "dlg":
            self.predictor.predict()
            # If we only want prediction stop here
            if extent == "predict":
                return

        # In case of dlg prediction or full reconstruction, the image reconstruction is needed.
        self.dlg.reconstruct()

    def copy(self):
        kwargs = {}
        kwargs.update(**self.parameter)
        kwargs.pop("orig_data", None)
        kwargs.pop("orig_label", None)
        kwargs.pop("targets", None)
        tmp_setting = Setting(self.dataloader, **kwargs)
        return tmp_setting

    def get_backup(self, store_individual_gradients=False):
        tmp_parameter = self.parameter.copy()
        tmp_parameter.pop("orig_data", None)
        tmp_parameter["orig_label"] = list(x.cpu().detach().numpy().tolist() for x in self.parameter["orig_label"])

        adjusted_gradients = self.dlg.gradient[-2].sum(-1) - self.predictor.offset
        adjusted_gradients = adjusted_gradients.cpu().detach().numpy().tolist()
        original_gradients = self.dlg.gradient[-2].sum(-1).cpu().detach().numpy().tolist()

        data_dic = {
            "parameter": tmp_parameter,
            "attack_results": {
                "losses": self.result.losses,
                "mses": self.result.mses.tolist(),
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

        if store_individual_gradients:
            data_dic["prediction_results"].update({"individual_gradients": self.dlg.gradient[-2].cpu().detach().numpy().tolist()})

        return data_dic

    def reinit_weights(self):
        weights_init(self.model)

    def train(self, train_size, batch=None, verbose=False, victim=False):
        if self.parameter["federated"] and not victim:
            train_federated(self)
            self.restore_model()
        else:
            train(self, train_size, batch)

        if verbose:
            print("Training finished, loss = {}, acc = {}".format(self.parameter["test_loss"], self.parameter["test_acc"]))
