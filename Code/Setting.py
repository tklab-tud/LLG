import datetime
import json
import time
from tkinter import Tk
from tkinter.filedialog import askopenfilenames

import numpy as np
import torch
import torchvision
from torchvision import datasets

from Dlg import Dlg
from Predictor import Predictor
from Result import Result
from net import Net1, Net2, weights_init
from test import test
from train import train


class Setting:
    def __init__(self, **kwargs):
        self.target = []
        self.ids = []
        self.orig_data = None
        self.orig_label = None

        self.device = None
        self.model = None
        self.dlg = None
        self.check_cuda()
        self.predictor = Predictor(self)

        self.parameter = {}

        self.train_dataset = None
        self.test_dataset = None

        self.restore_default_parameter()
        self.update_parameter(**kwargs)

        self.reset_seeds()
        self.load_dataset()
        self.load_model()

        self.result = Result(self)

        self.configure(**kwargs)

        if len(self.target) == 0:
            self.fill_ids()
            self.fix_targets()

    def configure(self, **kwargs):
        for key, value in kwargs.items():
            if key == "target":
                self.target = list(value)
                self.fill_targets()
                self.fix_ids()
            elif key == "ids":
                self.ids = list(value)
                self.fill_ids()
                self.fix_targets()
            elif key == "dataset":
                self.load_dataset()
            elif key == "model":
                self.load_model()
            elif key == "use_seed" or key == "seed":
                self.reset_seeds()
            elif key == "max_epoch_size" and value == 0:
                self.parameter["max_epoch_size"] = len(self.train_dataset) / self.parameter["batch_size"]
            elif key == "batch_size":
                self.parameter["batch_size"] = value
                self.fill_ids()
                self.fix_targets()
                #self.update_parameter(**kwargs)
            else:
                self.update_parameter(**{key: kwargs[key]})

        # Renew Attack with new settings, does not execute yet
        self.dlg = Dlg(self)

    def update_parameter(self, **kwargs):
        # update existing parameters
        for key, value in kwargs.items():
            if self.parameter.__contains__(key):
                self.parameter[key] = value
            elif key != "target" and key != "ids":
                exit("Unknown Parameter: " + key)

    def restore_default_parameter(self):
        self.parameter = {
            # General settings
            "dataset": "MNIST",
            "batch_size": 2,
            "model": 1,
            "log_interval": 5,
            "use_seed": False,
            "seed": 2,
            "result_path": "results/{}/".format(str(datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S"))),
            "run_name": "",

            # Attack settings
            "dlg_lr": 0.1,
            "dlg_iterations": 50,
            "prediction": "v1",
            "improved": True,

            # Pretrain settings
            "lr": 0.01,
            "epochs": 1,
            "max_epoch_size": 1000,
            "test_size": 1000,

            # dataset settings
            "shape_img": (32, 32),
            "num_classes": 100,
            "channel": 3,
            "hidden": 768,
            "hidden2": 12544

        }

    def check_cuda(self):
        # Check CUDA
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            # print("Torch successfully connected to CUDA device: {}".format(torch.cuda.current_device()))
        else:
            print("Error: Torch can't connect to CUDA")
            exit()

    def reset_seeds(self):
        # Setting Torch and numpy Seed
        if self.parameter["use_seed"]:
            torch.manual_seed(self.parameter["seed"])
            np.random.seed(self.parameter["seed"])
        else:
            torch.manual_seed(int(1000 * time.time() % 2 ** 32))
            np.random.seed(int(1000 * time.time() % 2 ** 32))

    def load_dataset(self):
        # Initialising datasets
        tt = torchvision.transforms.ToTensor()
        if self.parameter["dataset"] == "MNIST":
            self.parameter["shape_img"] = (28, 28)
            self.parameter["num_classes"] = 10
            self.parameter["channel"] = 1
            self.parameter["hidden"] = 588
            self.parameter["hidden2"] = 9216
            self.train_dataset = datasets.MNIST('./datasets', train=True, download=True, transform=tt)
            self.test_dataset = datasets.MNIST('./datasets', train=False, download=True, transform=tt)
        elif self.parameter["dataset"] == 'CIFAR':
            self.parameter["shape_img"] = (32, 32)
            self.parameter["num_classes"] = 100
            self.parameter["channel"] = 3
            self.parameter["hidden"] = 768
            self.parameter["hidden2"] = 12544
            self.train_dataset = datasets.CIFAR100('./datasets', train=True, download=True, transform=tt)
            self.test_dataset = datasets.CIFAR100('./datasets', train=False, download=True, transform=tt)
        else:
            print("Unsupported dataset '" + self.parameter["dataset"] + "'")
            exit()

    def load_model(self):
        # prepare the model, reloads it, undoes training
        if self.parameter["model"] == 1:
            self.model = Net1(self.parameter)
            self.model.apply(weights_init)
        elif self.parameter["model"] == 2:
            self.model = Net2(self.parameter)

        self.model = self.model.to(self.device)

    def print_parameter(self):
        # Log Parameters
        for entry in self.parameter:
            print("{}: {}".format(entry, self.parameter[entry]))

    def pretrain(self):
        train(self)
        test(self)

    def attack(self):
        self.dlg = Dlg(self)
        self.dlg.attack()

    def store_everything(self):
        if self.result is not None:
            self.result.store_everything()

    def store_composed_image(self):
        if self.result is not None:
            self.result.store_composed_image()

    def store_separate_images(self):
        if self.result is not None:
            self.result.store_separate_images()

    def store_data(self):
        if self.result is not None:
            self.result.store_data()

    def show_composed_image(self):
        if self.result is not None:
            self.result.show_composed_image()

    def delete(self):
        if self.result is not None:
            self.result.delete()

    def predict(self, verbose=False):
        self.predictor = Predictor(self)
        self.predictor.predict()
        if verbose:
            orig = self.target[:self.parameter["batch_size"]].copy()
            orig.sort()
            print("Orig:", orig)
            print("Pred:", self.predictor.prediction)
            print(
                "Correct: {}\tFalse: {}\tAcc: {}\tUsing: {}".format(self.predictor.correct, self.predictor.false,
                                                                    self.predictor.acc, self.parameter["prediction"]))
        return self.predictor.prediction

    def fix_targets(self):
        self.target = []
        for i in range(len(self.ids)):
            self.target.append(self.train_dataset[self.ids[i]][1])

    def fix_ids(self):
        self.ids = []

        for i in range(len(self.target)):
            random_offset = np.random.randint(0, len(self.train_dataset))
            # searching for a sample with target label
            for i_s in range(len(self.train_dataset)):
                sample_id = (i_s+random_offset)%len(self.train_dataset)
                # does this sample have the right label and was not used before?
                if self.train_dataset[sample_id][1] == self.target[i] and not self.ids.__contains__(sample_id):
                    self.ids.append(sample_id)
                    break

        if len(self.ids) != self.parameter["batch_size"]:
            exit("could not find enough samples in the dataset")

    def fill_targets(self):
        # fill missing targets if underspecified

        for i in range(self.parameter["batch_size"] - len(self.target)):
            self.target.append(np.random.randint(0, self.parameter["num_classes"]))

    def fill_ids(self):
        # fill missing targets if underspecified
        for i in range(self.parameter["batch_size"] - len(self.ids)):
            self.ids.append(np.random.randint(0, len(self.train_dataset)))


    def copy(self):
        kwargs = {}
        kwargs.update(**self.parameter)
        kwargs.update({"ids": self.ids})
        return Setting(**kwargs)

    def load_json(self):
        Tk().withdraw()
        filenames = askopenfilenames(initialdir="./results", defaultextension='.json', filetypes=[('Json', '*.json')])
        setting = []
        for f_name in filenames:
            with open(f_name) as f:
                dump = json.load(f)
                setting.append(Setting(**dump["parameter"]))
                setting[-1].result.losses = dump["losses"]
                setting[-1].result.mses = np.array(dump["mses"])
                setting[-1].target = dump["target"]
                setting[-1].ids = dump["ids"]
                setting[-1].result.snapshots = dump["snapshots"]
                setting[-1].predictor = Predictor(setting[-1])
                setting[-1].predictor.correct = dump["prediction"]["correct"]
                setting[-1].predictor.false = dump["prediction"]["false"]
                setting[-1].predictor.acc = dump["prediction"]["accuracy"]
                setting[-1].predictor.prediction = dump["prediction"]["prediction"]

        return setting

        # fix this
