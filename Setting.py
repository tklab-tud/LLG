import datetime
import time

import numpy as np
import torch
import torchvision
from torchvision import datasets

from Dlg import Dlg
from Prediction import Predictor
from net import Net1, Net2, weights_init
from test import test
from train import train
from Result import Result


class Setting:
    def __init__(self, **kwargs):
        self.initialised = False
        self.target = []
        self.device = None
        self.check_cuda()
        self.train_dataset = None
        self.test_dataset = None
        self.model = None

        self.dlg = None
        self.ids = []
        self.orig_data = None
        self.orig_label = None

        self.parameter = {}
        self.restore_default_parameter()
        self.configure(**kwargs)

        self.reset_seeds()
        self.load_dataset()
        self.load_model()

        self.result = Result(self)
        self.dlg = Dlg(self)
        self.predictor = Predictor(self)
        self.initialised = True

    def configure(self, **kwargs):
        for key, value in kwargs.items():
            if key == "target":
                self.target = value
                self.fix_ids()
            elif key == "ids":
                self.ids = value
                self.fix_targets()
            else:
                self.parameter[key] = value

            # The following parameter changes require additional action
            if key == "dataset":
                self.load_dataset()
            elif key == "model":
                self.load_model()
            elif key == "use_seed" or key == "seed":
                self.reset_seeds()
            elif key == "max_epoch_size" and value == 0:
                self.parameter["max_epoch_size"] = len(self.train_dataset) / self.parameter["batch_size"]

        # Renew Attack with new settings, does not execute yet
        if self.initialised: self.dlg = Dlg(self)

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
        }

    def check_cuda(self):
        # Check CUDA
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            #print("Torch successfully connected to CUDA device: {}".format(torch.cuda.current_device()))
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

    def predict(self, *verbose):
        self.predictor = Predictor(self)
        self.predictor.predict()
        if verbose:
            orig = self.target[:self.parameter["batch_size"]].copy()
            orig.sort()
            #print("Orig:", orig)
            #print("Pred:", self.predictor.prediction)
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
            # searching for a sample with target label
            for i_s, sample in enumerate(self.train_dataset):
                # does this sample have the right label and was not used before?
                if sample[1] == self.target[i] and not self.ids.__contains__(i_s):
                    self.ids.append(i_s)
                    break

    def fill_targets(self):
        # fill missing targets if underspecified
        for i in range(self.parameter["batch_size"] - len(self.target)):
            self.target.append(np.random.randint(0, self.parameter["num_classes"]))

        self.fix_ids()

    def copy(self):
        return Setting(**self.parameter)


