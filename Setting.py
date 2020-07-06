import datetime
import time

import numpy
import torch
import torchvision
from torchvision import datasets

from dlg import attack
from net import Net1, Net2, weights_init
from test import test
from train import train
from prediction import predict_setting


class Setting:
    def __init__(self, **kwargs):
        self.check_cuda()

        self.result = None
        self.target = []
        self.device = None
        self.train_dataset = None
        self.test_dataset = None
        self.model = None

        self.parameter = {}
        self.restore_default_parameter()
        self.configure(**kwargs)

        self.reset_seeds()
        self.load_dataset()
        self.load_model()


    def configure(self, **kwargs):
        for key, value in kwargs.items():
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
            elif key == "target":
                self.target = value

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
            print("Torch successfully connected to CUDA device: {}".format(torch.cuda.current_device()))
        else:
            print("Error: Torch can't connect to CUDA")
            exit()

    def reset_seeds(self):
        # Setting Torch and numpy Seed
        if self.parameter["use_seed"]:
            torch.manual_seed(self.parameter["seed"])
            numpy.random.seed(self.parameter["seed"])
        else:
            torch.manual_seed(int(1000 * time.time() % 2 ** 32))
            numpy.random.seed(int(1000 * time.time() % 2 ** 32))

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
        # prepare the model
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
        self.result = attack(self)

    def store_everything(self):
        self.result.store_everything()

    def store_composed_image(self):
        self.result.store_composed_image()

    def store_separate_images(self):
        self.result.store_separate_images()

    def store_data(self):
        self.result.store_data()

    def show_composed_image(self):
        self.result.show_composed_image()

    def delete(self):
        self.result.delete()

    def predict(self):
        return predict_setting(self)