import datetime

import numpy
import torch
import torchvision
from torchvision import datasets

from dlg import attack
from net import Net1, Net2, weights_init
from test import test
from train import train

if __name__ == '__main__':

    # Parameters
    parameter = {
        # General settings
        "dataset": "MNIST",
        "batch_size": 4,
        "model": 1,
        "log_interval": 2,
        "use_seed": True,
        "seed": 2,
        "result_path": "results/{}/".format(str(datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S"))),

        # Attack settings
        "dlg_lr": 0.2,
        "dlg_iterations": 20,
        "prediction": "v1",
        "improved": True,

        # Pretrain settings
        "pretrain": False,
        "lr": 0.1,
        "epochs": 1,
        "max_epoch_size": 1000,
        "test_size": 1000,
    }

    # Check CUDA
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Torch successfully connected to CUDA device: {}".format(torch.cuda.current_device()))
    else:
        print("Error: Torch can't connect to CUDA")
        exit()

    # Setting Torch Seed
    if parameter["use_seed"]:
        torch.manual_seed(parameter["seed"])
        numpy.random.seed(parameter["seed"])

    # Initialising datasets
    tt = torchvision.transforms.ToTensor()
    if parameter["dataset"] == "MNIST":
        parameter["shape_img"] = (28, 28)
        parameter["num_classes"] = 10
        parameter["channel"] = 1
        parameter["hidden"] = 588
        parameter["hidden2"] = 9216
        train_dataset = datasets.MNIST('./datasets', train=True, download=True, transform=tt)
        test_dataset = datasets.MNIST('./datasets', train=False, download=True, transform=tt)
        channel = 1
    elif parameter["dataset"] == 'CIFAR':
        parameter["shape_img"] = (32, 32)
        parameter["num_classes"] = 100
        parameter["channel"] = 3
        parameter["hidden"] = 768
        parameter["hidden2"] = 12544
        train_dataset = datasets.CIFAR100('./datasets', train=True, download=True, transform=tt)
        test_dataset = datasets.CIFAR100('./datasets', train=False, download=True, transform=tt)
    else:
        print("Unsupported dataset '" + parameter["dataset"] + "'")
        exit()

    # Setting max_epoch_size; 0 will set size to set-length
    if parameter["max_epoch_size"] == 0:
        parameter["max_epoch_size"] = len(train_dataset) / parameter["batch_size"]

    # Log Parameters
    for entry in parameter:
        print("{}: {}".format(entry, parameter[entry]))

    # prepare the model
    if parameter["model"] == 1:
        model = Net1(parameter)
        model.apply(weights_init)
    elif parameter["model"] == 2:
        model = Net2(parameter)

    model = model.to(device)

    ######################################################

    # pretrain model
    if parameter["pretrain"]:
        train(model, train_dataset, parameter, device)
        test(model, test_dataset, parameter, device)

    # filling the result with snapshots and loss-values
    dlg_result = attack(model, train_dataset, parameter, device, parameter["improved"])
    # calculate image, calculate mses, fix order
    dlg_result.process()
    # show composed image
    dlg_result.show_composed_image()
    # store composed image
    dlg_result.store_composed_image()
    # store seperate images
    dlg_result.store_separate_images()
    # store raw data
    dlg_result.store_data()

    #####################################################

    print("Run finished")
