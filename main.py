import datetime

import torch
import torchvision
from torchvision import datasets

import numpy
from dlg import dlg, idlg, attack
from net import Net1, Net2, weights_init
from train import train
from test import test

if __name__ == '__main__':

    # Parameters
    parameter = {
        "log_interval": 2,
        "lr": 0.1,
        "dataset": "MNIST",
        "batch_size": 8,
        "epochs": 1,
        "max_epoch_size": 1000,
        "test_size": 1000,
        "seed": 0,
        "result_path": "results/{}/".format(str(datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S"))),
        "dlg_lr": 1,
        "dlg_iterations": 20,
        "model": 1,
        "prediction": "simplified",
        "improved": True
        #"dlg_convergence": 0.00000001
    }

    # Check CUDA
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Torch successfully connected to CUDA device: {}".format(torch.cuda.current_device()))
    else:
        print("Error: Torch can't connect to CUDA")
        exit()

    # Setting Torch Seed
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
    if parameter["model"]==1:
        model = Net1(parameter)
        model.apply(weights_init)
    elif parameter["model"]==2:
        model = Net2(parameter)


    model = model.to(device)

    ######################################################

    # pretrain model
    #train(model, train_dataset, parameter, device)
    #test(model, test_dataset, parameter, device)

    # dlg
    dlg_result = attack(model, train_dataset, parameter, device, parameter["improved"])
    dlg_result.show()

    #####################################################

    print("Run finished")
