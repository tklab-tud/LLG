import datetime

import torch
from torchvision import datasets

from dlg import dlg
from net import Net
from train import train

if __name__ == '__main__':
    """
    # get the arguments
    args = vars(ap.parse_args())
    argstr = "arg"
    for arg in args:
        argstr += ("_" + str(args[arg]))
    """

    # Parameters

    parameter = {
        "log_interval": 2,
        "lr": 0.01,
        "dataset": "MNIST",
        "batch_size": 5,
        "epochs": 1,
        "max_epoch_size": 300,
        "seed": 1,
        "result_path": "results/{}/".format(str(datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S"))),
        "dlg_lr": 0.1,
        "dlg_iterations": 24,
        "dlg_start_at": 100,
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

    # Initialising datasets
    if parameter["dataset"] == "MNIST":
        parameter["shape_img"] = (28, 28)
        parameter["num_classes"] = 10
        parameter["channel"] = 1
        train_dataset = datasets.MNIST('./datasets', train=True, download=True)
        test_dataset = datasets.MNIST('./datasets', train=False, download=True)
        channel = 1
    elif parameter["dataset"] == 'CIFAR':
        parameter["shape_img"] = (32, 32)
        parameter["num_classes"] = 100
        parameter["channel"] = 3
        train_dataset = datasets.CIFAR100('./datasets', train=True, download=True)
        test_dataset = datasets.CIFAR100('./datasets', train=False, download=True)
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
    model = Net(parameter)
    model.weights_init()
    model = model.to(device)

    ######################################################

    # train x epochs
    train(model, train_dataset, parameter, device)

    # dlg
    dlg_result = dlg(model, train_dataset, parameter, device)

    dlg_result.show()

    #####################################################

    print("Run finished")
