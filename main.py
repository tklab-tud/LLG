import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision as torchvision
from torchvision import datasets, transforms
import syft as sy
from torchvision import transforms, datasets
import datetime
import logging
from pathlib import Path

import argparse

ap = argparse.ArgumentParser(description="Comparison framework for attacks on federated learning.")
ap.add_argument("-d", "--dataset", required=False, default="MNIST", help="Dataset to use",
                choices=["MNIST", "SVHN", "CIFAR", "ATT"])
ap.add_argument("-n", "--nodes", required=False, default=2, type=int, help="Amount of nodes")
ap.add_argument("-bs", "--batch_size", required=False, default=32, type=int, help="Samples per batch")
ap.add_argument("-es", "--epoch_size", required=False, default=0, type=int,
                help="Batches per epoch, 0 for the complete set")
ap.add_argument("-t", "--test_batch_size", required=False, default=100, type=int, help="Samples for test-batch")
ap.add_argument("-e", "--epochs", required=False, default=3, type=int, help="Amount of epochs")
ap.add_argument("-lr", "--learning_rate", required=False, default=0.5, type=float, help="Learning rate")
ap.add_argument("-a", "--attack", required=False, default="DLG", help="Attacks to perform",
                choices=["GAN", "MI", "UL", "DLG", "iDLG"])
ap.add_argument("-s", "--seed", required=False, default=1, type=int, help="Integer seed for torch")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train():
    model.train()

    for batch_idx, (data, target) in enumerate(federated_train_loader):
        model.send(data.location)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        model.get()

        if (batch_idx % log_interval== 0):
            loss = loss.get()  # <-- NEW: get the loss back
            log('Train Epoch: {}, Batch: {}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx , args["epoch_size"], 100 * batch_idx / args["epoch_size"], loss.item()))

        if batch_idx >= args["epoch_size"]:
            break


def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    log('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def log(str):
    print(str)
    logging.info(str)


if __name__ == '__main__':
    # get the arguments
    args = vars(ap.parse_args())
    argstr = "arg"
    for arg in args:
        argstr += ("_" + str(args[arg]))

    #Hardcoded Parameters
    log_interval = 10

    # logging
    time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    resultpath = "results/{}/{}/".format(argstr, time)
    Path(resultpath).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.DEBUG, filename=resultpath + "log.txt", filemode="w+", format="%(message)s")
    log("Model and Log will be saved to: " + resultpath)

    # Log Parameters
    for arg in args:
        log('{:15}: {:10}'.format(str(arg), str(args[arg])))

    # Creating Federation Nodes
    hook = sy.TorchHook(torch)
    nodes = tuple(sy.VirtualWorker(hook, id=str(id)) for id in range(0, args["nodes"]))

    # Check CUDA
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log("Torch successfully connected to CUDA device: {}".format(torch.cuda.current_device()))
    else:
        log("Error: Torch can't connect to CUDA")
        exit()

    # Setting Torch Seed
    torch.manual_seed(args["seed"])

    # Defining data transformation
    transformation = transforms.Compose([
        # transforms.Resize([32, 32]),
        transforms.ToTensor()  # ,
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Initialising datasets
    train_dataset = datasets.MNIST('./datasets', train=True, download=True, transform=transformation)
    test_dataset = datasets.MNIST('./datasets', train=False, download=True, transform=transformation)

    # Setting epoch_size if 0 (complete set) is choosen
    if args["epoch_size"] == 0:
        args["epoch_size"] = len(train_dataset) / args["batch_size"]
        log("Set epoch size to {}".format(args["epoch_size"]))

    # Initialise Federated Data loader for training
    federated_train_loader = sy.FederatedDataLoader(
        train_dataset.federate(nodes),
        batch_size=args["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    # Initialise Data Loader for testing
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args["test_batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    # Initialise Model
    model = Net().to(device)

    # Initialise Optimizer
    optimizer = optim.SGD(model.parameters(), lr=args["learning_rate"])

    # Training Loop
    for epoch in range(1, args["epochs"] + 1):
        train()

    # Testing
    test()

    # Saving Model
    torch.save(model.state_dict(), "{}model.pt".format(resultpath))
