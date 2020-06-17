import argparse
import datetime
import logging
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import syft as sy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import datasets, transforms

ap = argparse.ArgumentParser(description="Comparison framework for attacks on federated learning.")
ap.add_argument("-d", "--dataset", required=False, default="MNIST", help="Dataset to use",
                choices=["MNIST", "SVHN", "CIFAR", "ATT"])
ap.add_argument("-n", "--nodes", required=False, default=2, type=int, help="Amount of nodes")
ap.add_argument("-bs", "--batch_size", required=False, default=32, type=int, help="Samples per batch")
ap.add_argument("-es", "--epoch_size", required=False, default=100, type=int,
                help="Batches per epoch, 0 for the complete set")
ap.add_argument("-t", "--test_batch_size", required=False, default=100, type=int, help="Samples for test-batch")
ap.add_argument("-e", "--epochs", required=False, default=3, type=int, help="Amount of epochs")
ap.add_argument("-lr", "--learning_rate", required=False, default=0.5, type=float, help="Learning rate")
ap.add_argument("-a", "--attack", required=False, default="DLG", help="Attacks to perform",
                choices=["GAN", "MI", "UL", "DLG", "iDLG"])
ap.add_argument("-s", "--seed", required=False, default=1, type=int, help="Integer seed for torch")


##########################################################################
##########################################################################


class Net(nn.Module):
    def __init__(self, channel, num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(channel, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

##########################################################################

class LeNet(nn.Module):
    def __init__(self, channel=3, hideen=768, num_classes=10):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(hideen, num_classes)
        )

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


##########################################################################
##########################################################################

def weights_init(m):

    try:
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.weight' % m._get_name())
    try:
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.bias' % m._get_name())

##########################################################################

def train():
    model.train()

    for batch_idx, (data, label) in enumerate(federated_train_loader):
        model.send(data.location)
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, label)
        loss.backward()
        optimizer.step()
        model.get()

        if (batch_idx % log_interval== 0):
            loss = loss.get()  # <-- NEW: get the loss back
            log('Train Epoch: {}, Batch: {}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx , args["epoch_size"], 100 * batch_idx / args["epoch_size"], loss.item()))

        if batch_idx >= args["epoch_size"]:
            break

##########################################################################

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


##########################################################################


def dlg():
    train_dataset = 'MNIST'
    root_path = '.'
    data_path = './datasets'
    save_path = './results/iDLG'

    lr = 1.0
    num_dummy = 1
    Iteration = 300
    num_exp = 1000

    device = 'cuda'

    tt = transforms.Compose([transforms.ToTensor()])
    tp = transforms.Compose([transforms.ToPILImage()])

    print(train_dataset, 'root_path:', root_path)
    print(train_dataset, 'data_path:', data_path)
    print(train_dataset, 'save_path:', save_path)

    if not os.path.exists('results'):
        os.mkdir('results')
    if not os.path.exists(save_path):
        os.mkdir(save_path)


    num_classes = 10
    channel = 1
    hidden = 588
    train_dataset = datasets.MNIST(data_path, download=False)


    ''' train DLG and iDLG '''

    model = LeNet(channel=channel, hideen=hidden, num_classes=num_classes)
    model.apply(weights_init)
    model = model.to(device)

    idx = np.random.randint(len(train_dataset))  # id of image trying to recreate
    print("Selected idx: " + str(idx))


    for method in ['DLG', 'iDLG']:
        print("Try to generate %s image" % method)

        criterion = nn.CrossEntropyLoss().to(device)

        # imidx 0 for only one dummy to create
        #for imidx in range(num_dummy):

        # calculate one legit sample
        tmp_datum = tt(train_dataset[idx][0]).float().to(device)
        data =  tmp_datum.view(1, *tmp_datum.size())
        label = torch.Tensor([train_dataset[idx][1]]).long().to(device).view(1, )
        out = model(data)
        y = criterion(out, label)
        dy_dx = torch.autograd.grad(y, model.parameters())
        original_dy_dx = list((_.detach().clone() for _ in dy_dx))

        # generate dummy data and label and push to gpu
        dummy_data = torch.randn(data.size()).to(device).requires_grad_(True)
        dummy_label = torch.randn((data.shape[0], num_classes)).to(device).requires_grad_(True)

        if method == 'DLG':
            optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=lr)
        elif method == 'iDLG':
            optimizer = torch.optim.LBFGS([dummy_data, ], lr=lr)
            # predict the ground-truth label
            label_pred = torch.argmin(torch.sum(original_dy_dx[-2], dim=-1), dim=-1).detach().reshape(
                (1,)).requires_grad_(False)

        history = []
        history_iters = []
        losses = []
        mses = []
        train_iters = []

        print('lr =', lr)
        for iters in range(Iteration):


            def closure():
                optimizer.zero_grad()
                #feed dummy data to model
                pred = model(dummy_data)

                #calculate loss
                if method == 'DLG':
                    dummy_loss = - torch.mean(
                        torch.sum(torch.softmax(dummy_label, -1) * torch.log(torch.softmax(pred, -1)), dim=-1))
                    # dummy_loss = criterion(pred, label)
                elif method == 'iDLG':
                    dummy_loss = criterion(pred, label_pred)

                #calculate dummy gradient
                dummy_dy_dx = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

                #calculate square vector difference between original and dummy gradient
                grad_diff = 0
                for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                    grad_diff += ((gx - gy) ** 2).sum()

                #calculate gradient of that tensor
                grad_diff.backward()
                return grad_diff

            #perform a learning step
            optimizer.step(closure)
            current_loss = closure().item()

            #gather lists with iterations, losses, mses
            train_iters.append(iters)
            losses.append(current_loss)
            mses.append(torch.mean((dummy_data - data) ** 2).item())

            #log after some iterations
            if iters % int(Iteration / 30) == 0:
                print(iters, 'loss = %.8f, mse = %.8f' % (current_loss, mses[-1]))
                history.append([tp(dummy_data[0].cpu())])
                history_iters.append(iters)

                #plot the history of dummy visualisations
                plt.figure(figsize=(12, 8))
                plt.subplot(3, 10, 1)
                plt.imshow(tp(data[0].cpu()))
                for i in range(min(len(history), 29)):
                    plt.subplot(3, 10, i + 2)
                    plt.imshow(history[i][0])
                    plt.title('iter=%d' % (history_iters[i]))
                    plt.axis('off')
                if method == 'DLG':
                    plt.savefig('%s/%s DLG.png' % (save_path, idx))
                    plt.close()
                elif method == 'iDLG':
                    plt.savefig('%s/%s iDLG.png' % (save_path, idx))
                    plt.close()

                #eventually break
                if current_loss < 0.000001:  # converge
                    break


##########################################################################

def log(str):
    print(str)
    logging.info(str)


##########################################################################
##########################################################################

if __name__ == '__main__':

    dlg()
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
        transforms.ToTensor()
    ])

    # Initialising datasets
    train_dataset = datasets.MNIST('./datasets', train=True, download=True, transform=transformation)
    test_dataset = datasets.MNIST('./datasets', train=False, download=True, transform=transformation)
    num_classes = 10
    channel = 1

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
    model = Net(channel, num_classes).to(device)

    # Initialise Optimizer
    optimizer = optim.SGD(model.parameters(), lr=args["learning_rate"])

    # Training Loop
    for epoch in range(1, args["epochs"] + 1):
        train()

    # Testing
    test()

    # Saving Model
    torch.save(model.state_dict(), "{}model.pt".format(resultpath))





#################################################


