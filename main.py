import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision as torchvision
from torchvision import datasets, transforms
import syft as sy
from torchvision import transforms, datasets



ap = argparse.ArgumentParser(description="Comparison framework for attacks on federated learning.")

#ap.add_argument("-d", "--dataset", required=False, default="MNIST", help="Dataset to use", choices=["MNIST", "SVHN", "CIFAR", "ATT"])
#ap.add_argument("-a", "--attack", required=False, default="GAN", help="Attacks to perform", choices=["GAN", "MI", "UL", "DLG", "iDLG"])
#ap.add_argument("-n", "--nodes", required=False, default=10, type=int , help="Amount of noodes")
#ap.add_argument("-s", "--size", required=False, default=10, type=int, help="Amount of samples per node")
#ap.add_argument("-lr", "--lrounds", required=False, default=10 , help="Amount of learning rounds")
#ap.add_argument("-c", "--cpr", required=False, default=10 , help="Amount of clients per learning round")
#ap.add_argument("-S", "--selective", required=False, default=False , help="Selective up- and download", action='store_true')

#ap.add_argument("-ar", "--arounds", required=False, default=10 , help="Amount of attacking rounds")

args = vars(ap.parse_args())

for arg in args:
    print(str(arg)+" "+str(args[arg]))

###############################################



hook = sy.TorchHook(torch)
bob = sy.VirtualWorker(hook, id="bob")
alice = sy.VirtualWorker(hook, id="alice")


batch_size = 1
test_batch_size = 100
epochs = 3
lr = 0.01
momentum = 0.5
no_cuda = False
seed = 1
log_interval = 10
save_model = True
batches_per_epoch = 100



use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda")

torch.manual_seed(seed)

transformation = transforms.Compose([
    transforms.Resize([32, 32]),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


test_dataset = datasets.ImageFolder(root='./datasets/datasets/CIFAR-10/test/', transform=transformation)
train_dataset = datasets.ImageFolder(root='./datasets/datasets/CIFAR-10/train/', transform=transformation)

#train_dataset = datasets.MNIST('./datasets', train=True, download=True, transform=transformation)
#test_dataset = datasets.MNIST('./datasets', train=False, download=True, transform=transformation)



federated_train_loader = sy.FederatedDataLoader(
    train_dataset.federate((bob, alice)),
    batch_size=batch_size,
    shuffle=True,
    num_workers= 0,
    pin_memory=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=test_batch_size,
    shuffle=True,
    num_workers= 0,
    pin_memory=True
)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train():
    model.train()
    for batch_idx, (data, target) in enumerate(federated_train_loader): # <-- now it is a distributed dataset
        model.send(data.location) # <-- NEW: send the model to the right location
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        model.get()
        if batch_idx % log_interval == 0:
            loss = loss.get() # <-- NEW: get the loss back
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * batch_size, batches_per_epoch * batch_size,
                100* batch_idx / batches_per_epoch, loss.item()))

        if batch_idx >= batches_per_epoch:
            break




def test():
    print("Start Testing")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



if __name__ == '__main__':
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        train()

    print("Training finished")

    test()

    if (save_model):
        torch.save(model.state_dict(), "model.pt")