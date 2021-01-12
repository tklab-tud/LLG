import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms


class Dataloader():
    def __init__(self):
        self.train_dataset = None
        self.currently_loaded = None
        self.samples = None
        self.num_classes = None

    def load_dataset(self, dataset):
        if dataset=="DUMMY": return

        print("Loading dataset " + dataset + ". This may take some seconds.")

        tt = torchvision.transforms.ToTensor()
        tt_grey = transforms.Compose([torchvision.transforms.Grayscale(), tt])

        if dataset == "MNIST":
            self.train_dataset = datasets.MNIST('./datasets', train=True, download=True, transform=tt)
            self.num_classes = 10
        elif dataset == 'CIFAR':
            self.train_dataset = datasets.CIFAR100('./datasets', train=True, download=True, transform=tt)
            self.num_classes = 100
        elif dataset == 'CIFAR-grey':
            self.train_dataset = datasets.CIFAR100('./datasets', train=True, download=True, transform=tt_grey)
            self.num_classes = 100
        elif dataset == 'CELEB-A':
            self.train_dataset = datasets.CelebA('./datasets', 'all', 'identity', download=True, transform=tt)
            self.num_classes = 100
            self.train_dataset.targets = self.train_dataset.identity
        elif dataset == 'CELEB-A-male':
            self.train_dataset = datasets.CelebA('./datasets', 'all', 'attr', download=True, transform=tt)
            self.num_classes = 2
            #filtering male from attributes and set it as target
            length = len(self.train_dataset.attr)
            self.train_dataset.targets = torch.gather(self.train_dataset.attr, 1, torch.Tensor(length*[20]).long().view(-1, 1))
        elif dataset == 'SVHN':
            self.train_dataset = datasets.SVHN('./datasets', 'train', download=True, transform=tt)
            self.num_classes = 10
            self.train_dataset.targets = self.train_dataset.labels.tolist()
        else:
            print("Unsupported dataset '" + dataset + "'")
            exit()

        self.currently_loaded = dataset

        # indexing with fix for CIFAR
        if dataset == "CIFAR" or dataset == "CIFAR-grey" or dataset == "SVHN":
            self.samples = [[] for _ in range(max(self.train_dataset.targets) + 1)]
        else:
            self.samples = [[] for _ in range(max(self.train_dataset.targets).item()+1)]

        for i, sample in enumerate(self.train_dataset.targets):
            if dataset == "CIFAR" or dataset == "CIFAR-grey" or dataset == "SVHN":
                self.samples[sample-1].append(i)
            else:
                self.samples[sample.item() - 1].append(i)

        print("Finished loading dataset")

    # returns label and data of batch size. Will take targeted classes and fills it with random classes.
    def get_batch(self, dataset, targets, bs):
        # update dataset if necessary
        if self.currently_loaded != dataset:
            self.load_dataset(dataset)

        # remove additional targets or fill up
        targets = targets[:bs]
        while len(targets) < bs:
            targets.append(np.random.randint(self.num_classes))

        # prepare data and label tensor
        data = torch.Tensor(bs, self.train_dataset[0][0].shape[0], self.train_dataset[0][0].shape[1],
                            self.train_dataset[0][0].shape[2])
        labels = torch.Tensor(bs).long()

        # fill data and labels
        for i_target, target in enumerate(targets):
            rnd = np.random.randint(len(self.samples[target]))
            data[i_target] = self.train_dataset[self.samples[target][rnd]][0].float().unsqueeze(0)
            data[i_target] = data[i_target].view(1, *data[i_target].size())
            labels[i_target] = torch.Tensor([target]).long()
            labels[i_target] = labels[i_target].view(1, )

        # dummys
        if dataset=="DUMMY":
            data.zero_()

        return data, labels
