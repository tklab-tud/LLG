import numpy as np
import torch
import torchvision
from torchvision import datasets


class Dataloader():
    def __init__(self, setting, dataset):
        self.setting = setting
        self.load_dataset(dataset)



    def load_dataset(self, dataset):
        print("Loading dataset "+dataset+". This may take some seconds.")

        tt = torchvision.transforms.ToTensor()
        parameter = self.setting.parameter

        if dataset == "MNIST":
            parameter["shape_img"] = (28, 28)
            parameter["num_classes"] = 10
            parameter["channel"] = 1
            parameter["hidden"] = 588
            parameter["hidden2"] = 9216
            self.train_dataset = datasets.MNIST('./datasets', train=True, download=True, transform=tt)
        elif dataset == 'CIFAR':
            parameter["shape_img"] = (32, 32)
            parameter["num_classes"] = 100
            parameter["channel"] = 3
            parameter["hidden"] = 768
            parameter["hidden2"] = 12544
            self.train_dataset = datasets.CIFAR100('./datasets', train=True, download=True, transform=tt)
        else:
            print("Unsupported dataset '" + dataset + "'")
            exit()

        self.currently_loaded = dataset
        # indexing
        self.samples = [[] for _ in range(parameter["num_classes"])]

        for sample in self.train_dataset:
            self.samples[sample[1]].append(sample)

        print("Finished loading dataset")

    # returns label and data of batch size. Will take targeted classes and fills it with random classes.
    def get_batch(self, setting):
        self.setting = setting
        parameter = self.setting.parameter

        # update dataset if necessary
        if parameter["dataset"] != self.currently_loaded:
            self.load_dataset(parameter["dataset"])

        # remove additional targets or fill up
        targets = parameter["targets"][:parameter["batch_size"]]
        while len(targets) < parameter["batch_size"]:
            targets.append(np.random.randint(self.setting.parameter["num_classes"]))

        # prepare data and label tensor
        data = torch.Tensor(parameter["batch_size"], parameter["channel"],
                            parameter["shape_img"][0], parameter["shape_img"][1]).to(self.setting.device)
        labels = torch.Tensor(parameter["batch_size"]).long().to(self.setting.device).view(
            parameter["batch_size"])

        # fill data and labels
        for i_target, target in enumerate(targets):
            rnd = np.random.randint(len(self.samples[target]))
            data[i_target] = self.samples[target][rnd][0]
            labels[i_target] = self.samples[target][rnd][1]

        return data, labels



    def get_batch(self, setting):
        self.setting = setting
        parameter = self.setting.parameter

        # update dataset if necessary
        if parameter["dataset"] != self.currently_loaded:
            self.load_dataset(parameter["dataset"])

        # remove additional targets or fill up
        targets = parameter["targets"][:parameter["batch_size"]]
        while len(targets) < parameter["batch_size"]:
            targets.append(np.random.randint(self.setting.parameter["num_classes"]))

        # prepare data and label tensor
        data = torch.Tensor(parameter["batch_size"], parameter["channel"],
                            parameter["shape_img"][0], parameter["shape_img"][1]).to(self.setting.device)
        labels = torch.Tensor(parameter["batch_size"]).long().to(self.setting.device).view(
            parameter["batch_size"])

        # fill data and labels
        for i_target, target in enumerate(targets):
            rnd = np.random.randint(len(self.samples[target]))
            data[i_target] = self.samples[target][rnd][0]
            labels[i_target] = self.samples[target][rnd][1]

        return data, labels



