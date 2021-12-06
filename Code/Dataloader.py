import numpy as np
import statistics as stats
import torch
import torchvision
from torchvision import datasets, transforms

import FEMNIST.femnist as femnist

class Dataloader():
    def __init__(self):
        self.train_dataset = None
        self.currently_loaded = None
        self.samples = None
        self.num_classes = None

    def load_dataset(self, dataset):
        if dataset in ["DUMMY-ONE", "DUMMY-ZERO", "DUMMY-RANDOM"]: return

        print("Loading dataset " + dataset + ". This may take some seconds.")

        tt = torchvision.transforms.ToTensor()
        tt_grey = transforms.Compose([torchvision.transforms.Grayscale(), tt])

        if dataset == "MNIST":
            self.train_dataset = datasets.MNIST('./datasets', train=True, download=True, transform=tt)
        elif dataset == "EMNIST":
            self.train_dataset = datasets.EMNIST('./datasets', split="byclass", train=True, download=True, transform=tt)
        elif dataset == "FEMNIST":
            self.train_dataset = femnist.FEMNIST('./datasets', train=True, download=True, transform=tt)
        elif dataset == "FEMNIST-digits":
            self.train_dataset = femnist.FEMNIST_digits('./datasets', train=True, download=True, transform=tt)
        elif dataset == 'CIFAR':
            self.train_dataset = datasets.CIFAR100('./datasets', train=True, download=True, transform=tt)
        elif dataset == 'CIFAR-grey':
            self.train_dataset = datasets.CIFAR100('./datasets', train=True, download=True, transform=tt_grey)
        elif dataset == 'CELEB-A':
            self.train_dataset = datasets.CelebA('./datasets', 'all', 'identity', download=True, transform=tt)
            self.train_dataset.targets = self.train_dataset.identity
        elif dataset == 'CELEB-A-male':
            self.train_dataset = datasets.CelebA('./datasets', 'all', 'attr', download=True, transform=tt)
            # filtering male from attributes and set it as target
            length = len(self.train_dataset.attr)
            self.train_dataset.targets = torch.gather(self.train_dataset.attr, 1, torch.Tensor(length*[20]).long().view(-1, 1))
        elif dataset == 'CELEB-A-hair':
            self.train_dataset = datasets.CelebA('./datasets', 'all', 'attr', download=True, transform=tt)
            length = len(self.train_dataset.attr)
            self.train_dataset.targets = []

            for i, attr in enumerate(self.train_dataset.attr):
                bald = attr[4]
                black = attr[8]
                blond = attr[9]
                brown = attr[11]
                gray = attr[17]
                other = 0

                if bald+black+blond+brown+gray != 1:
                    bald = black = blond = brown = gray = 0
                    other = 1

                self.train_dataset.targets.append(np.argmax([bald,black,blond,brown,gray,other])+1)

            print("done")

        elif dataset == 'SVHN':
            self.train_dataset = datasets.SVHN('./datasets', 'train', download=True, transform=tt)
            self.train_dataset.targets = self.train_dataset.labels.tolist()
        else:
            print("Unsupported dataset '" + dataset + "'")
            exit()

        # number of classes
        if isinstance(self.train_dataset.targets, torch.Tensor):
            self.num_classes = int(max(self.train_dataset.targets).item() + 1)
        elif isinstance(self.train_dataset.targets, list):
            self.num_classes = int(max(self.train_dataset.targets) + 1)
        else:
            print("Unsupported target type '" + type(self.train_dataset.targets) + "'")
            exit()

        self.currently_loaded = dataset

        # print dataset statistics
        print("train dataset: ", self.train_dataset)
        print("number of classes: ", self.num_classes)
        print("number of targets: ", len(self.train_dataset.targets))
        # print("target count:      ", self.train_dataset.targets.bincount())
        print("train dataset: ", self.train_dataset)
        # print("sample data: ", self.train_dataset.data[0])
        # print("type: ", type(self.train_dataset.data[0]))
        # print("lenx: ", len(self.train_dataset.data[0]))
        # print("leny: ", len(self.train_dataset.data[0][0]))
        print("sample target: ", self.train_dataset.targets[0])
        print("type: ", type(self.train_dataset.targets[0]))
        #print("len: ", len(self.train_dataset.targets[0]))

        if hasattr(self.train_dataset, 'users_index'):
            print("number of users:   ", len(self.train_dataset.users_index))
            # users_index is not an index at all. it is the number of samples per user.
            # user 0 owns samples from 0 to users_index[0]
            # user i owns samples from sum(users_index[0:i-1]) to sum(users_index[0:i])
            print("samples per user:")
            print("min:   ", min(self.train_dataset.users_index))
            print("max:   ", max(self.train_dataset.users_index))
            print("mean:  ", stats.mean(self.train_dataset.users_index))
            print("stdev: ", stats.stdev(self.train_dataset.users_index))

            # min_samples = {8: 0, 16: 0, 32: 0, 64: 0, 128: 0}
            # for num_samples in self.train_dataset.users_index:
            #     for min_size in min_samples.keys():
            #         if num_samples < min_size:
            #             min_samples[min_size] += 1
            #             break

            # print(min_samples)
            # print(len(self.train_dataset.users_index))
            # exit()

        # indexing with fix for CIFAR
        self.samples = [[] for _ in range(self.num_classes)]

        for i, sample in enumerate(self.train_dataset.targets):
            if dataset == "CIFAR" or dataset == "CIFAR-grey" or dataset == "SVHN":
                self.samples[sample-1].append(i)
            else:
                self.samples[sample.item() - 1].append(i)

        print("Finished loading dataset")

    # get targets based on user_id for FEMNIST, FEMNIST-digits
    def get_batch_user_targets(self, user_id, bs):
        if user_id == None:
            user_id = np.random.randint(len(self.train_dataset.users_index))
        # user i owns samples from sum(users_index[0:i-1]) to sum(users_index[0:i])
        num_samples = self.train_dataset.users_index[user_id]
        index_start = sum(self.train_dataset.users_index[0:user_id-1])
        index_end = sum(self.train_dataset.users_index[0:user_id])

        samples = []
        index = len(self.train_dataset.data)
        i = 0
        while len(samples) < bs:
            while index >= len(self.train_dataset.data):
                index = index_start + np.random.randint(num_samples)
                if i >= 10 and index >= len(self.train_dataset.data):
                    index = len(self.train_dataset.data)-1
                i += 1
            samples.append(index)

        return samples

    # returns label and data of batch size. Will take targeted classes and fills it with random classes.
    def get_batch(self, dataset, targets, bs, random=False):
        # update dataset if necessary
        if self.currently_loaded != dataset:
            self.load_dataset(dataset)

        # prepare data and label tensor
        data = torch.Tensor(bs, self.train_dataset[0][0].shape[0], self.train_dataset[0][0].shape[1],
                            self.train_dataset[0][0].shape[2])
        labels = torch.Tensor(bs).long()

        # get user samples
        if hasattr(self.train_dataset, 'users_index') and not isinstance(targets, list):
            samples = self.get_batch_user_targets(targets, bs)
            for i, sample in enumerate(samples):
                data[i] = self.train_dataset.data[sample]
                labels[i] = self.train_dataset.targets[sample]

            return data, labels

        # get random non-iid targets
        if random:
            targets = self.get_random_targets(bs)

        # remove additional targets or fill up
        targets = targets[:bs]
        while len(targets) < bs:
            targets.append(np.random.randint(self.num_classes))

        # fill data and labels
        for i_target, target in enumerate(targets):
            rnd = np.random.randint(len(self.samples[target]))
            data[i_target] = self.train_dataset[self.samples[target][rnd]][0].float().unsqueeze(0)
            data[i_target] = data[i_target].view(1, *data[i_target].size())
            labels[i_target] = torch.Tensor([target]).long()
            labels[i_target] = labels[i_target].view(1, )

        # dummys
        if dataset=="DUMMY-ONE":
            data = torch.ones(data.size())
        elif dataset=="DUMMY-ZERO":
            data = data.zero_()
        elif dataset=="DUMMY-RANDOM":
            data = data.random_()

        return data, labels

    def get_random_targets(self, bs: int):
        # get random user index
        if hasattr(self.train_dataset, 'users_index'):
            return np.random.randint(len(self.train_dataset.users_index))
        # we define unbalanced as 50% class a, 25% class b, 25% random
        choice1 = np.random.choice(self.num_classes)
        choice2 = np.random.choice(
            np.setdiff1d(range(self.num_classes), choice1)).item()
        target = (bs // 2) * [choice1] + (bs // 4) * [choice2]
        target = target[:bs]
        return target
