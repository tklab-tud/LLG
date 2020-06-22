import torch
import numpy as np
from torchvision import datasets, transforms
import torch.nn as nn

def dlg(model, train_dataset, parameter, device):
    dummy_data = torch.rand((parameter["batch_size"], 1 ,28 ,28 )).to(device).requires_grad_(True)
    dummy_label = torch.randn((parameter["batch_size"], parameter["num_classes"])).to(device).requires_grad_(True)

    ids= np.random.permutation(len(train_dataset))[:parameter["batch_size"]]

    # prepare attacked_batch
    data = []
    labels = []
    for id in ids:
        #enque sample
        sample = train_dataset[id][0]
        sample = transforms.ToTensor()(sample)
        data.append(sample)
        #enque data
        label = train_dataset[id][1]
        labels.append(label)


    for method in ['DLG', 'iDLG']:
        criterion = nn.CrossEntropyLoss().to(device)





    return (dummy_label, dummy_data)