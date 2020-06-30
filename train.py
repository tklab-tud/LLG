import torch
import torchvision
from net import Net, Net2

def train(model, train_dataset, parameter, device):
    print("Training for {} epochs sized {} in batches of {}".format(parameter["epochs"], parameter["max_epoch_size"],
                                                                    parameter["batch_size"]))

    #model = Net2().to(device)

    train_loader = torch.utils.data.DataLoader(train_dataset, parameter["batch_size"], shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=parameter["lr"])
    criterion = torch.nn.CrossEntropyLoss().to(device)


    for epoch in range(parameter["epochs"]):
        print("### Epoch {} ###".format(epoch))
        for batch_idx, batch in enumerate(train_loader):
            data = batch[0].to(device)
            target = batch[1].to(device)

            optimizer.zero_grad()
            output = model(data)
            #loss = torch.nn.functional.nll_loss(output, target)
            loss = criterion(output, target)
            loss.backward()

            optimizer.step()

            if batch_idx % parameter["log_interval"] == 0:
                print('Epoch: {} [{}/{}]\tLoss: {}'.format(epoch, batch_idx, parameter["max_epoch_size"],
                                                           loss.item()))

            if batch_idx > parameter["max_epoch_size"]:
                break
