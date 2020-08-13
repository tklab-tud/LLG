import torch


def test(setting):
    # some abbreviations
    parameter = setting.parameter
    device = setting.device
    model = setting.model
    dataloader = setting.dataloader

    model.eval()
    test_loss = 0
    correct = 0

    criterion = torch.nn.CrossEntropyLoss().to(device)

    with torch.no_grad():
        for i in range(parameter["test_size"]):
            data, target = dataloader.get_batch(setting)
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= parameter["test_size"]

    test_acc = 100. * correct / (parameter["test_size"] * parameter["batch_size"])

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, parameter["test_size"] * parameter["batch_size"], test_acc))


    return test_loss, test_acc


def train(setting, train_size):
    # some abbreviations
    parameter = setting.parameter
    device = setting.device
    model = setting.model

    print("Training for {} batches".format(train_size))

    dataloader = setting.dataloader

    optimizer = torch.optim.SGD(model.parameters(), lr=parameter["train_lr"])
    criterion = torch.nn.CrossEntropyLoss().to(device)

    for i in range(train_size):

        data, target = dataloader.get_batch(setting)
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        optimizer.step()

        if i % 100 == 0:
            print('Sample [{}/{}]\tLoss: {}'.format(i, train_size, loss.item()))
