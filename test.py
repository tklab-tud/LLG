import torch


def test(setting):
    # some abbreviations
    parameter = setting.parameter
    device = setting.device
    test_dataset = setting.test_dataset
    model = setting.model

    model.eval()
    test_loss = 0
    correct = 0

    test_loader = torch.utils.data.DataLoader(test_dataset, parameter["batch_size"], shuffle=True)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            if i >= parameter["test_size"]:
                break

    test_loss /= parameter["test_size"]

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, parameter["test_size"] * parameter["batch_size"],
                            100. * correct / (parameter["test_size"] * parameter["batch_size"])))
