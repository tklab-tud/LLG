import torch

import aDPtorch.privacy_engine_xl as adp


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
            data, target = dataloader.get_batch(setting.parameter["dataset"], setting.parameter["targets"],
                                                setting.parameter["batch_size"])

            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= parameter["test_size"]

    test_acc = 100. * correct / (parameter["test_size"] * parameter["batch_size"])

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, parameter["test_size"] * parameter["batch_size"], test_acc))

    setting.parameter["test_loss"], setting.parameter["test_acc"] = test_loss, test_acc


def train(setting, train_size):
    # some abbreviations
    parameter = setting.parameter
    device = setting.device
    model = setting.model

    model.train()

    print("Training for {} batches".format(train_size))

    dataloader = setting.dataloader

    optimizer = torch.optim.SGD(model.parameters(), lr=parameter["train_lr"])
    #optimizer = torch.optim.LBFGS(model.parameters(), lr=parameter["train_lr"])

    criterion = torch.nn.CrossEntropyLoss().to(device)

    # Differential Privacy
    # if parameter["differential_privacy"]:
    #     privacy_engine = adp.PrivacyEngineXL(
    #         model,
    #         batch_size=parameter["batch_size"],
    #         sample_size=len(dataloader.train_dataset),
    #         alphas=parameter["alphas"],
    #         noise_multiplier=parameter["noise_multiplier"],
    #         secure_rng=True, # Note: this is not yet implemented in aDPtorch, it is set to avoid warning spamming
    #         max_grad_norm=parameter["max_norm"],
    #         noise_type=parameter["noise_type"]
    #     )
    #     privacy_engine.attach(optimizer)

    for i in range(train_size):
        def closure():
            optimizer.zero_grad()

            data, target = dataloader.get_batch(setting.parameter["dataset"], setting.parameter["targets"],
                                                setting.parameter["batch_size"])

            #data, target = data.to(device), target.to(device)

            output = model(data)

            loss = criterion(output, target)
            loss.backward()
            return loss

        optimizer.step(closure)


    #test(setting)
