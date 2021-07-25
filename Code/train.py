import copy
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

    #print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #    test_loss, correct, parameter["test_size"] * parameter["batch_size"], test_acc))

    setting.parameter["test_loss"], setting.parameter["test_acc"] = test_loss, test_acc


def train(setting, train_size, batch=None):
    # some abbreviations
    parameter = setting.parameter
    device = setting.device
    model = setting.model

    model.train()


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

            if batch is None:
                data, target = dataloader.get_batch(setting.parameter["dataset"], setting.parameter["targets"],
                                                    setting.parameter["batch_size"])
            else:
                data, target = batch

            data, target = data.to(device), target.to(device)

            output = model(data)

            loss = criterion(output, target)
            loss.backward()
            return loss

        optimizer.step(closure)

    # Dont need test for local training
    if batch is None:
        test(setting)


def update_weights(model, setting, victim: bool=False):
    device = setting.device
    parameter = setting.parameter
    local_iterations = setting.parameter["local_iterations"]
    if victim:
        local_iterations = 1

    # Set mode to train model
    model.train()
    epoch_loss = []

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

    for i in range(local_iterations):
        batch_loss = []
        def closure():
            optimizer.zero_grad()
            # FIXME: should we zero the gradients? where is the difference?
            # model.zero_grad()

            # TODO: HFL data distribution?
            data, target = dataloader.get_batch(setting.parameter["dataset"], setting.parameter["targets"],
                                                setting.parameter["batch_size"], random=(not victim))

            data, target = data.to(device), target.to(device)

            output = model(data)

            loss = criterion(output, target)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        # FIXME: loss does not get tracked constantely
        # I assume we just take the last loss instead of average like (H)FL code
        # logger.add_scalar('loss', loss.item())
        batch_loss.append(loss.item())
    epoch_loss.append(sum(batch_loss)/len(batch_loss))

    return model.state_dict(), sum(epoch_loss) / len(epoch_loss)


def average_weights(w, num_users):
    """
    Returns the average of the weights.
    """

    percentage = 1/num_users

    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = 0
        for i in range(0, len(w)):
            w_avg[key] += w[i][key] * percentage
    return w_avg


def train_federated(setting):
    # some abbreviations
    parameter = setting.parameter
    device = setting.device
    global_model = setting.model_backup
    victim_model = setting.model
    global_iterations = setting.parameter["train_size"]
    local_iterations = setting.parameter["local_iterations"]
    num_users = setting.parameter["num_users"]

    global_model.train()

    print("Training for {} global and {} local iterations".format(global_iterations, local_iterations))

    # m = min(int(len(keylist_cluster)), num_users_per_epoch)
    # idxs_users = np.random.choice(keylist_cluster, m, replace=False)

    for i in range(global_iterations):
        local_weights = []
        local_losses = []

        for i in range(num_users-1):
            w, loss = update_weights(copy.deepcopy(global_model), setting)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        victim_weights, victim_loss = update_weights(copy.deepcopy(victim_model), setting, victim=True)
        local_weights.append(copy.deepcopy(victim_weights))

        # Averaging local client weights to get global weights
        global_weights = average_weights(local_weights, num_users)

        # update global weights
        global_model.load_state_dict(global_weights)

    test(setting)
