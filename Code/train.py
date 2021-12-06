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

    criterion = torch.nn.CrossEntropyLoss().to(device)

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


def update_weights(model, setting, id, victim: bool=False):
    device = setting.device
    parameter = setting.parameter
    defs = setting.defenses
    local_iterations = setting.parameter["local_iterations"]
    local_training = setting.parameter["local_training"]
    if victim or not local_training:
        local_iterations = 1

    # Set mode to train model
    model.train()

    dataloader = setting.dataloader

    optimizer = torch.optim.SGD(model.parameters(), lr=parameter["train_lr"])

    criterion = torch.nn.CrossEntropyLoss().to(device)

    seperated_gradients = []

    for i in range(local_iterations):
        # TODO: HFL data distribution?
        data, target = dataloader.get_batch(setting.parameter["dataset"], setting.parameter["targets"],
                                            setting.parameter["batch_size"], random=(not victim))

        data, target = data.to(device), target.to(device)

        output = model(data)

        def closure():
            optimizer.zero_grad()
            # FIXME: should we zero the gradients? where is the difference?
            # model.zero_grad()
            output = model(data)

            loss = criterion(output, target)
            loss.backward()
            return loss

        loss = criterion(output, target)

        grad = torch.autograd.grad(loss, model.parameters(), retain_graph=True)
        seperated_gradients.append(list((_.detach().clone() for _ in grad)))

        loss = optimizer.step(closure)

    # Copy the structure of a grad, but make it zeroes
    aggregated = list(x.zero_() for x in grad)

    # iterate over the gradients for each local iteration
    for grad in seperated_gradients:
        # there iterate through the gradients and add to the aggregator
        for i_g,g in enumerate(grad):
            aggregated[i_g] = torch.add(aggregated[i_g], g)

    defs.apply(aggregated, id)

    if parameter["differential_privacy"] or parameter["compression"]:
        defs.inject(seperated_gradients, aggregated, model)

    return model.state_dict()


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

    print("Training {} users for {} local and {} global iterations".format(num_users, local_iterations, global_iterations))

    # m = min(int(len(keylist_cluster)), num_users_per_epoch)
    # idxs_users = np.random.choice(keylist_cluster, m, replace=False)

    for i in range(global_iterations):
        local_weights = []

        for i in range(num_users-1):
            w = update_weights(copy.deepcopy(global_model), setting, id=i)
            local_weights.append(copy.deepcopy(w))

        victim_weights = victim_model.state_dict()
        local_weights.append(copy.deepcopy(victim_weights))

        # Averaging local client weights to get global weights
        global_weights = average_weights(local_weights, num_users)

        # update global weights
        global_model.load_state_dict(global_weights)

    test(setting)
