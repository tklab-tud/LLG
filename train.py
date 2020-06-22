def train(model, train_dataset, parameter, device):
    print("Training for {} epochs sized {} in batches of {}".format(parameter["epochs"], parameter["max_epoch_size"],
                                                                    parameter["batch_size"]))