from examples import *


def main():
    ############## Build your attack here ######################

    ### Experiment 1: Prediction Accuracy ###
    # setting, graph = prediction_accuracy_vs_batchsize(10, [1, 2, 4, 6, 16, 32, 64, 128, 256], "MNIST", True)
    # setting, graph = prediction_accuracy_vs_batchsize(10, [1, 2, 4, 6, 16, 32, 64, 128, 256], "MNIST", False)
    # setting, graph = prediction_accuracy_vs_batchsize(10, [1, 2, 4, 6, 16, 32, 64, 128, 256], "CIFAR", True)
    # setting, graph = prediction_accuracy_vs_batchsize(10, [1, 2, 4, 6, 16, 32, 64, 128, 256], "CIFAR", False)

    ### Experiment 2: Good Fidelity ###
    setting, graph = good_fidelity(1, [1, 4, 16], "MNIST", True)
    # setting, graph = prediction_accuracy_vs_batchsize(10, [1, 2, 4, 6, 16, 32, 64, 128, 256], "MNIST", False)
    # setting, graph = prediction_accuracy_vs_batchsize(10, [1, 2, 4, 6, 16, 32, 64, 128, 256], "CIFAR", True)
    # setting, graph = prediction_accuracy_vs_batchsize(10, [1, 2, 4, 6, 16, 32, 64, 128, 256], "CIFAR", False)

    ############################################################
    print("Run finished")


if __name__ == '__main__':
    main()
