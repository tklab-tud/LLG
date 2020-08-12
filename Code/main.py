from examples import *


def main():
    ############## Build your attack here ######################

    ### Experiment 1: Prediction Accuracy ###
    # setting, graph = prediction_accuracy_vs_batchsize(100, [1, 2, 4, 8, 16, 32, 64, 128, 256], "MNIST", True)
    # setting, graph = prediction_accuracy_vs_batchsize(100, [1, 2, 4, 8, 16, 32, 64, 128, 256], "MNIST", False)
    # setting, graph = prediction_accuracy_vs_batchsize(100, [1, 2, 4, 8, 16, 32, 64, 128, 256], "CIFAR-grey", True)
    # setting, graph = prediction_accuracy_vs_batchsize(100, [1, 2, 4, 8, 16, 32, 64, 128, 256], "CIFAR", False)

    ### Experiment 2: Good Fidelity ###
    # setting, graph = good_fidelity(10, 8, 100, "MNIST", True)
    # setting, graph = good_fidelity(10, 8, 100, "MNIST", False)
    # setting, graph = good_fidelity(1, 8, 100, "CIFAR", True)
    # setting, graph = good_fidelity(10, 8, 100, "CIFAR", False)

    ### Bonus 1: Perfect Prediction ###
    setting, graph = perfect_prediction(100, [1, 2, 4, 8, 16, 32, 64, 128, 256], "MNIST", True)
    setting, graph = perfect_prediction(100, [1, 2, 4, 8, 16, 32, 64, 128, 256], "MNIST", False)
    setting, graph = perfect_prediction(100, [1, 2, 4, 8, 16, 32, 64, 128, 256], "CIFAR", True)
    setting, graph = perfect_prediction(100, [1, 2, 4, 8, 16, 32, 64, 128, 256], "CIFAR", False)

    ############################################################
    print("Run finished")


if __name__ == '__main__':
    main()
