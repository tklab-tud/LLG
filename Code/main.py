from experiments import *
from visualize_experiment import *


def main():
    ############## Build your attack here ######################

    ### Experiment 1: Prediction Accuracy ###
    # setting, graph = prediction_accuracy_vs_batchsize(10, [1, 2, 4, 8, 16, 32, 64, 128, 256], "MNIST", True)
    # setting, graph = prediction_accuracy_vs_batchsize(100, [1, 2, 4, 8, 16, 32, 64, 128, 256], "MNIST", False)
    # setting, graph = prediction_accuracy_vs_batchsize(100, [1, 2, 4, 8, 16, 32, 64, 128, 256], "CIFAR", True)
    # setting, graph = prediction_accuracy_vs_batchsize(100, [1, 2, 4, 8, 16, 32, 64, 128, 256], "CIFAR", False)
    #visualize_prediction_accuracy_vs_batchsize()

    ### Experiment 2: Good Fidelity ###
    # setting, graph = good_fidelity(3, 8, 50, "MNIST", True)
    # setting, graph = good_fidelity(10, 8, 100, "MNIST", False)
    # setting, graph = good_fidelity(1, 8, 100, "CIFAR", True)
    # setting, graph = good_fidelity(10, 8, 100, "CIFAR", False)
    #visualize_good_fidelity()

    ### Bonus 1: Perfect Prediction ###
    # setting, graph = perfect_prediction(10, [1, 2, 4, 8, 16, 32, 64, 128, 256], "MNIST", True)
    # setting, graph = perfect_prediction(100, [1, 2, 4, 8, 16, 32, 64, 128, 256], "MNIST", False)
    # setting, graph = perfect_prediction(100, [1, 2, 4, 8, 16, 32, 64, 128, 256], "CIFAR", True)
    # setting, graph = perfect_prediction(100, [1, 2, 4, 8, 16, 32, 64, 128, 256], "CIFAR", False)
    visualize_perfect_prediction()

    ### Bonus 2: Prediction vs Training ###
    #setting, graph = prediction_accuracy_vs_training(1, 8, "MNIST", True, 1000, 10)
    #visualize_prediction_accuracy_vs_training()

    ############################################################
    print("Run finished")


if __name__ == '__main__':
    main()
