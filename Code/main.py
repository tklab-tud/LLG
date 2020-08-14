from experiments import *
from visualize_experiment import *


def main():
    ############## Build your attack here ######################

    ### Experiment 1: Prediction Accuracy ###
    # setting = prediction_accuracy_vs_batchsize(3, [1, 2, 4, 8, 16, 32, 64, 128, 256], "MNIST", True)
    # setting = prediction_accuracy_vs_batchsize(100, [1, 2, 4, 8, 16, 32, 64, 128, 256], "MNIST", False)
    # setting = prediction_accuracy_vs_batchsize(100, [1, 2, 4, 8, 16, 32, 64, 128, 256], "CIFAR", True)
    # setting = prediction_accuracy_vs_batchsize(100, [1, 2, 4, 8, 16, 32, 64, 128, 256], "CIFAR", False)
    # visualize_prediction_accuracy_vs_batchsize()

    ### Experiment 2: Good Fidelity ###
    # setting = good_fidelity(3, 8, 50, "MNIST", True)
    # setting = good_fidelity(10, 8, 100, "MNIST", False)
    # setting = good_fidelity(1, 8, 100, "CIFAR", True)
    # setting = good_fidelity(10, 8, 100, "CIFAR", False)
    # visualize_good_fidelity()

    ### Bonus 1: Perfect Prediction ###
    # setting = perfect_prediction(10, [1, 2, 4, 8, 16, 32, 64, 128, 256], "MNIST", True)
    # setting = perfect_prediction(100, [1, 2, 4, 8, 16, 32, 64, 128, 256], "MNIST", False)
    # setting = perfect_prediction(100, [1, 2, 4, 8, 16, 32, 64, 128, 256], "CIFAR", True)
    # setting = perfect_prediction(100, [1, 2, 4, 8, 16, 32, 64, 128, 256], "CIFAR", False)
    # visualize_perfect_prediction()

    ### Bonus 2: Prediction vs Training ###
    # setting = prediction_accuracy_vs_training(1, 8, "MNIST", True, 1000, 10)
    visualize_prediction_accuracy_vs_training()

    ############################################################
    print("Run finished")


if __name__ == '__main__':
    main()
