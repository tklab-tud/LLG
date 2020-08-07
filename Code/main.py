from examples import *


def main():
    ############## Build your attack here ######################

    setting, graph = prediction_accuracy_vs_batchsize(10, [1, 16, 32, 64, 128, 256], "MNIST", True)
    # setting, graph = prediction_accuracy_vs_batchsize(10, [1, 16, 32, 64, 128, 256], "MNIST", False)
    # setting, graph = prediction_accuracy_vs_batchsize(10, [1, 16, 32, 64, 128, 256], "CIFAR", True)
    # setting, graph = prediction_accuracy_vs_batchsize(10, [1, 16, 32, 64, 128, 256], "CIFAR", False)

    # Experiment 2
    #mse_vs_iteration_line(1, 32, 60, "MNIST", True)
    #mse_vs_iteration_line(1, 32, 60, "MNIST", False)
    #mse_vs_iteration_line(1, 32, 60, "CIFAR", True)
    #mse_vs_iteration_line(1, 32, 60, "CIFAR", False)

    # Experiment 3
    # setting, graph = perfect_prediction(10, [1, 16, 32, 64, 128, 256], "MNIST", True)
    # setting, graph = perfect_prediction(10, [1, 16, 32, 64, 128, 256], "MNIST", False)
    # setting, graph = perfect_prediction(10, [1, 16, 32, 64, 128, 256], "CIFAR", True)
    # setting, graph = perfect_prediction(10, [1, 16, 32, 64, 128, 256], "CIFAR", False)

    ############################################################
    print("Run finished")


if __name__ == '__main__':
    main()
