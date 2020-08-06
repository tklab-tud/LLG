from examples import *


def main():
    ############## Build your attack here ######################

    # setting, graph = prediction_accuracy_vs_batchsize(10, range(1,257), "MNIST", True)
    # setting, graph = prediction_accuracy_vs_batchsize(10, range(1,257), "MNIST", False)
    # setting, graph = prediction_accuracy_vs_batchsize(10, range(1,257), "CIFAR", True)
    # setting, graph = prediction_accuracy_vs_batchsize(10, range(1,257), "CIFAR", False)

    # setting, graph = prediction_accuracy_vs_training(1, 64, "MNIST", False, 10, 100)

    setting, graph = perfect_prediction(10, [32], "MNIST", True)

    # setting, graph = mse_vs_iteration_line(1, 32, 30 ,"MNIST", True)

    ############################################################
    print("Run finished")


if __name__ == '__main__':
    main()
