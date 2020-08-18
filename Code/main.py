from experiments import *
from visualize_experiment import *


def main():
    ############## Build your attack here ######################

    ### Experiment 1: Class Prediction Accuracy ###
    #setting = class_prediction_accuracy_vs_batchsize(10, [1, 2, 4, 8, 16, 32, 64, 128, 256], "MNIST", True, "v2")
    #setting = class_prediction_accuracy_vs_batchsize(1000, [1, 2, 4, 8, 16, 32, 64, 128, 256], "MNIST", False, "v2")
    #setting = class_prediction_accuracy_vs_batchsize(10, [1, 2, 4, 8, 16, 32, 64, 128, 256], "CIFAR", True, "v2")
    #setting = class_prediction_accuracy_vs_batchsize(1000, [1, 2, 4, 8, 16, 32, 64, 128, 256], "CIFAR", False, "v2")
    run, path = load_json()
    visualize_class_prediction_accuracy_vs_batchsize(run, path )
    visualize_flawles_class_prediction_accuracy_vs_batchsize(run, path )
    negativ_value_check(run, path )
    magnitude_check(run, path, adjusted=True)
    magnitude_check(run, path, adjusted=False)

    ### Experiment 2: Class Prediction Accuracy vs Training ###
    # setting = class_prediction_accuracy_vs_training(1, 8, "MNIST", True, 1000, 10, "v2")
    # visualize_prediction_accuracy_vs_training()

    ### Experiment 3: Good Fidelity ###
    # setting = good_fidelity(10, 8, 3000, "MNIST", True)
    # setting = good_fidelity(10, 8, 3000, "MNIST", False)
    # setting = good_fidelity(10, 8, 3000, "CIFAR", True)
    # setting = good_fidelity(10, 8, 3000, "CIFAR", False)
    # visualize_good_fidelity()

    ############################################################
    print("Run finished")





if __name__ == '__main__':
    main()
