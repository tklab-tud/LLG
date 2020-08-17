from experiments import *
from visualize_experiment import *


def main():
    ############## Build your attack here ######################

    ### Experiment 1: Class Prediction Accuracy ###
    #setting = class_prediction_accuracy_vs_batchsize(10, [1, 2, 4, 8, 16, 32, 64, 128, 256], "MNIST", True, "v2")
    #setting = class_prediction_accuracy_vs_batchsize(1000, [1, 2, 4, 8, 16, 32, 64, 128, 256], "MNIST", False, "v2")
    #setting = class_prediction_accuracy_vs_batchsize(1000, [1, 2, 4, 8, 16, 32, 64, 128, 256], "CIFAR", True, "v2")
    #setting = class_prediction_accuracy_vs_batchsize(1000, [1, 2, 4, 8, 16, 32, 64, 128, 256], "CIFAR", False, "v2")
    run, path = load_json()
    #visualize_class_prediction_accuracy_vs_batchsize()
    #visualize_flawles_class_prediction_accuracy_vs_batchsize()
    #negativ_value_check()
    magnitude_check(run, path, adjusted=True)

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

def load_json():
    Tk().withdraw()
    filename = askopenfilename(initialdir="./results", defaultextension='.json',
                               filetypes=[('Json', '*.json')])

    with open(filename) as f:
        dump = OrderedDict(json.load(f))

    path = os.path.split(f.name)[0]

    return dump, path+"/"



if __name__ == '__main__':
    main()
