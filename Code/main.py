from experiments import *
from visualize_experiment import *
from Dataloader import Dataloader


def main():
    ############## Build your attack here ######################
    ### Identifying value analysis ###
    """
    dataloader = Dataloader()
    experiment(dataloader=dataloader,
               list_datasets=["MNIST"],
               list_bs=[2, 8, 32, 128],
               list_balanced=[True],
               list_versions=["v2", "idlg", "dlg"],
               n=1,
               extent="predict",
               trainsize=100,
               trainsteps=0,
               path=None,
               reconstruction_steps=100
               )
    """
    run, path = load_json()
    # negativ_value_check(run, path)

    #magnitude_check(run, path, adjusted=True, balanced=True, version="v2", dataset="MNIST", list_bs=[2, 8, 32, 128])
    #magnitude_check(run, path, adjusted=False, balanced=True, version="v2", dataset="MNIST", list_bs=[2, 8, 32, 128])
    #magnitude_check(run, path, adjusted=True, balanced=True, version="v2", dataset="CIFAR", list_bs=[2, 8, 32, 128])

    #heatmap(run, path, adjusted=True, balanced=True, version="v2", dataset="MNIST", list_bs=[2,8,32,128])
    #heatmap(run, path, adjusted=False, balanced=True, version="v2", dataset="MNIST", list_bs=[2,8,32,128])

    #pearson_check(run, path, version="v2")

    # visualize_class_prediction_accuracy_vs_batchsize(run, path, dataset="CIFAR")
    # visualize_flawles_class_prediction_accuracy_vs_batchsize(run, path, dataset="CIFAR")

    #visualize_good_fidelity(run, path, [0.1, 0.05, 0.01, 0.005, 0.001], 4, True)

    ############################################################
    print("Run finished")


if __name__ == '__main__':
    main()
