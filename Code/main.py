from experiments import *
from visualize_experiment import *
from Dataloader import Dataloader


def main():
    ############## Build your attack here ######################
    ### Identifying value analysis ###
    """
    dataloader = Dataloader()
    experiment(dataloader=dataloader,
               list_datasets=["MNIST", "CIFAR"],
               list_bs=[1, 2, 4, 8, 16, 32, 64, 128, 256],
               list_balanced=[True, False],
               list_versions=["random", "v1", "v2"],
               n=100,
               extent="predict",
               trainsize=100,
               trainsteps=0,
               path=None
               )
    """
    run, path = load_json()
    # negativ_value_check(run, path)

    #magnitude_check(run, path, adjusted=False, balanced=True, version="v2", dataset="CIFAR", list_bs=[2, 8, 32, 128])
    #magnitude_check(run, path, adjusted=True, balanced=True, version="v2", dataset="CIFAR", list_bs=[2, 8, 32, 128])

    heatmap(run, path, adjusted=True, balanced=True, version="v2", dataset="CIFAR", list_bs=[2, 8, 32, 128])

    # visualize_class_prediction_accuracy_vs_batchsize(run, path, dataset="CIFAR")
    # visualize_flawles_class_prediction_accuracy_vs_batchsize(run, path, dataset="CIFAR")

    # visualize_good_fidelity()

    ############################################################
    print("Run finished")


if __name__ == '__main__':
    main()
