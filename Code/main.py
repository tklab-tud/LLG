from experiments import *
from visualize_experiment import *
from Dataloader import Dataloader


def main():
    ############## Build your attack here ######################
    ### Identifying value analysis ###
    """
    dataloader = Dataloader()
    setting = experiment(dataloader=dataloader,
                         list_datasets=["MNIST"],
                         list_bs=[1, 2, 4, 8, 16, 32, 64, 128, 256],
                         list_balanced=[True],
                         list_versions=["v2"],
                         n=10,
                         extent="predict",
                         trainsize=100,
                         trainsteps=0,
                         path=None
                         )
    """

    run, path = load_json()
    #negativ_value_check(run, path)

    # extent = "prediction" visualisation
    #magnitude_check(run, path, adjusted=True, balanced=True, version="v2")
    #magnitude_check(run, path, adjusted=False, balanced=True, version="v2")

    #visualize_class_prediction_accuracy_vs_batchsize(run, path, dataset="MNIST")
    visualize_flawles_class_prediction_accuracy_vs_batchsize(run, path)


    # visualize_good_fidelity()

    ############################################################
    print("Run finished")


if __name__ == '__main__':
    main()
