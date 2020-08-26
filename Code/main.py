from experiments import *
#from visualize_experiment import *
from Dataloader import Dataloader


def main():
    ############## Build your attack here ######################
    ### Identifying value analysis ###

    dataloader = Dataloader()
    setting = experiment(dataloader=dataloader,
                         list_datasets=["MNIST", "CIFAR"],
                         list_bs=[1, 2, 4, 8, 16, 32, 64, 128],
                         list_balanced=[True, False],
                         list_versions=["random", "v1", "v2"],
                         extent="predict",
                         n=100,
                         trainsize=100,
                         trainsteps=0,
                         path=None
                         )


    #run, path = load_json()

    # extent = "prepare" visualisation
    #negativ_value_check(run, path)


    # extent = "prediction" visualisation
    #magnitude_check(run, path, adjusted=True, balanced=True, dataset="MNIST", version="v2")
    #magnitude_check(run, path, adjusted=True, balanced=False, dataset="MNIST", version="v2")
    #visualize_class_prediction_accuracy_vs_batchsize(run, path)
    # visualize_flawles_class_prediction_accuracy_vs_batchsize(run, path )
    # visualize_prediction_accuracy_vs_training()
    #magnitude_check(run, path, adjusted=True, balanced=True)
    #magnitude_check(run, path, adjusted=False, balanced=True)


    # extent = "full" visualisation
    # visualize_good_fidelity()

    ############################################################
    print("Run finished")


if __name__ == '__main__':
    main()
