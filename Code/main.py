from experiments import *
from visualize_experiment import *
from Dataloader import Dataloader


def main():
    ############## Build your attack here ######################
    ### Identifying value analysis ###
    """
    dataloader = Dataloader()
    experiment(dataloader=dataloader,
               list_datasets=["CIFAR"],
               list_bs=[8],
               list_balanced=[True],
               list_versions=["v1", "v2", "random"],
               n=100,
               extent="predict",
               trainsize=100,
               trainsteps=100,
               path=None,
               reconstruction_steps=100
               )
    """
    run, path = load_json()

    #visualize_class_prediction_accuracy_vs_batchsize(run, path)

    #negativ_value_check(run, path, dataset="MNIST", balanced=True)
    #negativ_value_check(run, path, dataset="MNIST", balanced=False)
    #negativ_value_check(run, path, dataset="CIFAR", balanced=True)
    #negativ_value_check(run, path, dataset="CIFAR", balanced=False)
    #pearson_check(run, path, version="v2")

    """
    for adjusted in [True, False]:
        for balanced in [True, False]:
            for dataset in ["MNIST", "CIFAR"]:
                magnitude_check(run, path, adjusted=adjusted, balanced=balanced, version="v2", dataset=dataset, list_bs=[2, 8, 32, 128])
    """


    #heatmap(run, path, adjusted=False, balanced=True, version="v2", dataset="MNIST", list_bs=[32])

    #visualize_hellinger_vs_batchsize(run, path)


    """  
    for dataset in ["MNIST", "CIFAR"]:
        visualize_class_prediction_accuracy_vs_batchsize(run, path , dataset=dataset)
        visualize_flawles_class_prediction_accuracy_vs_batchsize(run, path, dataset=dataset)
    """



    visualize_class_prediction_accuracy_vs_training(run, path, train_step_stop=100)

    #visualize_good_fidelity(run, path, [0.1, 0.05, 0.01, 0.005, 0.001], 4, True)




    ############################################################
    print("Run finished")


if __name__ == '__main__':
    main()
