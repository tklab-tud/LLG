from experiments import *
from visualize_experiment import *
from Dataloader import Dataloader


def main():
    ############## Build your attack here ######################

    new = True
    new = False

    if new:

        dataloader = Dataloader()
        experiment(dataloader=dataloader,
                   list_datasets=["CELEB-A-male"],
                   list_bs=[1,2,4,8,16,32,64,128,256],
                   list_balanced=[False],
                   list_versions=["v2"],
                   n=1,
                   extent="predict",
                   trainsize=0,
                   trainsteps=0,
                   path=None,
                   reconstruction_steps=0
                   )
    else:

        ########### Load an existing json an create graphs from it ##########

        run, path = load_json()




        ### Some examples below ###

        #visualize_class_prediction_accuracy_vs_batchsize(run, path)

        #negativ_value_check(run, path, dataset="MNIST", balanced=True)
        #negativ_value_check(run, path, dataset="MNIST", balanced=False)
        #negativ_value_check(run, path, dataset="CIFAR", balanced=True)
        #negativ_value_check(run, path, dataset="CIFAR", balanced=False)
        #pearson_check(run, path, version="v2")

        """
        for dataset in ["MNIST", "CIFAR", "CELEB-A-male"]:
            for adjusted in [True, False]:
                magnitude_check(run, path, adjusted=adjusted, balanced=True, dataset=dataset)
        """

        for adjusted in [True, False]:
            magnitude_check(run, path, adjusted=adjusted)

        #heatmap(run, path, adjusted=False, balanced=True, version="v2", dataset="MNIST", list_bs=[32])

        #visualize_hellinger_vs_batchsize(run, path)


        #"""
        for dataset in ["CELEB-A-male"]:
            visualize_class_prediction_accuracy_vs_batchsize(run, path, dataset=dataset)
            visualize_flawles_class_prediction_accuracy_vs_batchsize(run, path, dataset=dataset)
        #"""

        #visualize_class_prediction_accuracy_vs_training(run, path, train_step_stop=1000)

        #visualize_good_fidelity(run, path, [0.1, 0.05, 0.01, 0.005, 0.001], 4, True)




    ############################################################
    print("Run finished")


if __name__ == '__main__':
    main()
