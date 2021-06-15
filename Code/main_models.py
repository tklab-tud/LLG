from experiments import *
from visualize_experiment import *
from Dataloader import Dataloader


def main():
    ############## Build your attack here ######################

    dlg_iterations = [100,200,400,800]
    models = ["LeNet", "LeNetNew", "MLP", "ResNet"]

    job = "custom-experiment"
    #job = "custom-visualize"

    if job == "custom-experiment":

        dataloader = Dataloader()
        for model in models:
            experiment(dataloader=dataloader,
                    list_datasets=["MNIST"],
                    list_bs=[1,2,4,8,16,32,64,128],
                    list_balanced=[True, False],
                    list_versions=["v2"],   # "v1"(LLG), "v2"(LLG+), "v3-zero", "v3-one", "v3-random", "dlg", "idlg"
                    n=100,                     # Amount of attacks
                    extent="predict",        # "victim_side", "predict", "reconstruct"
                    trainsize=100,             # Iterations per Trainstep
                    trainsteps=100,           # Number of Attack&Train cycles
                    path=None,
                    model=model,
                    store_individual_gradients=False, # Will store the ~500 gradients connected to one output node and not just their sum
                    dlg_lr= 1, # learrate of (i)dlg image reconstruction
                    dlg_iterations= 100, # amount of (i)dlg reconstruction iterations
                    log_interval=10,  # Won't store each (i)dlg iteration's images but every n-th iteration's
                    store_composed_image = False, # storing dlg output images as composed image
                    store_separate_images = False, # storing dlg output images as seperate images

                    )

    elif job == "custom-visualize":

        ########### Load an existing json an create graphs from it ##########

        run, path = load_json()


        # magnitude_check plots a scatterplot of the gradients of a run.
        # gradient_type: "individual_gradients", "original_gradients", "adjusted_gradients"
        # Grads before summing up, after summing up and after adjustment
        # magnitude_check(run, path, gradient_type="individual_gradients", group_by_class=True)


        # negativ_value_check partitions the gradients into 4 categories: (non)present x sign
        # gradient_type: "individual_gradients", "original_gradients", "adjusted_gradients"
        #
        #negativ_value_check(run, path, gradient_type = "individual_gradients")

        # same_sign_check(run, path, dataset=None, balanced=None)
        # checks the split gradient_sum_sign x individual_grad_sign
        #same_sign_check(run, path)

        # Comparing accuracies
        visualize_class_prediction_accuracy_vs_batchsize(run, path)


    # Set 1 and 2 generation
    elif job == "Untrained_MNIST-experiment":
        print("todo")
    elif job == "Untrained_CIFAR-experiment":
        print("todo")
    elif job == "Untrained_CELEB-A-experiment":
        print("todo")
    elif job == "Untrained_CIFAR-experiment":
        print("todo")

    # Set 3 generation
    elif job == "Trained_MNIST-experiment":
        print("todo")
    elif job == "Trained_CIFAR-experiment":
        print("todo")
    elif job == "Trained_CELEB-A-experiment":
        print("todo")
    elif job == "Trained_CIFAR-experiment":
        print("todo")

    #Visualization Set 1
    elif job == "Set1-visualization":
        visualize_class_prediction_accuracy_vs_batchsize(run, path)

    # Visualization Set 2
    elif job == "Set2-visualization":
        visualize_class_prediction_accuracy_vs_batchsize(run, path)

    # Visualization Set 3
    elif job == "Set3-visualization":
        visualize_class_prediction_accuracy_vs_training(run, path)

    # Visualization Set 4
    elif job == "Set4-visualization":
        visualize_class_prediction_accuracy_vs_training(run, path)


    else:
        print("Unknown job")


    ############################################################
    print("Run finished")


if __name__ == '__main__':
    main()
