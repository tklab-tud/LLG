from experiments import *
from visualize_experiment import *
from Dataloader import Dataloader


def main():
    ############## Build your attack here ######################

    dlg_iterations = [100]
    datasets = ["MNIST"] #, "CIFAR", "CELEB-A-male", "SVHN"]
    # datasets = ["SVHN", "CELEB-A-male", "CIFAR", "MNIST"]
    v3 = {"MNIST": "v3-zero", "CIFAR": "v3-one", "CELEB-A-male": "v3-zero", "SVHN": "v3-random"}

    job = "custom-experiment"
    #job = "custom-visualize"

    dataset = "MNIST"
    # dataset = "CIFAR"
    # dataset = "CELEB-A-male"
    # dataset = "SVHN"

    if job == "custom-experiment":

        dataloader = Dataloader()
        for dataset in datasets:
            experiment(dataloader=dataloader,
                    list_datasets=[dataset],
                    list_bs=[8],
                    list_balanced=[False],
                    list_versions=["v2"],   # "v1"(LLG), "v2"(LLG+), "v3-zero", "v3-one", "v3-random", "dlg", "idlg"
                    n=100,                     # Amount of attacks
                    extent="predict",        # "victim_side", "predict", "reconstruct"
                    trainsize=0,#10000,             # Iterations per Trainstep
                    trainsteps=0,           # Number of Attack&Train cycles
                    path=None,
                    model="LeNet",
                    store_individual_gradients=False, # Will store the ~500 gradients connected to one output node and not just their sum
                    dlg_lr= 1, # learrate of (i)dlg image reconstruction
                    dlg_iterations=100, # amount of (i)dlg reconstruction iterations
                    log_interval=100000,  # Won't store each (i)dlg iteration's images but every n-th iteration's
                    store_composed_image = False, # storing dlg output images as composed image
                    store_separate_images = False, # storing dlg output images as seperate images
                    defenses=["dp"],
                    differential_privacy=True,
                    noise_type="normal",
                    noise_multiplier=0.1

                    )

    elif job == "custom-visualize":

        ########### Load an existing json an create graphs from it ##########

        run, path = load_json()
        # _, meta = get_meta(run)

        # num_files = 6
        # for i in range(num_files-1):
        #     run2, path2 = load_json()
        #     run2, meta2 = get_meta(run2, cut_meta=True)

        #     compare_meta(meta, meta2)

        #     # run = merge_runs(run, run2)
        #     # for models
        #     run = append_runs(run, run2)

        # magnitude_check plots a scatterplot of the gradients of a run.
        # gradient_type: "individual_gradients", "original_gradients", "adjusted_gradients"
        # Grads before summing up, after summing up and after adjustment
        # magnitude_check(run, path, gradient_type="individual_gradients", group_by="class")


        # negativ_value_check partitions the gradients into 4 categories: (non)present x sign
        # gradient_type: "individual_gradients", "original_gradients", "adjusted_gradients"
        #
        #negativ_value_check(run, path, gradient_type = "individual_gradients")

        # same_sign_check(run, path, dataset=None, balanced=None)
        # checks the split gradient_sum_sign x individual_grad_sign
        #same_sign_check(run, path)

        # Comparing accuracies

        # visualize_class_prediction_accuracy_vs_training(run, path, dataset="MNIST")
        # visualize_class_prediction_accuracy_vs_training(run, path, dataset="CIFAR")
        # visualize_class_prediction_accuracy_vs_training(run, path, dataset="CELEB-A-male")
        # visualize_class_prediction_accuracy_vs_training(run, path, dataset="SVHN")
        # for i in range(num_files):
        #     visualize_class_prediction_accuracy_vs_training(run, path, dataset=dataset, balanced=True, model_id=i)
        #     visualize_class_prediction_accuracy_vs_training(run, path, dataset=dataset, balanced=False, model_id=i)

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

    # Visualization Set 0
    elif job == "Set0-visualization":
        run, path = load_json()
        magnitude_check(run, path, gradient_type="original_gradients", group_by="bs", y_range=[-300, 400], dataset="MNIST", trainstep=0, list_bs=[2,8,32,128], balanced=True, legend_location="lower right")
        magnitude_check(run, path, gradient_type="adjusted_gradients", group_by="bs", y_range=[-300, 400], dataset="MNIST", trainstep=0, list_bs=[2,8,32,128], balanced=True, legend_location="lower right")
        heatmap(run, path, gradient_type="original_gradients", y_range=[-300, 400], dataset="MNIST", trainstep=0, list_bs=[2,8,32,128], balanced=True)

    # Visualization Set 1
    elif job == "Set1-visualization":
        visualize_class_prediction_accuracy_vs_batchsize(run, path, dataset=dataset, balanced=True)

    # Visualization Set 2
    elif job == "Set2-visualization":
        visualize_class_prediction_accuracy_vs_batchsize(run, path, dataset=dataset, balanced=False)

    # Visualization Set 3
    elif job == "Set3-visualization":
        visualize_class_prediction_accuracy_vs_training(run, path, dataset=dataset, balanced=True)

    # Visualization Set 4
    elif job == "Set4-visualization":
        visualize_class_prediction_accuracy_vs_training(run, path, dataset=dataset, balanced=False)

    # Visualization Set 5
    elif job == "Set5-visualization":
        visualize_class_prediction_accuracy_vs_batchsize(run, path, dataset=dataset, balanced=True, labels="model")
        visualize_class_prediction_accuracy_vs_batchsize(run, path, dataset=dataset, balanced=False, labels="model")

    # Visualization Set 6
    elif job == "Set6-visualization":
        visualize_class_prediction_accuracy_vs_batchsize(run, path, dataset=dataset, balanced=True, labels="noise_multiplier")
        visualize_class_prediction_accuracy_vs_batchsize(run, path, dataset=dataset, balanced=False, labels="noise_multiplier")
        # visualize_class_prediction_accuracy_vs_batchsize(run, path, dataset=dataset, balanced=True, labels="noise_type")
        # visualize_class_prediction_accuracy_vs_batchsize(run, path, dataset=dataset, balanced=False, labels="noise_type")

    # Visualization Set 7
    elif job == "Set7-visualization":
        visualize_class_prediction_accuracy_vs_batchsize(run, path, dataset=dataset, balanced=True, labels="threshold")
        visualize_class_prediction_accuracy_vs_batchsize(run, path, dataset=dataset, balanced=False, labels="threshold")

    else:
        print("Unknown job")


    ############################################################
    print("Run finished")


if __name__ == '__main__':
    main()
