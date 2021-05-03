from experiments import *
from visualize_experiment import *
from Dataloader import Dataloader


def main():
    ############## Build your attack here ######################

    experiment_set = 2

    # visualization parameters
    job = "visualize"
    # specify the number of json files you want to select for plotting
    num_files = 5

    # experiment parameters
    # FIXME: outcomment this line if you want the run the visualization
    job = "experiment"

    # defaults to fill vars, when unused
    train_lr = 0.1
    defenses = ["none"]
    differential_privacy = False
    noise_type = "normal"
    noise_multipliers = [0.0]
    max_norms = [None]
    compression = False
    thresholds = [0.0]
    n = 100

    # with exception of set 1 and 3 all sets use non-IID
    balanced = False

    if experiment_set in [1, 3]:
        balanced = True

    # with exception of set 1, 2, 3, and 4 all sets use MNIST and LLG+
    dataset = "MNIST"
    version = "v2"

    v3 = {"MNIST": "v3-zero", "CIFAR": "v3-one", "CELEB-A-male": "v3-zero", "CELEB-A-hair": "v3-zero", "SVHN": "v3-random"}

    if experiment_set == 0:
        n=3
        dataset = "CELEB-A-hair"
        version = "v2"
        model = "LeNet"

    if experiment_set in [1, 2, 3, 4]:
        # TODO: run all DATASETS separately ("in parallel")
        dataset = "MNIST"
        # dataset = "CIFAR"
        # dataset = "CELEB-A-male"
        dataset = "CELEB-A-hair"
        # dataset = "SVHN"
        # versions in order of importance
        # TODO: run all VERSIONS separately per DATASET ("in parallel")
        version = "v2"
        # version = "random"
        # version = "v1"
        # version = v3[dataset]
        # TODO: this will take forever, only run when absolutely necessary
        # FIXME: don't run this for set 3&4 (trained) only for 1&2 (untrained)
        # version = "dlg"

    # with exception of set 3 and 4 all sets use all batch sizes and don't train
    list_bs = [1,2,4,8,16,32,64,128]
    trainsize = 0
    trainsteps = 0

    # Set 3 and 4 generation
    if experiment_set in [3, 4]:
        list_bs = [8]
        trainsize = 100
        trainsteps = 100
        # TODO: ask Aidmar about the LEARNING RATE
        # we use 0.1 as the default, but we also ran some experiments with 0.01
        # I don't remember which results we ended up using for the paper.
        # train_lr = 0.01

    # with exception of set 5 all sets use the ConvNet model (old LeNet)
    model = "LeNet"

    # Set 5 generation
    if experiment_set == 5:
        # TODO: run all MODELS separately ("in parallel")
        # model = "LeNet"       # old LeNet = ConvNet
        model = "NewNewLeNet"   # new LeNet = LeNet
        # model = "ResNet"      # ResNet
        # model = "MLP"         # FCNN

    # TODO: You can use the old LeNet run form set 5 as a baseline for sets 5, 6, and 7

    # Set 6 generation
    if experiment_set == 6:
        defenses = ["dp"]
        differential_privacy = True
        # TODO: run all NOISE TYPES separately ("in parallel")
        noise_type = "normal"
        # noise_type = "laplace"
        # noise_type = "exponential"
        noise_multipliers = [0.0, 0.1, 0.01, 0.001, 0.0001]
        noise_multipliers = [noise_multipliers[1]]
        # noise_multipliers = [0.0, 0.1] #, 0.25, 0.5, 1.0]
        # version = "v1"
        # version = v3[dataset]

    # Set 7 generation
    if experiment_set == 7:
        defenses = ["compression"]
        compression = True
        thresholds = [0.1, 0.2, 0.4, 0.8]

    # Set 8 generation
    # differential privacy = clipping + noise
    if experiment_set == 8:
        defenses = ["dp"]
        differential_privacy = True
        # TODO: run all NOISE TYPES separately ("in parallel")
        noise_type = "normal"
        # noise_type = "laplace"
        # noise_type = "exponential"
        noise_multipliers = [0.0, 0.1, 0.01, 0.001, 0.0001]
        max_norms = [None, 0.05, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 5, 10]

    if job == "experiment":
        dataloader = Dataloader()
        for noise_multiplier in noise_multipliers:
            for max_norm in max_norms:
                if max_norm != None and max_norm > 0:
                    noise_multiplier = noise_multiplier/max_norm
                for threshold in thresholds:
                    experiment(dataloader=dataloader,
                            list_datasets=[dataset],
                            list_bs=list_bs,
                            list_balanced=[balanced],
                            list_versions=[version],   # "v1"(LLG), "v2"(LLG+), "v3-zero", "v3-one", "v3-random", "dlg", "idlg"
                            n=n,                     # Amount of attacks
                            extent="predict",        # "victim_side", "predict", "reconstruct"
                            trainsize=trainsize,             # Iterations per Trainstep
                            trainsteps=trainsteps,           # Number of Attack&Train cycles
                            train_lr=train_lr,
                            path=None,
                            model=model,
                            store_individual_gradients=False, # Will store the ~500 gradients connected to one output node and not just their sum
                            dlg_lr= 1, # learrate of (i)dlg image reconstruction
                            dlg_iterations=100, # amount of (i)dlg reconstruction iterations
                            log_interval=100000,  # Won't store each (i)dlg iteration's images but every n-th iteration's
                            store_composed_image = False, # storing dlg output images as composed image
                            store_separate_images = False, # storing dlg output images as seperate images
                            defenses=defenses,
                            differential_privacy=differential_privacy,
                            noise_type=noise_type,
                            noise_multiplier=noise_multiplier,
                            max_norm=max_norm,
                            compression=compression,
                            threshold=threshold
                            )

    elif job == "visualize":

        ########### Load an existing json an create graphs from it ##########

        run, path = load_json()
        _, meta = get_meta(run)

        for i in range(num_files-1):
            run2, path2 = load_json()
            run2, meta2 = get_meta(run2, cut_meta=True)

            compare_meta(meta, meta2)

            # run = merge_runs(run, run2)
            # for models
            run = append_runs(run, run2)

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

        # Visualization Set 0
        if experiment_set == 0:
            visualize_class_prediction_accuracy_vs_batchsize(run, path)


        # Visualization Set 1
        elif experiment_set == 1:
            visualize_class_prediction_accuracy_vs_batchsize(run, path, dataset=dataset, balanced=True)

        # Visualization Set 2
        elif experiment_set == 2:
            visualize_class_prediction_accuracy_vs_batchsize(run, path, dataset=dataset, balanced=False)

        # Visualization Set 3
        elif experiment_set == 3:
            visualize_class_prediction_accuracy_vs_training(run, path, dataset=dataset, balanced=True)

        # Visualization Set 4
        elif experiment_set == 4:
            visualize_class_prediction_accuracy_vs_training(run, path, dataset=dataset, balanced=False)

        # Visualization Set 5
        elif experiment_set == 5:
            visualize_class_prediction_accuracy_vs_batchsize(run, path, dataset=dataset, balanced=True, labels="model")
            visualize_class_prediction_accuracy_vs_batchsize(run, path, dataset=dataset, balanced=False, labels="model")

        # Visualization Set 6
        elif experiment_set == 6:
            # visualize_class_prediction_accuracy_vs_batchsize(run, path, dataset=dataset, balanced=True, labels="noise_multiplier")
            visualize_class_prediction_accuracy_vs_batchsize(run, path, dataset=dataset, balanced=False, labels="noise_multiplier")
            # visualize_class_prediction_accuracy_vs_batchsize(run, path, dataset=dataset, balanced=True, labels="noise_type")
            # visualize_class_prediction_accuracy_vs_batchsize(run, path, dataset=dataset, balanced=False, labels="noise_type")

        # Visualization Set 7
        elif experiment_set == 7:
            visualize_class_prediction_accuracy_vs_batchsize(run, path, dataset=dataset, balanced=True, labels="threshold")
            visualize_class_prediction_accuracy_vs_batchsize(run, path, dataset=dataset, balanced=False, labels="threshold")

        # Visualization Set 8
        elif experiment_set == 8:
            visualize_class_prediction_accuracy_vs_batchsize(run, path, dataset=dataset, balanced=False, labels="max_norm")
            # visualize_class_prediction_accuracy_vs_batchsize(run, path, dataset=dataset, balanced=True, labels="noise_type")
            # visualize_class_prediction_accuracy_vs_batchsize(run, path, dataset=dataset, balanced=False, labels="noise_type")

    else:
        print("Unknown job")


    ############################################################
    print("Run finished")


if __name__ == '__main__':
    main()
