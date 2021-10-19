from experiments import *
from visualize_experiment import *
from Dataloader import Dataloader
import argparse
import time


def args_parser():
    parser = argparse.ArgumentParser(description="Arguments for LLG Experiment")

    # required arguments
    parser.add_argument('-s', '--set', type=int, default=2,
                            help='experiment set')
    parser.add_argument('-p', '--plot', type=int, default=None,
                            help='number of files to be ploted')
    parser.add_argument('-j', '--job', type=str, default='experiment',
                            help='job to execute. either "experiment" or "visualize".')

    # optional arguments
    parser.add_argument('-g', '--gpu_id', type=int, default=0,
                            help='cuda_id to use, if available')

    args = parser.parse_args()

    if args.plot != None:
        args.job = "visualize"

    return args


def main():
    # load parameters
    args = args_parser()
    job = args.job

    # visualization parameters
    # specify the number of json files you want to select for plotting
    num_files = args.plot

    # experiment parameters
    experiment_set = args.set
    cuda_id = args.gpu_id

    # defaults to fill vars, when unused
    extent="predict"
    train_lr = 0.1
    defenses = ["none"]
    differential_privacy = False
    noise_type = "normal"
    noise_multipliers = [0.0]
    max_norms = [None]
    compression = False
    thresholds = [0.0]
    n = 100
    local_training = False
    federated = False
    num_users = 1

    # with exception of set 1 and 3 all sets use non-IID
    balanced = False

    if experiment_set in [1, 3]:
        balanced = True

    # with exception of set 1, 2, 3, and 4 all sets use MNIST and LLG+
    dataset = "MNIST"
    version = "v2"

    # with exception of set 3 and 4 all sets use all batch sizes and don't train
    list_bs = [1, 2, 4, 8, 16, 32, 64, 128]
    trainsize = 0
    trainsteps = 0
    local_iterations = 1
    id = "set_"+str(experiment_set)

    v3 = {"MNIST": "v3-zero", "CIFAR": "v3-one", "CELEB-A-male": "v3-zero", "CELEB-A-hair": "v3-zero",
          "SVHN": "v3-random"}

    # FedAvg
    fedAvg = True
    if fedAvg:
        local_training = True
        local_iterations = 10
    else:
        local_training = False
        local_iterations = 1

    if experiment_set == 0:
        local_iterations = 4
        n = 1
        list_bs = [1,4,8]
        dataset = "MNIST"
        balanced = True
        #version = ["v1", "v2", v3[dataset], "dlg", "random"]
        version = ["v2", "dlg"]
        model = ["ResNet"]
        local_training = False

    if experiment_set in [1, 2, 3, 4]:
        # TODO: run all DATASETS separately ("in parallel")
        dataset = "MNIST"
        # dataset = "EMNIST"
        # dataset = "FEMNIST"
        # dataset = "FEMNIST-digits"
        # dataset = "CIFAR"
        # dataset = "CELEB-A-male"
        # dataset = "CELEB-A-hair"
        # dataset = "SVHN"
        # versions in order of importance
        # TODO: run all VERSIONS separately per DATASET ("in parallel")
        # version = "v2"
        version = "random"
        # version = "v1"
        # version = v3[dataset]
        # TODO: this will take forever, only run when absolutely necessary
        # FIXME: don't run this for set 3&4 (trained) only for 1&2 (untrained)
        # version = "dlg"
        idx = dataset+"_"+version

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
        model = "LeNet"       # old LeNet = ConvNet
        # model = "NewNewLeNet"  # new LeNet = LeNet
        # model = "ResNet"      # ResNet
        # model = "MLP"         # FCNN

        version = "dlg"
        # version ="v2"
        idx = model+"_"+version

    # TODO: You can use the old LeNet run form set 5 as a baseline for sets 5, 6, and 7

    # Set 6 generation
    if experiment_set == 6:
        defenses = ["dp"]
        differential_privacy = True
        # TODO: run all NOISE TYPES separately ("in parallel")
        noise_type = "normal"
        # noise_type = "laplace"
        # noise_type = "exponential"

        noise_multipliers = [0.01, 0.1, 1]
        idx = noise_multipliers[0]

    # Set 7 generation
    if experiment_set == 7:
        defenses = ["compression"]
        compression = True
        thresholds = [0.2, 0.4, 0.8]
        idx = thresholds[0]

    # Set 8 generation
    # differential privacy = clipping + noise
    if experiment_set == 8:
        defenses = ["dp"]
        differential_privacy = True
        # TODO: run all NOISE TYPES separately ("in parallel")
        noise_type = "normal"
        # noise_type = "laplace"
        # noise_type = "exponential"

        # variance = 1
        noise_multipliers = [0.1]
        # epsilon | beta
        # max_norms = [1, 5, 10]
        max_norms = [10]
        idx = max_norms[0]

    if experiment_set == 9:
        n=1
        extent="victim_side"
        list_bs = [8]
        trainsize = int(10/local_iterations)
        trainsteps = 100
        federated = True
        num_users = 100
        id = "base"
        idx = "FedAvg" if local_training else "FedSGD"
        # TEMP: defense experiments with training
        # defenses = ["dp"]
        # differential_privacy = True
        # noise_type = "normal"
        # noise_multipliers = [0.01, 0.1, 1]
        # noise_multipliers = [0.01]
        # id = "noise"
        # idx = noise_multipliers[0]
        # max_norms = [1, 5, 10]
        # max_norms = [1]
        # noise_multipliers = [0.1]
        # id = "dp"
        # idx = max_norms[0]
        defenses = ["compression"]
        compression = True
        # thresholds = [0.2, 0.4, 0.8]
        thresholds = [0.2]
        id = "comp"
        idx = thresholds[0]
        version = "v2"
        # version = "random"
        # version = "v1"
        # version = v3[dataset]
        # version = "dlg"
        dataset = "MNIST"
        # dataset = "CIFAR"
        # dataset = "CELEB-A-hair"
        # dataset = "SVHN"
        model = "LeNet"       # old LeNet = ConvNet
        # model = "NewNewLeNet"  # new LeNet = LeNet
        # model = "ResNet"      # ResNet
        # model = "MLP"         # FCNN

    if experiment_set == -1:
        n = 1000
        versions = ["v2", "random"]
    elif isinstance(version, list):
        versions = version
    else:
        versions = [version]

    if experiment_set == 10:
        local_iterations_list = [1, 10, 100, 1000]

    start = time.time()

    if job == "experiment":
        dataloader = Dataloader()
        for noise_multiplier in noise_multipliers:
            for max_norm in max_norms:
                if max_norm != None and max_norm > 0:
                    noise_multiplier = noise_multiplier / max_norm
                for threshold in thresholds:
                # for local_iterations in local_iterations_list:
                    # threshold = 0.0
                    experiment(dataloader=dataloader,
                               list_datasets=dataset if isinstance(dataset, list) else [dataset],
                               list_bs=list_bs,
                               list_balanced=[balanced],
                               list_versions=versions,
                               # "v1"(LLG), "v2"(LLG+), "v3-zero", "v3-one", "v3-random", "dlg", "idlg"
                               n=n,  # Amount of attacks
                               extent=extent,  # "victim_side", "predict", "reconstruct"
                               trainsize=trainsize,  # Iterations per Trainstep
                               trainsteps=trainsteps,  # Number of Attack&Train cycles
                               train_lr=train_lr,
                               federated=federated,
                               num_users=num_users,
                               path="results/{}/{}/{}/{}/".format("FedAvg" if local_training else "FedSGD", id, idx, str(datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S"))),
                               model=model,
                               store_individual_gradients=False,
                               # Will store the ~500 gradients connected to one output node and not just their sum
                               dlg_lr=1,  # learrate of (i)dlg image reconstruction
                               dlg_iterations=100,  # amount of (i)dlg reconstruction iterations
                               log_interval=100000,
                               # Won't store each (i)dlg iteration's images but every n-th iteration's
                               store_composed_image=False,  # storing dlg output images as composed image
                               store_separate_images=False,  # storing dlg output images as seperate images
                               cuda_id=cuda_id,
                               defenses=defenses,
                               differential_privacy=differential_privacy,
                               noise_type=noise_type,
                               noise_multiplier=noise_multiplier,
                               max_norm=max_norm,
                               compression=compression,
                               threshold=threshold,
                               local_iterations=local_iterations,
                               local_training=local_training
                               )

        end = time.time()
        duration = end - start
        print(duration)

    elif job == "visualize":

        ########### Load an existing json an create graphs from it ##########

        run, path = load_json()
        _, meta = get_meta(run)

        for i in range(num_files - 1):
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
        # negativ_value_check(run, path, gradient_type = "individual_gradients")

        # same_sign_check(run, path, dataset=None, balanced=None)
        # checks the split gradient_sum_sign x individual_grad_sign
        # same_sign_check(run, path)

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
            visualize_class_prediction_accuracy_vs_batchsize(run, path, labels=["version", "model"])
            #magnitude_check(run, path, gradient_type="original_gradients", group_by="bs")
            #magnitude_check(run, path, gradient_type="adjusted_gradients", group_by="bs")


        # Visualization Set 1
        elif experiment_set == 1:
            visualize_class_prediction_accuracy_vs_batchsize(run, path, dataset=dataset, balanced=True, width=4.8)

        # Visualization Set 2
        elif experiment_set == 2:
            # default set 2: attack versions
            visualize_class_prediction_accuracy_vs_batchsize(run, path, dataset=dataset, balanced=False, width=4.8, labels=["version"])
            # local iteration comparison
            # visualize_class_prediction_accuracy_vs_batchsize(run, path, dataset=dataset, balanced=False, labels=["local_training"])
            # local iterations & attack versions
            # visualize_class_prediction_accuracy_vs_batchsize(run, path, dataset=dataset, balanced=False, width=4.8, labels=["version", "local_training"])

        # Visualization Set 1 & 2
        elif experiment_set == 12:
            visualize_class_prediction_accuracy_vs_batchsize(run, path, dataset=dataset, balanced=True, width=4.8)
            visualize_class_prediction_accuracy_vs_batchsize(run, path, dataset=dataset, balanced=False, width=4.8)
            # location="lower right"

        # Visualization Set 3
        elif experiment_set == 3:
            visualize_class_prediction_accuracy_vs_training(run, path, dataset=dataset, balanced=True, width=4.8)

        # Visualization Set 4
        elif experiment_set == 4:
            visualize_class_prediction_accuracy_vs_training(run, path, dataset=dataset, balanced=False)
                                                            # width=4.8)  # , location="lower right"
            # visualize_class_prediction_accuracy_vs_training(run, path, dataset=dataset, balanced=False) #, width=4.8) # , location="lower right"

        # Visualization Set 3 & 4
        elif experiment_set == 34:
            visualize_class_prediction_accuracy_vs_training(run, path, dataset=dataset, balanced=True, width=4.8)
            visualize_class_prediction_accuracy_vs_training(run, path, dataset=dataset, balanced=False, width=4.8)

        # Visualization Set 5
        elif experiment_set == 5:
            # visualize_class_prediction_accuracy_vs_batchsize(run, path, dataset=dataset, balanced=True, labels="model")
            visualize_class_prediction_accuracy_vs_batchsize(run, path, dataset=dataset, balanced=False,
                                                             labels=["version", "model"])

        # Visualization Set 6
        elif experiment_set == 6:
            # visualize_class_prediction_accuracy_vs_batchsize(run, path, dataset=dataset, balanced=True, labels="noise_multiplier")
            visualize_class_prediction_accuracy_vs_batchsize(run, path, dataset=dataset, balanced=False,
                                                             labels="noise_multiplier")
            # visualize_class_prediction_accuracy_vs_batchsize(run, path, dataset=dataset, balanced=True, labels="noise_type")
            # visualize_class_prediction_accuracy_vs_batchsize(run, path, dataset=dataset, balanced=False, labels="noise_type")

        # Visualization Set 7
        elif experiment_set == 7:
            # visualize_class_prediction_accuracy_vs_batchsize(run, path, dataset=dataset, balanced=True, labels="threshold")
            visualize_class_prediction_accuracy_vs_batchsize(run, path, dataset=dataset, balanced=False,
                                                             labels="threshold")

        # Visualization Set 8
        elif experiment_set == 8:
            visualize_class_prediction_accuracy_vs_batchsize(run, path, dataset=dataset, balanced=False,
                                                             labels="max_norm")
            # visualize_class_prediction_accuracy_vs_batchsize(run, path, dataset=dataset, balanced=True, labels="noise_type")
            # visualize_class_prediction_accuracy_vs_batchsize(run, path, dataset=dataset, balanced=False, labels="noise_type")

        # Visualization Set 9
        elif experiment_set == 9:
            visualize_class_prediction_accuracy_vs_training(run, path, dataset=dataset, balanced=False,
                                                            width=4.8)
            # visualize_class_prediction_accuracy_vs_training(run, path, dataset=dataset, balanced=False,
            #                                                 labels=["num_users"], width=4.8)
            # visualize_class_prediction_accuracy_vs_training(run, path, dataset=dataset, balanced=False,
            #                                                 labels=["local_iterations"], width=4.8)
            # visualize_class_prediction_accuracy_vs_training(run, path, dataset=dataset, balanced=False,
            #                                                 labels=["noise_multiplier"], width=4.8)

        elif experiment_set == 10:
            visualize_class_prediction_accuracy_vs_batchsize(run, path, balanced=False, width=4.8, labels=["dataset", "local_training"], fontsize=14)

        # Gradient Plots
        if experiment_set == -1:
            y_range = [-330, 300]
            magnitude_check(run, path, gradient_type="original_gradients", balanced="True", dataset="MNIST",
                            version="v2", list_bs=[2, 8, 32, 128], group_by="bs", y_range=y_range)
            magnitude_check(run, path, gradient_type="adjusted_gradients", balanced="True", dataset="MNIST",
                            version="v2", list_bs=[2, 8, 32, 128], group_by="bs", y_range=y_range)
            heatmap(run, path, gradient_type="original_gradients", balanced="True", dataset="MNIST",
                    list_bs=[2, 8, 32, 128], y_range=y_range)

    else:
        print("Unknown job")

    ############################################################
    print("Run finished")


if __name__ == '__main__':
    main()
