import torch
import torch.nn as nn
import numpy as np

from net import Net1, Net2, weights_init

def predict_setting(setting):
    # some abbreviations
    parameter = setting.parameter
    targets = setting.target
    device = setting.device
    train_dataset = setting.train_dataset
    model = setting.model

    # prepare attacked batch's data and label on device
    orig_data = torch.Tensor(parameter["batch_size"], parameter["channel"], parameter["shape_img"][0],
                             parameter["shape_img"][1]).to(device)
    orig_label = torch.Tensor(parameter["batch_size"]).long().to(device).view(parameter["batch_size"], )

    # fill missing targets if underspecified
    for i in range(parameter["batch_size"] - len(targets)):
        targets.append(np.random.randint(0, parameter["num_classes"]))

    # get one example for target class and put it into the attacked batch
    ids = []
    for i_t, target in enumerate(targets[:parameter["batch_size"]]):
        # searching for a sample with target label
        for i_s, sample in enumerate(train_dataset):
            # does this sample have the right label and was not used before?
            if sample[1] == target and not ids.__contains__(i_s):
                ids.append(i_s)
                orig_data[i_t] = sample[0]
                orig_label[i_t] = sample[1]
                break

    # Check if enough sample were found and cut off sample if target list was to long
    if len(ids) < parameter["batch_size"]:
        exit("Could not find enough samples in the dataset")


    # calculate orig gradients
    orig_out = model(orig_data)
    criterion = nn.CrossEntropyLoss().to(device)
    y = criterion(orig_out, orig_label)
    gradient = torch.autograd.grad(y, model.parameters())
    gradient_list = list((_.detach().clone() for _ in gradient))

    return prediction(parameter, gradient_list, model, orig_data, orig_label, device)




def prediction(parameter, gradient_list, model, orig_data, orig_label, device):
    idlg_pred = []

    # Classic way from the authors repository does not allow bs <> 1
    if parameter["prediction"] == "classic":
        if parameter["batch_size"] == 1:
            idlg_pred.append(torch.argmin(torch.sum(gradient_list[-2], dim=-1), dim=-1).detach().reshape(
                (1,)).requires_grad_(False))
        else:
            exit("classic prediction does not support batch_size <> 1")

    elif parameter["prediction"] == "random":
        for _ in range(parameter["batch_size"]):
            idlg_pred.append(np.random.randint(0, parameter["num_classes"]))

    # Simplified Way as described in the paper
    elif parameter["prediction"] == "simplified":
        # creating bs 1 model for single prediction
        parameter_bs1 = parameter.copy()
        parameter_bs1["batch_size"] = 1

        if parameter["model"] == 1:
            model_bs1 = Net1(parameter_bs1)
            model_bs1.apply(weights_init)
        elif parameter["model"] == 2:
            model_bs1 = Net2(parameter_bs1)

        model_bs1 = model.to(device)

        for i_s, sample in enumerate(orig_data):
            orig_out = model_bs1(sample.view(1, 1, 28, 28))
            sample_label = torch.split(orig_label, 1)[i_s]
            criterion = nn.CrossEntropyLoss().to(device)
            y = criterion(orig_out, sample_label)
            gradient = torch.autograd.grad(y, model_bs1.parameters())
            gradient_list = list((_.detach().clone() for _ in gradient))
            idlg_pred.append(torch.argmin(torch.sum(gradient_list[-2], dim=-1), dim=-1).detach().reshape(
                (1,)).requires_grad_(False))


    # Version 1 improvement suggestion
    elif parameter["prediction"] == "v1":
        gradients_for_prediction = torch.sum(gradient_list[-2], dim=-1).clone()
        candidates = []
        idlg_pred = []
        mean = 0

        # filter negative values
        for i_cg, class_gradient in enumerate(gradients_for_prediction):
            if class_gradient < 0:
                candidates.append((i_cg, class_gradient))
                mean += class_gradient

        # mean value
        mean /= parameter["batch_size"]

        # save predictions
        for (i_c, _) in candidates:
            idlg_pred.append(i_c)

        # predict the rest
        for _ in range(parameter["batch_size"] - len(idlg_pred)):
            # add minimal candidat, likely to be doubled, to prediction
            min_id = torch.argmin(gradients_for_prediction)
            idlg_pred.append(min_id)

            # add the mean value of one accurance to the candidate
            gradients_for_prediction[min_id] = gradients_for_prediction[min_id].add(-mean)

    # convert to tensor
    idlg_pred = torch.Tensor(idlg_pred).long().to(device)

    # print
    pred_srt = idlg_pred.data.tolist()
    pred_srt.sort()
    orig_srt = orig_label.data.tolist()
    orig_srt.sort()
    #print("Predicted: \t{}\nOrignal:\t{}".format(pred_srt, orig_srt))


    correct = 0
    false = 0
    for p in pred_srt:
        if orig_srt.__contains__(p):
            orig_srt.remove(p)
            correct+=1
        else:
            false+=1


    print("Correct: {}, False: {}, Acc: {}".format(correct,false,correct/(correct+false)))

    return idlg_pred