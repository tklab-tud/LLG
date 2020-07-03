import torch
import torch.nn as nn
from net import Net1, Net2, weights_init

def prediction(parameter, gradient_list, model, orig_data, orig_label, device):
    idlg_pred = []

    # Classic way from the authors repository does not allow bs <> 1
    if parameter["prediction"] == "classic":
        if parameter["batch_size"] == 1:
            idlg_pred.append(torch.argmin(torch.sum(gradient_list[-2], dim=-1), dim=-1).detach().reshape(
                (1,)).requires_grad_(False))
        else:
            exit("classic prediction does not support batch_size <> 1")


    # Simplified Way as described in the paper
    elif parameter["prediction"] == "simplified":
        #creating bs 1 model for single prediction
        parameter_bs1 = parameter.copy()
        parameter_bs1["batch_size"] = 1

        if parameter["model"] == 1:
            model_bs1 = Net1(parameter_bs1)
            model_bs1.apply(weights_init)
        elif parameter["model"] == 2:
            model_bs1 = Net2(parameter_bs1)

        model_bs1 = model.to(device)

        for i_s, sample in enumerate(orig_data):
            orig_out = model_bs1(sample.view(1,1,28,28))
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
            min = (0, 0)
            min_id = 0
            for (i, tuple) in enumerate(candidates):
                if tuple[1] < min[1]:
                    min = tuple
                    min_id = i

            idlg_pred.append(min[0])

            # add the mean value of one accurance to the candidate
            candidates[min_id] = (min[0], candidates[min_id][1].add(-mean))


    return idlg_pred