import torch
import torch.nn as nn

import aDPtorch.privacy_engine_xl as adp

# TODO: fix gradient compression
# gradients below threshold are accumulated until they reach said threshold
# 1. create a class for defenses
# 2. add cache for low_gradients
# 3. find way to reinject low_gradients into model
#    a) separate from or added to aggregated_gradients from FedAvg?
#    b) just use inject method with [] (empty list for grad param)


# class Defenses:

#     def __init__(self, setting):
#         self.setting = setting

#         # self.m = nn.Dropout(p=self.setting.parameter["dropout_prob"])

#     def update_setting(self, setting):
#         # Update setting
#         self.setting = setting

#         # # Update dropout probability
#         # if self.setting.parameter["dropout_prob"] != self.setting.parameter["dropout_prob"]:
#         #     self.m = nn.Dropout(p=self.setting.parameter["dropout_prob"])

def apply(grad, setting):
    # Noisy Gradients
    if setting.parameter["differential_privacy"]:
        clipping = True if setting.parameter["max_norm"] != None else False
        adp.apply_noise(grad, setting.parameter["batch_size"], setting.parameter["max_norm"], setting.parameter["noise_multiplier"], setting.parameter["noise_type"], setting.device, loss_reduction="none", clipping=clipping)

    # Gradient Compression
    if setting.parameter["compression"]:
        values = torch.sum(grad[-2], dim=-1).clone()
        magnitudes = [torch.abs(value) for value in values]
        magnitudes_sorted = sorted(magnitudes)

        threshold = int(len(magnitudes_sorted) * setting.parameter["threshold"]) - 1
        max_magnitude = magnitudes_sorted[threshold]
        max_mag_count = 1
        first_idx = threshold
        for i, mag in enumerate(magnitudes_sorted):
            if mag == max_magnitude:
                first_idx = i
        max_mag_count = threshold - first_idx

        count = 0
        for magnitude, tens in zip(magnitudes, grad):
            if magnitude < max_magnitude:
                tens.zero_()
            elif magnitude == max_magnitude:
                if count <= max_mag_count:
                    tens.zero_()
                else:
                    continue
                count += 1
            elif magnitude > max_magnitude:
                continue

    # # Dropout
    # # https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html
    # if setting.parameter["dropout"]:
    #     for tens in grad:
    #         tens = self.m(tens)

def inject(grads, grad_def, model):
    # print(type(setting.model.parameters()))
    # print(len(setting.model.parameters()))
    params = []
    for i_g, p in enumerate(model.parameters()):
        params.append(p)
        for grad in grads:
            params[i_g] = torch.sub(params[i_g], grad[i_g])
        params[i_g] = torch.add(params[i_g], grad_def[i_g])
    state_dict = model.state_dict()
    # print(len(grad))
    # print(state_dict.keys())
    for i, (key, tens) in enumerate(state_dict.items()):
        # print(len(grad[i]))
        # print(len(state_dict[key]))
        # print(grad[i][0])
        # print(state_dict[key][0])
        state_dict[key] = params[i]
    model.load_state_dict(state_dict)
