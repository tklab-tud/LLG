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


class Defenses:

    def __init__(self, setting):
        self.setting = setting
        self.device = setting.device
        self.parameter = setting.parameter

    def update_setting(self, setting):
        # Update setting
        self.setting = setting
        self.device = setting.device
        self.parameter = setting.parameter

    def apply(self, grad):
        # Noisy Gradients
        if self.parameter["differential_privacy"]:
            clipping = True if self.parameter["max_norm"] != None else False
            adp.apply_noise(grad, self.parameter["batch_size"], self.parameter["max_norm"], self.parameter["noise_multiplier"], self.parameter["noise_type"], self.device, loss_reduction="none", clipping=clipping)

        # Gradient Compression
        if self.parameter["compression"]:
            values = torch.sum(grad[-2], dim=-1).clone()
            magnitudes = [torch.abs(value) for value in values]
            magnitudes_sorted = sorted(magnitudes)

            threshold = int(len(magnitudes_sorted) * self.parameter["threshold"]) - 1
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

    def inject(self, grads, grad_def, model):
        params = []
        for i_g, p in enumerate(model.parameters()):
            params.append(p)
            for grad in grads:
                params[i_g] = torch.sub(params[i_g], grad[i_g])
            params[i_g] = torch.add(params[i_g], grad_def[i_g])
        state_dict = model.state_dict()
        for i, (key, tens) in enumerate(state_dict.items()):
            state_dict[key] = params[i]
        model.load_state_dict(state_dict)
