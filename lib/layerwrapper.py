import torch
import torch.nn as nn


# Define WrappedGPT class
class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        # self.activations = [torch.zeros((self.columns), device=self.dev)]
        self.activations = []
        self.nsamples = 0

        self.layer_id = layer_id
        self.layer_name = layer_name

    def add_batch(self, inp, out, tar):
        """
        tar: batch_size * seq_len, inp corresponding to the position where tar == -100 will be ignored
        """
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        if len(tar.shape) == 2:
            tar = tar.unsqueeze(0)

        tmp = inp.shape[0]  # bs

        mask = tar.ne(-100)
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            mask = mask.flatten()
            inp = inp[mask]  # remove -100's
            inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples
        self.activations.append(inp)
