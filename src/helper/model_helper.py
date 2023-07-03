from collections import OrderedDict
# from flwr.server.server import shutdown
import torch


def get_weights(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


# Function to set the weights of a model
def set_weights(model, weights) -> None:
    params_dict = zip(model.state_dict().keys(), weights)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
