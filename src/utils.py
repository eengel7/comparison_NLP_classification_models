import torch
import warnings
from torch.nn import CrossEntropyLoss




def calculate_loss(model, inputs, num_labels, weight, device):

    if weight:
        loss_fct = CrossEntropyLoss(weight=torch.Tensor(weight).to(device))
    else:
        warnings.warn(
            f"No weights set for cross entropy loss."
        )
        loss_fct = None
        
    outputs = model(**inputs)
    # model outputs are always tuple in pytorch-transformers (see doc)
    loss = outputs[0]
    if loss_fct:
        logits = outputs[1]
        labels = inputs["labels"]

        loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
    return (loss, *outputs[1:])