from src.classification import (MultiLabelClassificationArgs,
                                MultiLabelClassificationModel)

def measure_flop_per_example(classficiation_model, sequence_length: int = 512, only_forward: bool = True, exclude_embeddings: bool = True):
    """
    re-implements the HF function floating_point_ops outside the trainer
    """

    if only_forward:
        cost_backward = 1
    else:
        cost_backward = 3
    cost_forward = 2

    flop_count = cost_forward * cost_backward * sequence_length * classficiation_model.model.num_parameters(exclude_embeddings=exclude_embeddings)

    print("Kaplan estimation: FLOPs per example (Inference, only forwardpass): ", flop_count)

    return flop_count

if __name__ == "__main__":

    num_labels = 305
    only_forward= True 
    exclude_embeddings = False 

    bert = MultiLabelClassificationModel("bert", "bert-base-uncased", num_labels=num_labels)
    bert_flop = measure_flop_per_example(bert, sequence_length = 512, only_forward = only_forward, exclude_embeddings = exclude_embeddings)
    
    bert_128 = MultiLabelClassificationModel("bert", "bert-base-uncased", num_labels=num_labels)
    bert_flop_128 = measure_flop_per_example(bert_128, sequence_length = 128, only_forward = only_forward, exclude_embeddings = exclude_embeddings)

    distilbert = MultiLabelClassificationModel("distilbert", "distilbert-base-uncased", num_labels=num_labels)
    distilbert_flop = measure_flop_per_example(distilbert, sequence_length = 512, only_forward = only_forward, exclude_embeddings = exclude_embeddings)
    print( bert_flop, bert_flop_128, distilbert_flop)