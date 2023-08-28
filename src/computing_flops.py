"""Computes the flops needed for training/running transformer networks."""

import collections

DROPOUT_FLOPS = 4
LAYER_NORM_FLOPS = 5
ACTIVATION_FLOPS = 8
SOFTMAX_FLOPS = 5

# Additional estimation of computing sigmoid: f(x) = 1 / (1 + exp(-x))
SIGMOID_FLOPS = 4


class TransformerHparams(object):
    """Computes the train/inference FLOPs for transformers."""

    def __init__(self, h, l, s=512, v=30522, e=None, i=None, heads=None, head_size=None, output_frac=0.15625, sparse_embed_lookup=False, decoder=False, num_labels = 305):
        self.h = h  # hidden size
        self.l = l  # number of layers
        self.s = s  # sequence length
        self.v = v  # vocab size 
        self.e = h if e is None else e  #embedding size
        self.i = h*4 if i is None else i  #intermediate size

        # attn proj sizes
        self.kqv = h if head_size is None else head_size * heads  
        # attention heads
        self.heads = max(h // 64, 1) if heads is None else heads 
        # percent of tokens using an output softmax
        self.output_frac = output_frac  
        # sparse embedding lookups
        self.sparse_embed_lookup = sparse_embed_lookup  
        # decoder has extra attn to encoder states
        self.decoder = decoder  
        # number of labels for multi-label/multi-class classification
        self.num_labels = num_labels

    def get_block_flops(self):
        """Get the forward-pass FLOPs for a single transformer block."""
        attn_mul = 2 if self.decoder else 1
        block_flops = dict(
            kqv=3 * 2 * self.h * self.kqv * attn_mul,
            kqv_bias=3 * self.kqv * attn_mul,
            attention_scores=2 * self.kqv * self.s * attn_mul,
            attn_softmax=SOFTMAX_FLOPS * self.s * self.heads * attn_mul,
            attention_dropout=DROPOUT_FLOPS * self.s * self.heads * attn_mul,
            attention_scale=self.s * self.heads * attn_mul,
            attention_weighted_avg_values=2 * self.h * self.s * attn_mul,
            attn_output=2 * self.h * self.h * attn_mul,
            attn_output_bias=self.h * attn_mul,
            attn_output_dropout=DROPOUT_FLOPS * self.h * attn_mul,
            attn_output_residual=self.h * attn_mul,
            attn_output_layer_norm=LAYER_NORM_FLOPS * attn_mul,
            intermediate=2 * self.h * self.i,
            intermediate_act=ACTIVATION_FLOPS * self.i,
            intermediate_bias=self.i,
            output=2 * self.h * self.i,
            output_bias=self.h,
            output_dropout=DROPOUT_FLOPS * self.h,
            output_residual=self.h,
            output_layer_norm=LAYER_NORM_FLOPS * self.h,
        )
        return sum(block_flops.values()) * self.s

    def get_embedding_flops(self, output=False):
        """Get the forward-pass FLOPs the transformer inputs or output softmax."""
        embedding_flops = {}
        if output or (not self.sparse_embed_lookup):
            embedding_flops["main_multiply"] = 2 * self.e * self.v
        # input embedding post-processing
        if not output:
            embedding_flops.update(dict(
                tok_type_and_position=2 * self.e * (self.s + 2),
                add_tok_type_and_position=2 * self.e,
                emb_layer_norm=LAYER_NORM_FLOPS * self.e,
                emb_dropout=DROPOUT_FLOPS * self.e
            ))
        # projection layer if e != h
        if self.e != self.h or output:
            embedding_flops.update(dict(
                hidden_kernel=2 * self.h * self.e,
                hidden_bias=self.e if output else self.h
            ))
        # extra hidden layer and output softmax
        if output:
            embedding_flops.update(dict(
                hidden_activation=ACTIVATION_FLOPS * self.e,
                hidden_layernorm=LAYER_NORM_FLOPS * self.e,
                output_softmax=SOFTMAX_FLOPS * self.v,
                output_target_word=2 * self.v
            ))
            return self.output_frac * sum(embedding_flops.values()) * self.s
        return sum(embedding_flops.values()) * self.s

    def get_binary_classification_flops(self):
        classification_flops = dict(
            hidden=2 * self.h * self.h,
            hidden_bias=self.h,
            hidden_act=ACTIVATION_FLOPS * self.h,
            logits=2 * self.h
        )
        return sum(classification_flops.values()) * self.s
    
    def get_multi_label_classification_flops(self):
        classification_flops = dict(
            hidden=2 * self.h * self.h,
            hidden_bias=self.h,
            hidden_act=ACTIVATION_FLOPS * self.h,
            logits= self.num_labels * self.h,
            sigmoid_flops = SIGMOID_FLOPS * self.num_labels
        )
        return sum(classification_flops.values()) * self.s 
    
    def get_infer_flops_multi_label(self, include_embedding = True):
        """Get the FLOPs for running inference with the transformer on a multi-label
        classification task."""
        if include_embedding:
            embedding_flops = self.get_embedding_flops(output=False)
        else:
            embedding_flops = 0 

        return ((self.l * self.get_block_flops()) +
                embedding_flops + 
                self.get_multi_label_classification_flops())

    def get_train_flops(self, batch_size, train_steps, discriminator=False):
        """Get the FLOPs for pre-training the transformer."""
        # 2* for forward/backward pass
        return 2 * batch_size * train_steps * (
            (self.l * self.get_block_flops()) +
            self.get_embedding_flops(output=False) +
            (self.get_binary_classification_flops() if discriminator else
            self.get_embedding_flops(output=True))
        )

    def get_infer_flops(self):
        """Get the FLOPs for running inference with the transformer on a binary
        classification task."""
        return ((self.l * self.get_block_flops()) +
                self.get_embedding_flops(output=False) +
                self.get_binary_classification_flops())




MODEL_FLOPS = collections.OrderedDict([
    ("bert_small", TransformerHparams(256, 12, e=128, s=128).get_infer_flops_multi_label()),
    ("bert_base", TransformerHparams(768, 12, s = 512 ).get_infer_flops_multi_label()),
    ("bert_base_128", TransformerHparams(768, 12, s = 128).get_infer_flops_multi_label()),
    ("distilbert", TransformerHparams(768, 6, s = 512 ).get_infer_flops_multi_label()),
    # RoBERTa, ALBERT, and T5 have  minor architectural differences from
    # BERT/ELECTRA, but I believe they don't significantly effect the runtime,
    # so we use this script for those models as well.
    ("roberta", TransformerHparams(1024, 24, v=50265).get_infer_flops_multi_label()),
])

MODEL_FLOPS_NO_EMBEDDING = collections.OrderedDict([
    ("bert_small", TransformerHparams(256, 12, e=128, s=128).get_infer_flops_multi_label(include_embedding = False)),
    ("bert_base", TransformerHparams(768, 12, s = 512 ).get_infer_flops_multi_label(include_embedding = False)),
    ("bert_base_128", TransformerHparams(768, 12, s = 128).get_infer_flops_multi_label(include_embedding = False)),
    ("distilbert", TransformerHparams(768, 6, s = 512 ).get_infer_flops_multi_label(include_embedding = False)),
    # RoBERTa, ALBERT, and T5 have  minor architectural differences from
    # BERT/ELECTRA, but I believe they don't significantly effect the runtime,
    # so we use this script for those models as well.
    ("roberta", TransformerHparams(1024, 24, v=50265).get_infer_flops_multi_label(include_embedding = False)),
])

def main():
    for k, v in MODEL_FLOPS.items():
        print(k, v)

    print('WITHOUT EMBEDDING:')

    for k, v in MODEL_FLOPS_NO_EMBEDDING.items():
        print(k, v)
    
    vocab_size = 10000
    print(f'logistic regression with vocab size {vocab_size}: {305*(305+ 2*vocab_size)}')


if __name__ == "__main__":
  main()