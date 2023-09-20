# Balancing Performance and Usage Cost: A Comparative Study of Language Models for Scientific Text Classification 

## Master's Thesis of Eva Engel 

### Abstract

The emergence of large language models, such as BERT and GPT-3, has revolutionized natural language processing tasks. However, the development and deployment of these models pose challenges, including concerns about computational resources and environmental impact. This study aims to compare discriminative language models for text classification based on their performance and usage cost. We evaluate the models using a hierarchical multi-label text classification task and assess their performance using primarly F1-score. Additionally, we analyze the usage cost by calculating the Floating Point Operations (FLOPs) required for inference. We compare a baseline model, which consists of a classifier chain with logistic regression models, with fine-tuned discriminative language models, including BERT with two di erent sequence lengths and DistilBERT, a distilled version of BERT. Results show that the DistilBERT model performs optimally in terms of performance, achieving an F1-score of 0.56 averaged on all classification layers. The baseline model and BERT with a maximal sequence length of 128 achieve F1-scores of 0.51. However, the baseline model outperforms the transformers at the most specific classification level with an F1-score of 0.33. Regarding usage cost, the baseline model significantly requires fewer FLOPs compared to the transformers. Furthermore, restricting BERT to a maximum sequence length of 128 tokens instead of 512 sacrifices some performance but o ers substantial gains in usage cost. 

One can find the final report in this repository. 
