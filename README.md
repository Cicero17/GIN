We reproduce and extend the Graph Isomorphism Network architecture proposed by the authors of "How Powerful are Graph Neural Networks?" (2018, https://arxiv.org/abs/1810.00826).

# Requirements
Libraries: os, torch(from 1.7.1), torch_geometric(from 1.6.3), torch_scatter(from 2.0.5).

# Overview
The notebook consists of 2 classes: MLP (used as part of the GIN) and GIN (the model implementation). GIN accommodates both versions proposed by the authors: GIN-e (where e is a learnable parameter) and GIN-0 (e = 0); SUM, MEAN, MAX neighborhood aggregation; SUM graph readout. We extend the GIN models to accommodate LSTM neighborhood aggregation and random node initializaion (RNI). 

We provide with functions to do k-fold-cross-validation, train and test the model.

The datasets used are part of the TUDataset: MUTAG, PROTEINS, IMDB-BINARY, IMDB-MULTI, REDDIT-BINARY, REDDIT-MULTI-5K, COLLAB, NCI1, PTC_MR, SYNTHETICnew, Cuneiform. We provide with functions to shuffle the datasets and add node features (zero tensors) when they are not provided by the datasets.

# How to run
Download the jupyter notebook and run the cells. The last cell provides with an example how to define and run a GIN model. In general:

model = GIN(num_layers, num_mlps, input_dim, hidden_dim, output_dim, dropout_rate, nbh_agg, graph_agg, learn_eps, random)
k_fold_cross_validation(k, model, dataset, epochs, batch_size, output_dim)

For more details, check the descriptions of each function from the notebook.
