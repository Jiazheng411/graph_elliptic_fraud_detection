# Graph Elliptic Fraud Detection
https://www.kaggle.com/datasets/ellipticco/elliptic-data-set

paper: https://arxiv.org/pdf/1908.02591

## What code are available

### Data split
* data_split.py - data split script 
* run_simple_split.py - run data split
* validate_splits_fixed.py - validate split succeeded

just run `run_simple_split.py` to split data. data will be saved to `splits/` folder. Note: put the raw data under `data/` folder


## Model & experiment
* xgboost_experiment.py - XGboost experiments
* xgboost_results_with_timestep.csv (0.78 f1 best so far) - result using timestep as 1 of the feature; result for all test data and for each time step
* xgboost_results_without_timestep.csv (0.78 f1 best so far) - result does not using timestep; result for all test data and for each time step

* gcn_paper_reproduction.py - gcn model and training according to paper setting
    * Settings: 2 gcn layers, 100 node features, Adam optimizer learning rate 0.001, weighted crossentropy 0.7 / 0.3 for illicit / licit (from paper)
    * Experiment on both without unknown nodes (0.57 f1), and with unknown nodes (0.51 f1). With unknown nodes does not really help, and did not really reach 0.62 - the f1 in the paper
    * Please note illicit is 0, evaluation based on illicit class

* graph_transformer_clean.py - graph transformer training similar for gcn paper
    * Adam optimizer learning rate 0.001, weighted crossentropy 0.7 / 0.3 for illicit / licit
    * Experiment on both without unknown nodes (0.65 f1), and with unknown nodes (0.62 f1). With unknown nodes does not really help
    * Please note illicit is 0, evaluation based on illicit class
