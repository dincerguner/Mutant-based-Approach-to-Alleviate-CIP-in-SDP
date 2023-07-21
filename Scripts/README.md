# Scripts

## mutation
-----

[mutate_auto.py](mutation/mutate_auto.py) is used to mutate projects which are presented in following repository. Also, mutated versions are presented.

* [master_thesis_dataset](https://github.com/dincerguner/master_thesis_dataset)

[prepare_calculated_dataset_auto.py](mutation/prepare_calculated_dataset_auto.py) is used to prepare software measure datasets of mutated projects which stored in following repository.

* [master_thesis_dataset](https://github.com/dincerguner/master_thesis_dataset)


## prediction
-----

There are the scripts that used to build and evaluate SDP models.

Following scripts takes the necessary [Software Measure Datasets](../Software%20Measures%20Datasets) and builds SDP models then saves the results in [Performance Results](../Performance%20Results) folder.


* For Rebalancing Methods (Results of 5 different over-sampling techniques, Baseline and MBA):

    |          | IRDP| CPDP|
    | -------- | :---: | :---: | 
    | **Baseline** | [rebalancing_methods_irdp_baseline.py](prediction/rebalancing_methods_irdp_baseline.py) | [rebalancing_methods_cpdp_baseline.py](prediction/rebalancing_methods_cpdp_baseline.py) |
    | **Mutated**  | [rebalancing_methods_irdp_mutated.py](prediction/rebalancing_methods_irdp_mutated.py) |  [rebalancing_methods_cpdp_mutated.py](prediction/rebalancing_methods_cpdp_mutated.py) |
    | **Sampling** | [rebalancing_methods_irdp_sampling.py](prediction/rebalancing_methods_irdp_sampling.py) | [rebalancing_methods_cpdp_sampling.py](prediction/rebalancing_methods_cpdp_sampling.py) |

* For Stability of MBA (Results of 3 different defect-ratios (0.3, 0.4 and 0.5) of MBA and Baseline):  

    |          | IRDP| CPDP|
    | -------- | :---: | :---: | 
    | **Baseline** | [stability_of_mba_irdp_baseline.py](prediction/stability_of_mba_irdp_baseline.py) | [stability_of_mba_cpdp_baseline.py](prediction/stability_of_mba_cpdp_baseline.py) |
    | **Mutated**  | [stability_of_mba_irdp_mutated.py](prediction/stability_of_mba_irdp_mutated.py) |  [stability_of_mba_cpdp_mutated.py](prediction/stability_of_mba_cpdp_mutated.py) |
