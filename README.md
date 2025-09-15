# A MUTATION-BASED APPROACH TO ALLEVIATE THE CLASS IMBALANCE PROBLEM IN SOFTWARE DEFECT PREDICTION

Highly imbalanced training datasets considerably degrade the performance of software defect predictors. Software Defect Prediction (SDP) datasets have a general problem, which is class imbalance. Therefore, a variety of methods have been developed to alleviate Class Imbalance Problem (CIP). However, these classical methods, like data-sampling, balance datasets without connecting any relation with SDP. Over-sampling techniques generate synthetic minor class instances, which generalize a small number of minor class instances and result in less diverse instances, whereas under-sampling techniques eliminate major class instances, resulting in significant information loss. In this study, we present an approach that uses software mutations to balance software repositories. Mutation-based Approach (MBA) injects mutants into defect-free instances, causing them to transform into defective instances. In this way, MBA balances datasets with diverse data produced by mutation operators, and there is no loss on instances as in under-sampling.

## In This Repository

The experimentation and data generating scripts, as well as the resulting data and performance findings of our study, are all represented in this repository.

* [Performance Results](Performance%20Results)

* [Software Measures Datasets](Software%20Measures%20Datasets)

* [Scripts](Scripts)

## Rebalancing Software Defect Datasets via Mutation: Performance Insights from Prediction Models based on Software Measures Updates

From the original study, we developed an article (Rebalancing Software Defect Datasets via Mutation: Performance Insights from Prediction Models based on Software Measures Updates) and subsequently extended the work to enhance the reliability of the findings. In this extended study, we incorporated two additional machine learning methods—XGBoost and neural networks. Furthermore, to strengthen the robustness of the results, we repeated the sampling experiments five times using different random seeds.
