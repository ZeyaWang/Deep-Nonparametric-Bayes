# Deep Nonparametric Bayes (DNB): A Joint Learning Framework for Deep Bayesian Nonparametric Clustering


## Overview
DNB is a unified Bayesian Nonparametric framework for jointly learning image clusters and deep representations in a doubly-unsupervised manner, where we estimate not only the unknown image labels but also the unknown number of labels as well.


## Dependencies
All the dependecies are provided in the requirement.txt.

## Train model
Run train/dnb.py with the corresponding arguments to implement the clustering algorithm, you need to specify the dataset directory by passing your path of dataset to --dataset_dir and the dataset names by passing to --dataset (e.g., ytf). The dataset can be downloaded from https://github.com/jwyang/JULE.torch.

```
python3 train/dnb.py --dataset_dir=$DATASETPATH --dataset=ytf
```

## License
This code is released under the MIT License (refer to the LICENSE file for details).

