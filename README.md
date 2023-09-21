# Theory Infused Neural Network (TinNet)

The TinNet software package is adapted from Crystal Graph Convolutional Neural Networks (CGCNN) codes of Jeffrey C. Grossman and Zachary W. Ulissi.
- [Crystal Graph Convolutional Neural Networks (CGCNN)](https://link.aps.org/doi/10.1103/PhysRevLett.120.145301).
- [Tian Xie](https://github.com/txie-93/cgcnn).
- [Zachary W. Ulissi](https://github.com/ulissigroup/cgcnn).

The package provides three major functions:

- Train a TinNet model with an user-defined physical model and dataset.
- Predict material properties and parameters of user-defined physical model of new crystals with a pre-trained TinNet model.
- Extract physical trends and phenomena from predicted parameters of user-defined physical model.

The following paper describes the details of the TinNet framework:

[Infusing Theory into Machine Learning for Interpretable Reactivity Prediction](URL TBD)

## Table of Contents

- [How to cite]
- [Prerequisites]
- [Usage]
  - [Load images and properties]
  - [Tune hyperparameters]
  - [Train a TinNet model]
  - [Predict material properties with a pre-trained TinNet model]
- [Data]
- [Authors]
- [License]

## How to cite

Please cite the following work for TinNet:

Wang, S.-H.; Pillai, H. S.; Wang, S.; Achenie, L. E. K.; Xin, H. Infusing Theory into Deep Learning for Interpretable Reactivity Prediction. Nat. Commun. 2021, 12 (1), 5288. https://doi.org/10.1038/s41467-021-25639-8.

##  Prerequisites

This package requires:

- [PyTorch](http://pytorch.org)
- [pymatgen](http://pymatgen.org)

*Note: The package is built in the PyTorch v1.3.1 environment.

## Usage

To use the TinNet package, you need to load images and properties in the Python script.
There are three examples of TinNet models in the repository: 'Main/Tuning_Hyperparameters.py', 'Main/Training.py' and 'Main/Test.py'. 

arguments:

(1) task = {'train', 'test'}

(2) data_format = {'regular', 'nested', 'test'}

(3) phys_model = {'gcnn', 'newns_anderson_semi', 'user-defined'}

### Load images and properties

Images should be one of Atomic Simulation Environment (ASE) formats (e.g. traj, cif, xyz) (https://wiki.fysik.dtu.dk/ase/ase/io/io.html).
Properties contain the major property (e.g. adsorption energy, formation energy) and extra information (e.g. orbital information)(optional).

### Tune hyperparameters

Set the range of hyperparameters you want to tune.

In directory 'Main', you can tune hyperparameters by:

'''
python Tuning_Hyperparameters.py
'''

After tuning hyperparameters, you will get four files in 'Main' directory.

- 'final_ans_test_mae_HYPERPARAMETERS.txt': stores mean absolute errors (MAE) of different test dataset with HYPERPARAMETERS.
- 'final_ans_test_mse_HYPERPARAMETERS.txt': stores mean absolute errors (MSE) of different test dataset with HYPERPARAMETERS.
- 'final_ans_val_mae_HYPERPARAMETERS.txt': stores mean absolute errors (MAE) of different validation dataset with HYPERPARAMETERS.
- 'final_ans_val_mse_HYPERPARAMETERS.txt': stores mean absolute errors (MSE) of different validation dataset with HYPERPARAMETERS.

### Train a TinNet model

Use optimized hyperparameters to train models.

In directory 'Main', you can train by:

'''
python Training.py
'''

After training, you will get one file in 'Main' directory.
- 'model_best_train_idx_val_X_idx_test_Y.pth.tar': stores the TinNet model with the best validation accuracy. (X and Y represent the fold of the validation dataset and the test dataset, respectively.)

### Predict material properties with a pre-trained TinNet model

Once you have a pre-trained TinNet model, you can use it to predict material properties of structures that you are interested

In directory 'Main', you can predict material properties of structures by:

'''
python Test.py
'''

After predicting, you will get six files in 'Main' directory:

- 'test_results_idx_val_X_idx_test_Y.csv': stores the 'ID', target value, and predicted value for each image in test dataset.
- 'train_results_idx_val_X_idx_test_Y.csv': stores the 'ID', target value, and predicted value for each image in train dataset.
- 'validation_results_idx_val_X_idx_test_Y.csv': stores the 'ID', target value, and predicted value for each image in validation dataset.
- 'parm_test_idx_val_X_idx_test_Y.txt': stores parameters of user-defined physical model for each image in test dataset.
- 'parm_train_idx_val_X_idx_test_Y.txt': stores parameters of user-defined physical model for each image in train dataset.
- 'parm_validation_idx_val_X_idx_test_Y.txt': stores parameters of user-defined physical model for each image in validation dataset.

## Data

To reproduce our paper, you can download the corresponding datasets following the [instruction](Data).
https://github.com/hlxin/tinnet/tree/master/data

## Authors

This software was primarily written by [Shih-Han Wang] who was advised by [Prof. Luke E. K. Achenie] and [Prof. Hongliang Xin].

## License

TinNet is released under the MIT License.
