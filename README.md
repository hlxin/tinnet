# Theory Infused Neural Network (TinNet)

TinNet (short for Theory Infused Neural Network) is a conceptual framework that integrates traditional machine learning models with physics-based models. Traditional ML models often lack interpretability, while purely physics-based models can sometimes sacrifice accuracy. By combining the strengths of both, TinNet aims to achieve high accuracy, enhanced interpretability, and improved performance in both interpolation and extrapolation tasks.

## Why TinNet?

1. Bridging the interpretability gap

    Traditional machine learning methods, such as deep neural networks, can yield highly accurate predictions but often act as “black boxes,” providing limited insight into the underlying mechanisms.

2. Improving accuracy

    While physics-based models offer solid theoretical grounding and interpretability, they can sometimes struggle to match the predictive accuracy of ML models, especially when dealing with complex or high-dimensional data.

3. Better extrapolation

    By leveraging physics-based understanding, TinNet helps guide machine learning models beyond their training domain, mitigating large errors when extrapolating to new parameter spaces.

## Core Concept of TinNet

TinNet is not a single model but a methodological framework. In our research, we integrate CGCNN (Crystal Graph Convolutional Neural Network) with a variety of physics-based models:

1. Combining CGCNN and the Newns–Anderson model (This repository)

    Applied to predict and analyze *OH, *O, and *N adsorption energies on $d$-block metal alloys. The Newns–Anderson model provides physical insights into electronic structure and adsorption states.

2. Combining CGCNN and $d$-band theory (https://github.com/hlxin/tinnet_dos)

    Used to predict the $d$-band center, an important descriptor for the electronic structure of transition metals and alloys. $d$-band theory underpins our understanding of metal-adsorbate interactions.

3. Combining CGCNN and the tight-binding model (https://github.com/hlxin/tinnet_dos)

    Used to predict higher order moments, allowing for a more comprehensive analysis of electronic structures. The tight-binding model offers a simplified yet insightful perspective on electron propagation in solids.

## Software package

The TinNet software package is adapted from Crystal Graph Convolutional Neural Networks (CGCNN) codes of Jeffrey C. Grossman and Zachary W. Ulissi.
- [Crystal Graph Convolutional Neural Networks (CGCNN)](https://link.aps.org/doi/10.1103/PhysRevLett.120.145301).
- [Tian Xie](https://github.com/txie-93/cgcnn).
- [Zachary W. Ulissi](https://github.com/ulissigroup/cgcnn).

The package provides three major functions:

- Train a TinNet model with an user-defined physical model and dataset.
- Predict material properties and parameters of user-defined physical model of new crystals with a pre-trained TinNet model.
- Extract physical trends and phenomena from predicted parameters of user-defined physical model.

The following paper describes the details of the TinNet framework:

[Infusing Theory into Machine Learning for Interpretable Reactivity Prediction](https://www.nature.com/articles/s41467-021-25639-8)

## How to cite

Please cite the following work for TinNet:

Wang, S.-H.; Pillai, H. S.; Wang, S.; Achenie, L. E. K.; Xin, H. Infusing Theory into Deep Learning for Interpretable Reactivity Prediction. Nat. Commun. 2021, 12 (1), 5288. https://doi.org/10.1038/s41467-021-25639-8.

Huang, Y., Wang, S.-H., Achenie, L. E., Choudhary, K., & Xin, H. Origin of unique electronic structures of single-atom alloys unraveled by interpretable deep learning. The Journal of Chemical Physics, 2024, 161(16). https://doi.org/10.1063/5.0232141.

Huang, Y., Wang, S.-H., Kamanuru, M., Achenie, L. E., Kitchin, J. R., & Xin, H. Unifying theory of electronic descriptors of metal surfaces upon perturbation. Physical Review B, 2024, 110(12), L121404. https://doi.org/10.1103/PhysRevB.110.L121404.

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

To reproduce our paper, you can download the corresponding datasets following the [instruction](https://github.com/hlxin/tinnet/tree/master/data).
https://github.com/hlxin/tinnet/tree/master/data

## Authors

This software was primarily written by Shih-Han Wang who was advised by Prof. Luke E. K. Achenie and Prof. Hongliang Xin.

## License

TinNet is released under the MIT License. Feel free to use and cite it, respecting the specified terms and conditions.
