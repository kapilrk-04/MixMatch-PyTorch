[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/Zksn1waN)

### TEAM NAME - ~

### TEAM MEMBERS - 

- Harshavardhan P - 2021111003
- Kapil Rajesh Kavitha - 2021101028

#### INTRODUCTION

- MixMatch is a semi-supervised learning technique that combines labeled and unlabeled data to improve the performance of the model. This is a Python implementation of a deep learning model using the MixMatch algorithm. 


#### INPUTS

- We utilise the parse_args() function from the argparse library to parse the arguments passed to the script. The arguments are used for choosing the dataset, setting hyperparameters, and specifying GPU usage. This allows us to customise the training process without having to change the code.

#### DATASET CREATION

- The dataset specified by the `--dataset` argument is loaded using a custom `dataloader` module, which splits the data into labeled and unlabeled subsets. The number of labeled examples is controlled by the --labeled_n argument. The data is also split into training, validation, and test sets.

#### MODEL AND OPTIMIZERS

- The model used is a WideResNet (Wide Residual Network) model. Xavier weight initialization is used for initializing the weights of the model's layers. 

- The Adam optimizer is used for training the model. Additionally, an EMA Optimizer is used to maintain an exponential moving average of the model's parameters and weights. This is used to generate predictions for the unlabeled data.

#### IMAGE AUGMENTATION

- Before passing the training images to the model, augmentation is performed on them - for each image, `k` augmented images are generated and added to the dataloader. Augmentation is performed by applying a series of transformations such as 
1. random cropping, 
2. horizontal flipping, 
3. applying random color jitter,
4. random grayscale conversion, and
5. applying random Gaussian blur
The images are then normalized using the mean and standard deviation of the dataset. The number of augmented images generated for each image is controlled by the `--K` argument.

#### MIXUP

- The mix of labeled and unlabeled data is generated using the MixUp algorithm. The data is then mixed together based on a randomly generated mixing ratio, which is controlled by the `--alpha` argument. Two sets of mixed input pairs and their corresponding target pairs are generated, which are then combined to create the final `mixed_input` and `mixed_target` tensors that are used for training the model.

#### TRAINING AND EVALUATION

- The model is trained over the data generated from the MixUp algorithm.The labeled and unlabeled data are combined and processed together using the `interleave_tensors` function - this is a critical step in the MixMatch algorithm. 

- The model is evaluated on the validation set after every epoch. The model's performance is evaluated using the accuracy metric. The model's performance on the test set is also evaluated after training is complete.

- After around 3 epochs, the model achieves 100% accuracy on the training set, and the accuracy also gradually increases for the validation set. This happens without the model overfitting on the training data, due to the augmentation techniques used on the training data. 