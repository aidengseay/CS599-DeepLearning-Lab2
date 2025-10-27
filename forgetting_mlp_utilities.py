################################################################################
# Forgetting MLP Utilities
# Aiden Seay, CS 599 - Deep Learning - Fall 2025
################################################################################
# Imports

import tensorflow as tf
from mnist_reader import *

################################################################################
# Constants

# optimizers
SGD = "SGD"
ADAM = "ADAM"
RMSPROP = "RMSPROP"

# loss functions
NLL = "NLL"
L1 = "L1"
L2 = "L2"
L1_L2 = "L1 + L2"

################################################################################
# Supporting Functions

def set_seed(seed_in = 12345):

    """
    Definition: Sets the same seed for both random TensorFlow operations and
                random package operations.
    
    Inputs: seed_in (int) - the seed used to set randomized seed
    
    Outputs: None
    """

    tf.random.set_seed(seed_in)
    np.random.seed(seed_in)
    

def get_data(valid_fract = 0.10):

    """
    Definition: Gets training, validation, and testing datasets from external
                source. Additionally normalizes the data and ensures datatypes
                are correct for training.

    Inputs: valid_fract (float) - the fraction in decimal form of the proportion
                                  of training data that is validation data.

    Outputs: X_val (tensor) - the features in validation set
             y_val (tensor) - the results in the validation set
             X_train (tensor) - the features in the training set
             y_train (tensor) - the results in the training set
             X_test (tensor) - the features in the testing set
             y_test (tensor) -  the results in the testing set
    """

    # read in the data
    fmnist_folder = "fmnist_data"
    X_train_all, y_train_all = load_mnist(fmnist_folder, kind='train')
    X_test, y_test = load_mnist(fmnist_folder, kind='t10k')

    # compute fraction of training data being used for validation
    num_val = int(len(X_train_all) * valid_fract)

    # shuffle to ensure random validation set
    indices = np.arange(len(X_train_all))
    np.random.shuffle(indices)

    X_train_all = X_train_all[indices]
    y_train_all = y_train_all[indices]

    # split training data into training and validation subsets
    X_val = X_train_all[:num_val]
    y_val = y_train_all[:num_val]
    X_train = X_train_all[num_val:]
    y_train = y_train_all[num_val:]

    # ensure all values are flt or int for later computations
    # also need to normalize the features between 0 - 1
    X_val = X_val.astype("float32") / 255.0
    y_val = y_val.astype("int32")
    X_train = X_train.astype("float32") / 255.0
    y_train = y_train.astype("int32")
    X_test = X_test.astype("float32") / 255.0
    y_test = y_test.astype("int32")

    # display train, validation, and test split for check
    print(f"Train Set:        {X_train.shape}, {y_train.shape}")
    print(f"Validation Set:   {X_val.shape} , {y_val.shape}")
    print(f"Test Set:         {X_test.shape}, {y_test.shape}")

    return X_val, y_val, X_train, y_train, X_test, y_test


def get_optimizer(optimizer_type, input_learning_rate):

    """
    Definition: gets the specific optimizer used for a model.

    Inputs: optimizer_type (int) - the constant used to define what optimizer is
                                   used.

    Outputs: optimizer function - the function used to optimize the model.
    """

    if optimizer_type == SGD:

        return tf.keras.optimizers.SGD(learning_rate=input_learning_rate)

    elif optimizer_type == ADAM:

        return tf.keras.optimizers.Adam(learning_rate=input_learning_rate)

    elif optimizer_type == RMSPROP:

        return tf.keras.optimizers.RMSprop(learning_rate=input_learning_rate)
    
    # default to None to deliver error message
    else:

        return None
    
    
def get_regularizer(loss_type, l1_lambda=0.0, l2_lambda=0.0):
    
    """
    Definition: returns the appropriate kernel regularizer based on the loss 
                type.

    Inputs: loss_type (int)  - constant defining which loss/regularization to 
                               use. (NLL, L1, L2, or L1_L2)
            l1_lambda (float) - strength of L1 regularization
            l2_lambda (float) - strength of L2 regularization

    Output: A tf.keras.regularizers object
    """

    if loss_type == NLL:

        return None

    elif loss_type == L1:

        return tf.keras.regularizers.l1(l1_lambda)

    elif loss_type == L2:

        return tf.keras.regularizers.l2(l2_lambda)

    elif loss_type == L1_L2:

        return tf.keras.regularizers.L1L2(l1=l1_lambda, l2=l2_lambda)

    # default to None to deliver error message
    else:
        
        return None
    
def summarize_metrics(acc_matrix):
    
    """
    Definition: Computes Average Accuracy (ACC) and Backward Transfer (BWT)
                from the accuracy matrix collected across all tasks.

    Inputs: acc_matrix (ndarray) - a matrix of shape [num_tasks x num_tasks]
                                   where acc_matrix[i, j] is the accuracy
                                   on task j after training on task i.

    Outputs: ACC (float) - average accuracy across all tasks after the final task.
             BWT (float) - backward transfer measuring forgetting.
    """


    T = acc_matrix.shape[0]

    # final average accuracy (ACC):
    ACC = np.mean(acc_matrix[T-1, :])

    # backward transfer (BWT):
    BWT = np.mean(acc_matrix[T-1, :T-1] - np.diag(acc_matrix)[:T-1])

    # true backward transfer
    TBWT = (acc_matrix[T-1, :T-1] - np.diag(acc_matrix)[:T-1])

    # cumulative backward transfer over time
    CBWT = []
    for t in range(1, T):
        deltas = acc_matrix[t, :t] - np.diag(acc_matrix)[:t]
        CBWT_t = np.mean(deltas)
        CBWT.append(CBWT_t)

    return ACC, BWT, TBWT, CBWT