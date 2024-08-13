import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
import matplotlib.pyplot as plt
from IPython.display import clear_output
from sklearn.model_selection import KFold
import numpy as np
import os
import datetime
import keyboard  # Import keyboard module to detect key presses
import sys  # Import sys to terminate the program
import shutil

# Ensure TensorFlow uses the GPU
gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess the data
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add a channel dimension to the images (but keep the input shape constant)
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# Initialize lists to store data for plotting
num_nodes_list = []
loss_list = []

def build_model(hp):
    model = keras.Sequential()

    # Fixed input shape
    input_shape = (28, 28, 1)
    model.add(layers.InputLayer(input_shape=input_shape))

    # Track the number of nodes
    num_nodes = 0

    # Define the possible activation functions
    activation = hp.Choice('activation', values=['relu', 'tanh', 'sigmoid', 'selu', 'leaky_relu'])

    # Build a fixed architecture without altering input shape
    for i in range(hp.Int('num_layers', 8, 18)):
        filters = hp.Int(f'filters_{i}', min_value=16, max_value=256, step=32)
        kernel_size = hp.Choice(f'kernel_size_{i}', values=[1, 3, 5, 7, 9, 11, 13])

        model.add(layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=None, padding='same', kernel_regularizer=keras.regularizers.l2(0.01)))
        model.add(layers.BatchNormalization())
        
        # Use the chosen activation function
        if activation == 'leaky_relu':
            model.add(layers.LeakyReLU())
        else:
            model.add(layers.Activation(activation))
        
        model.add(layers.MaxPooling2D(pool_size=2, padding='same'))
        model.add(layers.Dropout(rate=hp.Float(f'dropout_{i}', min_value=0, max_value=0.75, step=0.05)))
        
        num_nodes += filters * (28 // (2 ** (i + 1))) * (28 // (2 ** (i + 1)))

    model.add(layers.Flatten())
    dense_units = hp.Int('units', min_value=32, max_value=512, step=32)
    
    # Apply the chosen activation function in the dense layer as well
    model.add(layers.Dense(units=dense_units, activation=activation if activation != 'leaky_relu' else None, kernel_regularizer=keras.regularizers.l2(0.01)))
    
    if activation == 'leaky_relu':
        model.add(layers.LeakyReLU())
    
    model.add(layers.Dropout(rate=hp.Float('dense_dropout', min_value=0, max_value=0.75, step=0.05)))
    model.add(layers.Dense(10, activation='softmax'))

    num_nodes += dense_units + 10  # Add the number of nodes in dense layers

    # Update the lists
    num_nodes_list.append(num_nodes)

    # Define a hyperparameterized learning rate scheduler with decay steps
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG'),
        decay_steps=hp.Int('decay_steps', min_value=1000, max_value=10000, step=1000),
        decay_rate=hp.Float('decay_rate', min_value=0.8, max_value=0.99, step=0.01)
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Define K-Fold Cross-Validation
kfold = KFold(n_splits=7, shuffle=True, random_state=42)

# Loop through the K-Fold splits
val_accuracies = []

for train_idx, val_idx in kfold.split(x_train):
    x_train_fold, x_val_fold = x_train[train_idx], x_train[val_idx]
    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

    # Initialize the tuner
    tuner = kt.Hyperband(
        build_model,
        objective='val_accuracy',
        max_epochs=128, 
        executions_per_trial=1,  # Number of times each model is trained
        factor=2, #Prune rate 
        directory='my_dir',
        project_name='mnist_hypermodel'
    )

    # Define a callback to stop training early if the validation loss does not improve
    stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    # Start the search for the best hyperparameters
    print("==========================================================\nStart the search for the best hyperparameters\n==========================================================")
    tuner.search(x_train_fold, y_train_fold, epochs=1, validation_data=(x_val_fold, y_val_fold))
    
    # Get the optimal hyperparameters
    print("==========================================================\nGet the optimal hyperparameters\n==========================================================")
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Build the model with the best hyperparameters
    print("==========================================================\nBuild the model with the best hyperparameters\n==========================================================")
    model = tuner.hypermodel.build(best_hps)
    model.summary()

    # Train the best model with escape termination callback
    print("==========================================================\nTrain the best model\n==========================================================")
    history = model.fit(
        x_train_fold, y_train_fold, 
        epochs=350, 
        validation_data=(x_val_fold, y_val_fold), 
        callbacks=[stop_early]
    )

    # Evaluate the model on the validation data
    val_loss, val_acc = model.evaluate(x_val_fold, y_val_fold)
    val_accuracies.append(val_acc)
    print(f'Validation accuracy for fold: {val_acc}')

# Calculate the average accuracy across all folds
avg_val_accuracy = np.mean(val_accuracies)
print(f'Average validation accuracy: {avg_val_accuracy}')

# Evaluate the final model on the test data
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')

save_directory = "E:\\Model"

model_path_savedmodel = os.path.join(save_directory, 'best_mnist_model')
model.save(model_path_savedmodel)
