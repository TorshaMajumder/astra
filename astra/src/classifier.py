# =========================================================
# Import all dependencies
# =========================================================
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, InputLayer


def mlp_classifier(X_train, y_train, X_test, y_test, input_dim, num_classes, mlp_params):
    """
    Builds, trains, and evaluates a Multi-Layer Perceptron (MLP) classifier.

    Parameters:
    ------------------------------------------------------------------------------------
        X_train (np.ndarray): Scaled training embeddings
        y_train (np.ndarray): Training labels
        X_test (np.ndarray): Scaled testing embeddings
        y_test (np.ndarray): Testing labels
        input_dim (int): The dimension of the input embeddings (e.g., 512)
        num_classes (int): The total number of unique classes
        mlp_params (dict): A dictionary containing hyperparameters for the MLP

    Returns:
    ------------------------------------------------------------------------------------
        tuple: A tuple containing (history, test_accuracy, y_pred)
    """
    # ----------------------------------------------------------------------------------
    # Define the Model Architecture
    # ----------------------------------------------------------------------------------
    model = Sequential()
    model.add(InputLayer(input_shape=(input_dim,)))
    # Add hidden layers based on the config
    for neurons in mlp_params['hidden_layers']:
        model.add(Dense(neurons, activation='relu'))
        model.add(Dropout(mlp_params['dropout_rate']))
    # Add the final output layer
    # Softmax for multi-class classification
    model.add(Dense(num_classes, activation='softmax')) 
    model.summary()
    # ----------------------------------------------------------------------------------
    # Compile the Model
    # ----------------------------------------------------------------------------------
    optimizer = Adam(learning_rate=mlp_params['learning_rate'])
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    # ----------------------------------------------------------------------------------
    # Define Callbacks and EarlyStopping
    # ----------------------------------------------------------------------------------
    early_stopping = EarlyStopping(
                                    monitor='val_loss',
                                    patience=mlp_params['patience'],
                                    verbose=1,
                                    restore_best_weights=True
                                )
    # ----------------------------------------------------------------------------------
    # Train the Model
    # ----------------------------------------------------------------------------------
    history = model.fit(
                        X_train,
                        y_train,
                        epochs=mlp_params['epochs'],
                        batch_size=mlp_params['batch_size'],
                        validation_split=0.1,  # Using 10% of training data for validation during training
                        callbacks=[early_stopping],
                        verbose=1
                    )
    # ----------------------------------------------------------------------------------
    # Evaluate the Final Model on the Hold-out Test Set
    # ----------------------------------------------------------------------------------
    print("\n--- Evaluating on the held-out test split...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n--- Final Test Accuracy: {accuracy * 100:.2f}%")
    # ----------------------------------------------------------------------------------
    # Get Predictions for the Test Set
    # ----------------------------------------------------------------------------------
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    # ----------------------------------------------------------------------------------
    return history, accuracy, y_pred