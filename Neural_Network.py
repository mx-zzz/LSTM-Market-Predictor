import os
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

CATEGORY_DEPTH = 30

def save_data_slices(X, prefix, num_slices_to_save=10):
    dimensions = X.ndim
    if dimensions == 3:
        a, b, c = X.shape
        reshaped_data = X.reshape(a, b * c)
        for i, data_slice in enumerate(reshaped_data):
            if i >= num_slices_to_save: break
            np.savetxt(f"{prefix}_{i}.csv", data_slice.reshape(b, c), delimiter=",")
    elif dimensions == 2:
        for i, data_slice in enumerate(X):
            if i >= num_slices_to_save: break
            np.savetxt(f"{prefix}_{i}.csv", data_slice, delimiter=",")
    else:
        raise ValueError(f"Unexpected number of dimensions in {prefix}. Expected 2 or 3, got {dimensions}")

def create_lstm_model():
    model = Sequential()
    model.add(LSTM(100, activation='tanh', input_shape=(100, 7), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense((2*CATEGORY_DEPTH)+1, activation='softmax'))
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')
    return model

def train_lstm_model(model, X_train, y_train, epochs):
    model.fit(X_train, y_train, epochs=epochs, batch_size=128, verbose=0, validation_split=0.2)

def load_and_preprocess_data(data_dir):
    X = []
    y = []
    column_names = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
    for filename in os.listdir("Training Data"):
        try:
            data = pd.read_csv("Training Data\\" + filename)
        except ValueError as e:
            raise ValueError(f"Error processing file {filename}: {str(e)}")
        data = data[1:]
        for i in range(100, len(data)):
            X.append(data.iloc[i - 100:i, :7].values)
            y.append(((data.iloc[i, 3] - data.iloc[i - 1, 3]) / (data.iloc[i - 1, 3] + 0.00000000001 )) * 100)

    X = np.array(X)
    y = np.array(y)
    y = np.clip(y, -1*CATEGORY_DEPTH, CATEGORY_DEPTH)
    y = np.round(y)
    y = y + CATEGORY_DEPTH
    y = to_categorical(y.astype(int), num_classes=(2 * CATEGORY_DEPTH) + 1)
    X, y = shuffle(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    save_data_slices(X_train, "Debug Info\\X_train")
    save_data_slices(X_test, 'Debug Info\\X_test')
    save_data_slices(y_train, 'Debug Info\\y_train')
    save_data_slices(y_test, 'Debug Info\\y_test')

    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test, num_samples=140):
    shuffled_indices = np.random.permutation(len(X_test))
    selected_indices = shuffled_indices[:num_samples]
    predictions = model.predict(X_test[selected_indices])
    expected_values = [np.sum(prediction*(j - CATEGORY_DEPTH)) for prediction in predictions]
    predicted_percentage_change = (np.argmax(predictions, axis=1) - CATEGORY_DEPTH)
    actual_percentage_change = (np.argmax(y_test[selected_indices], axis=1) - CATEGORY_DEPTH)

    print_prediction_details(actual_percentage_change, predicted_percentage_change, expected_values)
    calculate_mean_squared_error(actual_percentage_change, predicted_percentage_change)

def print_prediction_details(actual, predicted, expected_values):
    print("    Predicted    Result       Error     E(X)")
    print("---------------------------------------------------")
    for i in range(len(predicted)):
        error = np.abs(predicted[i] - actual[i])
        print(f"{i + 1}\t{predicted[i]:.2f}%\t\t{actual[i]:.2f}%\t\t{error:.2f}%\t\t{expected_values[i]:.2f}")

def calculate_mean_squared_error(actual, predicted):
    error = np.abs(predicted - actual)
    mse = np.square(error).mean()
    print(f"\nMean Squared Error for the test set: {mse:.3f}")
