# LSTM Stock Market Predictor

LSTM Stock Market Predictor is a Python program that uses a Long Short-Term Memory (LSTM) neural network to predict future stock prices based on historical data. The tool ingests data in CSV format and produces both a predicted trend as well as a probable future stock price.

This model is trained using Keras with a TensorFlow backend. 

## Features

- Uses LSTM, a type of Recurrent Neural Network (RNN), which is particularly good at processing sequential data.
- The model ingests seven features: 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Revenue'
- The output is a percentage change in the closing price.
- The model can be trained on any stock data provided in the correct CSV format.
- Allows saving and loading of trained models for future use.
- Option to evaluate the model on a testing set.

## Requirements

To run the model, you'll need:

- Python 3.6 or higher
- Keras
- TensorFlow
- NumPy
- Pandas
- scikit-learn

To install the required libraries, run:

pip install keras tensorflow numpy pandas scikit-learn

vbnet
Copy code

## How to Use

1. Clone the repository to your local machine.
2. Ensure the requirements are installed in your Python environment.
3. Place your CSV files in the `Training Data` directory.
4. Run the `main.py` file in your Python environment.
5. Follow the prompts to train or test the model:

    ```
    1. Train model
    2. Test model
    3. Save model
    4. Exit
    ```

Note: When training the model, an "AutoSave" model will be created after every training session.

## Disclaimer

This project is intended for educational and research purposes. It is not a tool for financial advice. Stock market investments carry risk and investors should only invest what they can afford to lose. Always consult with a professional financial advisor before making any investments.
