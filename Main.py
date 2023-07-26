import os

from keras.models import load_model
from Neural_Network import load_and_preprocess_data, create_lstm_model, train_lstm_model, evaluate_model

def main():
    while True:
        print("\n---------------------")
        print("1. Train model")
        print("2. Test model")
        print("3. Save model")
        print("4. Exit")
        print("---------------------")

        option = input("Please choose an option: ")

        if option == "1":
            X_train, X_test, y_train, y_test = load_and_preprocess_data("Training Data\\")
            model = create_lstm_model()
            train_lstm_model(model, X_train, y_train, epochs)
            model.save("Saved Models\\AutoSave")
            print("Model trained and saved successfully!")

        elif option == "2":
            X_train, X_test, y_train, y_test = load_and_preprocess_data("Training Data\\")
            model = load_model("Saved Models\\AutoSave")
            evaluate_model(model, X_test, y_test)
            print("Model tested successfully!")

        elif option == "3":
            model = create_lstm_model()
            model_file = input("Please enter the model filename to save as: ")
            model.save(os.path.join('Saved Models\\', model_file))
            print("Model saved successfully!")

        elif option == "4":
            print("Exiting program...")
            break

        else:
            print("Invalid option, please try again.")

epochs = 100

if __name__ == "__main__":
    main()
