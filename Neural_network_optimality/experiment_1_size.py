import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import keras
import functions as uf
from keras import layers
from keras.datasets import mnist
from keras.callbacks import Callback
from keras.utils import to_categorical
from keras.utils.layer_utils import count_params
from keras.optimizers import Adam


class Measures(Callback):

    def __init__(self, x_train, y_train, digit):
        super().__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.digit = digit
        self.measures = []

    def on_epoch_end(self, batch, logs=None):
        x_train_ = self.x_train
        y_train_ = self.y_train
        digit_ = self.digit
        x_digit = np.array([x_train_[i] for i in range(len(y_train_)) if y_train_[i] == digit_])

        probabilities = self.model.predict(x=x_digit)
        probabilities = np.mean(probabilities, axis=0)
        self.measures.append(probabilities)

    def get_measures(self):
        return np.vstack(self.measures)


def get_optimality_and_size(num_of_filters: int):

    # Configuration
    decomposition_levels = 4
    num_classes = 10
    input_shape = (28, 28, 1)
    digit = 3

    # Load the data and split it between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # Convert class vectors to binary class matrices
    y_train_digits = y_train
    y_train = keras.utils.to_categorical(y_train, num_classes)

    # Build model
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(num_of_filters, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(2*num_of_filters, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    trainable_count = count_params(model.trainable_weights)
    model.summary()

    # Optimizer
    opt = Adam(learning_rate=1e-5)

    # Training configuration
    batch_size = 128
    epochs = 99
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # Train model
    print("Training model number {}".format(number_of_filter-1))
    measures_instance = Measures(x_train=x_train, y_train=y_train_digits, digit=digit)
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[measures_instance])
    results = measures_instance.get_measures()

    # Experiment details
    string = 'measures_for_digit_{}_with_{}_epochs_{}_batch_and_{}_weights.csv'.format(digit, epochs, batch_size, trainable_count)

    # Save measures sequence
    pdf = pd.DataFrame(results)
    pdf.to_csv('Measures_results/'+string, index=False)

    # Save training history
    hist_df = pd.DataFrame(history.history)
    hist_csv_file = 'Training_history/'+string
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

    # Plot heatmap
    plt.figure(num=num_of_filters, figsize=(8, 6))
    sns.heatmap(pdf, cmap='jet')
    plt.savefig('Figures/heat_map_{}.pdf'.format(number_of_filter-1), format='pdf', bbox_inches='tight')
    plt.show()

    # clear model memory
    del model

    # return size and optimality
    return trainable_count, uf.elementary_curve_optimality(curve=results, levels=decomposition_levels)


if __name__ == '__main__':

    size = []
    optimality = []

    for number_of_filter in np.linspace(10, 11, 2, dtype=int):
        s, o = get_optimality_and_size(num_of_filters=number_of_filter)
        size.append(s)
        optimality.append(o)
        print("run: {}, size: {}, optimality: {}".format(number_of_filter-1, size, optimality))

    df = pd.DataFrame({'size': size, 'optimality': optimality})
    df.to_csv('Experiments_results/size_optimality.csv', index=False)

    plt.figure(num=99, figsize=(8, 6))
    plt.plot(size, optimality, linewidth=3, color='blue')
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.savefig('Figures/size_optimality.pdf', format='pdf', bbox_inches='tight')
    plt.show()
