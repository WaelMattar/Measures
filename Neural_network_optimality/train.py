import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import keras
from keras import layers
from keras.datasets import mnist
from keras.callbacks import Callback
from keras.utils import to_categorical
from keras.utils.layer_utils import count_params
from keras.optimizers import Adam


# Predictions
class Measures(Callback):

    def __init__(self, x_test, y_test, digit):
        super().__init__()
        self.x_test = x_test
        self.y_test = y_test
        self.digit = digit
        self.measures = []

    def on_epoch_end(self, batch, logs=None):
        x_test_ = self.x_test
        y_test_ = self.y_test
        digit_ = self.digit
        x_digit = np.array([x_test_[i] for i in range(len(y_test_)) if y_test_[i] == digit_])

        probabilities = self.model.predict(x=x_digit)
        probabilities = np.mean(probabilities, axis=0)
        self.measures.append(probabilities)

    def get_measures(self):
        return np.vstack(self.measures)


# Configuration
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
y_train = keras.utils.to_categorical(y_train, num_classes)

# Build model
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(2, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(4, kernel_size=(3, 3), activation="relu"),
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
epochs = 200
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train model
measures_instance = Measures(x_test=x_test, y_test=y_test, digit=digit)
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[measures_instance])
results = measures_instance.get_measures()

# Experiment details
string = 'measures_for_digit_{}_with_{}_epochs_{}_batch_and_{}_weights.csv'.format(digit, epochs, batch_size, trainable_count)

# Save measures sequence
df = pd.DataFrame(results)
df.to_csv('Measures_results/'+string, index=False)

# Save training history
hist_df = pd.DataFrame(history.history)
hist_csv_file = 'Training_history/'+string
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

# Plot heatmap
sns.heatmap(df, cmap='seismic')
plt.show()
