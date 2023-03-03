import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

root_dir = "/home/je/Documents/HAW/IS/FuelPricePredictionRNN/monthly_data"

# Hyperparameter
batch_size = 64
num_epochs = 20
num_features = 1
hidden_size = 128

# Laden der Daten
def load_data(filepath):
    df = pd.read_csv(filepath)
    diesel_prices = df["diesel"].values
    return diesel_prices[:-1], diesel_prices[1:]

# Erstellen des Modells
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(hidden_size, input_shape=(None, num_features), activation='tanh'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam())
    return model

# Ausführen des Trainings
def train_model(model, train_inputs, train_outputs, val_inputs, val_outputs):
    history = model.fit(train_inputs, train_outputs, validation_data=(val_inputs, val_outputs), epochs=num_epochs, batch_size=batch_size, verbose=2)
    return history

# Laden der Daten, Aufteilung in Trainings- und Validierungsdaten
train_inputs_all = []
train_outputs_all = []
val_inputs_all = []
val_outputs_all = []
for year_dir in os.listdir(root_dir):
    if os.path.isdir(os.path.join(root_dir, year_dir)):
        filenames = sorted([filename for filename in os.listdir(os.path.join(root_dir, year_dir)) if filename.endswith(".csv")])
        for i, filename in enumerate(filenames):
            filepath = os.path.join(root_dir, year_dir, filename)
            print("Aktuelle Datei:", filename)
            train_inputs, train_outputs = load_data(filepath)
            num_examples = 1000
            train_inputs = train_inputs[:num_examples]
            train_outputs = train_outputs[:num_examples]
            train_inputs, val_inputs, train_outputs, val_outputs = train_test_split(train_inputs, train_outputs, train_size=0.8, test_size=0.2, shuffle=True)
            train_inputs_all.append(train_inputs)
            train_outputs_all.append(train_outputs)
            val_inputs_all.append(val_inputs)
            val_outputs_all.append(val_outputs)

# Konvertieren der Trainings- und Validierungsdaten in das erwartete Format
train_inputs_all = np.concatenate(train_inputs_all).reshape(-1, 1, num_features)
train_outputs_all = np.concatenate(train_outputs_all).reshape(-1, 1)
val_inputs_all = np.concatenate(val_inputs_all).reshape(-1, 1, num_features)
val_outputs_all = np.concatenate(val_outputs_all).reshape(-1, 1)

# Erstellen und Trainieren des Modells
model = create_model()
history = train_model(model, train_inputs_all, train_outputs_all, val_inputs_all, val_outputs_all)

# Testen des Modells auf den Validierungsdaten
val_predictions = model.predict(val_inputs_all)
val_rmse = sqrt(mean_squared_error(val_outputs_all, val_predictions))
print('Validation RMSE: %.3f' % val_rmse)

# Speichern des Modells
model.save("my_model.h5")
