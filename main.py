import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

root_dir = "/Users/elshoff/Documents/HAW/IS/Hausarbeit/FuelPricePredictionRNN/monthly_data"

# Erstelle eine leere Liste, in die die DataFrames aller CSV-Dateien gespeichert werden
dfs = []

# Durchlaufe alle Jahresordner
for year_dir in os.listdir(root_dir):
    if os.path.isdir(os.path.join(root_dir, year_dir)):
        # Durchlaufe alle CSV-Dateien im Jahresordner
        for filename in os.listdir(os.path.join(root_dir, year_dir)):
            if filename.endswith(".csv"):
                filepath = os.path.join(root_dir, year_dir, filename)
                print("Aktuelle Datei:", filename)

                # Lade das CSV in ein DataFrame und füge es der Liste hinzu
                df = pd.read_csv(filepath)
                dfs.append(df)

# Fasse alle DataFrames in der Liste zu einem großen DataFrame zusammen
full_df = pd.concat(dfs)

# Nur die Diesel-Preise in ein Numpy-Array umwandeln
# diesel_prices = full_df.groupby('station_id')['diesel'].apply(np.array).values
# print(diesel_prices.head())

# Nur die Diesel-Preise in ein Numpy-Array umwandeln
diesel_prices = full_df["diesel"].values
input_seqs = []
output_seqs = []
for diesel_prices in diesel_prices:
    inputs, outputs = create_sequences(diesel_prices)
    input_seqs.append(inputs)
    output_seqs.append(outputs)


# Zusammenführen aller Input- und Output-Sequenzen in einen großen Datensatz
all_inputs = np.concatenate(input_seqs, axis=0)
all_outputs = np.concatenate(output_seqs, axis=0)

# Aufteilen in Trainings-, Validierungs- und Testsets
train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(all_inputs, all_outputs, test_size=0.2, shuffle=True)

# Definition des RNN-Modells
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(units=64, input_shape=(24, 1)),
    tf.keras.layers.Dense(units=1)
])

# Kompilieren des Modells
model.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam())

# Training des Modells
history = model.fit(train_inputs, train_outputs, validation_split=0.2, epochs=10, batch_size=32)
