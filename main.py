import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

root_dir = "/home/je/Documents/HAW/IS/FuelPricePredictionRNN/monthly_data"


# def create_sequences(values, time_steps=24):
#     """
#     Funktion, um Sequenzen von Input- und Output-Werten zu erstellen
#     values: Ein numpy-Array von Kraftstoffpreisen
#     time_steps: Die Länge der Sequenzen in Stunden (Standard: 24)
#     """
#     input_sequences = []
#     output_sequences = []
#     for i in range(len(values) - time_steps):
#         input_sequences.append(values[i:i + time_steps])
#         output_sequences.append(values[i + time_steps])
#     return np.array(input_sequences), np.array(output_sequences)


batch_size = 64

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(units=64, batch_input_shape=(batch_size, 24, 1), stateful=True),
    tf.keras.layers.Dense(units=1)
])

model.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam())

for year_dir in os.listdir(root_dir):
    if os.path.isdir(os.path.join(root_dir, year_dir)):
        filenames = sorted(
            [filename for filename in os.listdir(os.path.join(root_dir, year_dir)) if filename.endswith(".csv")])
        for i, filename in enumerate(filenames):
            filepath = os.path.join(root_dir, year_dir, filename)
            print("Aktuelle Datei:", filename)

            df = pd.read_csv(filepath)
            # print(df.head())
            diesel_prices = df["diesel"].values
            # print(diesel_prices.shape)
            # diesel_prices = diesel_prices.T
            # print(diesel_prices.shape)
            # print(diesel_prices[:-1])
            train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(diesel_prices[:-1],
                                                                                      diesel_prices[1:],
                                                                                      train_size=0.8,
                                                                                      test_size=0.2,
                                                                                      shuffle=True)
            # print(train_inputs, len(train_inputs))
            train_inputs = np.expand_dims(train_inputs, axis=1)  # Füge Batch-Dimension hinzu
            train_outputs = np.expand_dims(train_outputs, axis=1)  # Füge Batch-Dimension hinzu
            print(len(train_inputs))
            print(len(train_outputs))
            print(len(test_inputs))
            print(len(test_outputs))
            clean_batch = len(train_inputs) % batch_size
            print("kkk", clean_batch)
            print(train_inputs[:-clean_batch].shape)
            train_inputs = np.vsplit(train_inputs[:-clean_batch], len(train_inputs[:-clean_batch])/batch_size)
            train_outputs = np.vsplit(train_outputs[:-clean_batch], len(train_outputs[:-clean_batch]) / batch_size)
            # train_inputs = train_inputs.reshape()
            history = model.fit(train_inputs, train_outputs, validation_split=0.2, epochs=10, batch_size=batch_size,
                                callbacks=[tf.keras.callbacks.TensorBoard(log_dir='./logs'),
                                           tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])

            # test_inputs, test_outputs = create_sequences(test_inputs)
            loss = model.evaluate(test_inputs, test_outputs)
            print("Test Loss:", loss)
