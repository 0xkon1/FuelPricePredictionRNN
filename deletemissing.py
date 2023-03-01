import os
import pandas as pd

root_dir = "/home/je/Documents/HAW/IS/FuelPricePredictionRNN/monthly_data"

# Durchlaufe alle Jahresordner
for year_dir in os.listdir(root_dir):
    if os.path.isdir(os.path.join(root_dir, year_dir)):
        # Durchlaufe alle CSV-Dateien im Jahresordner
        for filename in os.listdir(os.path.join(root_dir, year_dir)):
            if filename.endswith(".csv"):
                filepath = os.path.join(root_dir, year_dir, filename)
                print("Aktuelle Datei:", year_dir,  filename)

                # Lade CSV-Datei in DataFrame
                df = pd.read_csv(filepath)

                # Bereinige Spalte "station_uuid"
                cleaned_df = df.dropna()

                # Schreibe bereinigte Daten in neue CSV-Datei
                cleaned_filepath = os.path.join(root_dir, year_dir, f"{filename}")
                cleaned_df.to_csv(cleaned_filepath, index=False)
