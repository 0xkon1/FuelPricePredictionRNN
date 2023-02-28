import os
import pandas as pd

root_dir = "/Users/elshoff/Documents/HAW/IS/Hausarbeit/FuelPricePredictionRNN/monthly_data"
output_dir = "/Users/elshoff/Documents/HAW/IS/Hausarbeit/FuelPricePredictionRNN/sorted_data"

# Durchlaufe alle Jahresordner
for year_dir in os.listdir(root_dir):
    if os.path.isdir(os.path.join(root_dir, year_dir)):
        # Durchlaufe alle CSV-Dateien im Jahresordner
        for filename in os.listdir(os.path.join(root_dir, year_dir)):
            if filename.endswith(".csv"):
                filepath = os.path.join(root_dir, year_dir, filename)
                print("Aktuelle Datei:", filename)

                # Lade das CSV in ein DataFrame
                df = pd.read_csv(filepath)

                # Gruppiere den DataFrame nach Station ID
                station_groups = df.groupby('station_uuid')

                # Speichere jeden Gruppen-DataFrame als eigene CSV-Datei
                for station_uuid, group_df in station_groups:
                    group_filepath = os.path.join(output_dir, f"{station_uuid}.csv")
                    group_df.to_csv(group_filepath, index=False)
