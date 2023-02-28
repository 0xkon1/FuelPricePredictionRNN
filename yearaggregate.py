import os
import pandas as pd


def aggregate_by_month(root_dir, parent_dir):
    # Durchlaufe alle Jahresordner
    for year_dir in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, year_dir)):
            # Erstelle den Jahres Ordner
            try:
                os.mkdir(os.path.join(parent_dir, year_dir))
            except FileExistsError:
                pass
            # Durchlaufe alle Monatsordner
            for month_dir in os.listdir(os.path.join(root_dir, year_dir)):
                if os.path.isdir(os.path.join(root_dir, year_dir, month_dir)):
                    # Initialisiere ein leeres DataFrame
                    aggregated_df = pd.DataFrame()

                    # Durchlaufe alle CSV-Dateien im Monatsordner
                    for filename in os.listdir(os.path.join(root_dir, year_dir, month_dir)):
                        if filename.endswith("clean.csv"):
                            filepath = os.path.join(root_dir, year_dir, month_dir, filename)

                            # Lade das bereinigte CSV in ein DataFrame
                            df = pd.read_csv(filepath)

                            # Füge die Daten zum aggregierten DataFrame hinzu
                            aggregated_df = pd.concat([aggregated_df, df], axis=0)

                    print("Monat", month_dir)
                    # Erstelle den Pfad zur aggregierten CSV-Datei
                    aggregated_filepath = os.path.join(parent_dir, year_dir, f"{month_dir}.csv")

                    # Speichere das aggregierte DataFrame als CSV-Datei im richtigen Verzeichnis
                    aggregated_df.to_csv(aggregated_filepath, index=False)


root_dir = "/Users/elshoff/Documents/HAW/IS/Hausarbeit/FuelPricePredictionRNN/dataset"
parent_dir = "/Users/elshoff/Documents/HAW/IS/Hausarbeit/FuelPricePredictionRNN/monthly_data"
aggregate_by_month(root_dir, parent_dir)
