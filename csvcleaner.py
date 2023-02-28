import os
import pandas as pd

import hashlib


def shorten_id(id_str):
    # Erstelle einen MD5-Hash der ID
    hash_object = hashlib.md5(id_str.encode())

    # Konvertiere den Hash in eine hexadezimale Zeichenfolge und kürze sie auf 8 Zeichen
    shorter_id = hash_object.hexdigest()[:8]
    print("ShorterID", shorter_id)
    return shorter_id


# Verzeichnis mit den CSV-Dateien
root_dir = "/Users/elshoff/Documents/HAW/IS/Hausarbeit/Historic_Data/tankerkoenig-data/prices"
station_dir = "/Users/elshoff/Documents/HAW/IS/Hausarbeit/Historic_Data/tankerkoenig-data/stations"
parent_dir = "/Users/elshoff/Documents/HAW/IS/Hausarbeit/FuelPricePredictionRNN/dataset"

# Lade die Datei mit den Stationen
stations_df = pd.read_csv(os.path.join(station_dir, "stations.csv"))

# Füge eine neue Spalte "short_id" mit gekürzten IDs hinzu
stations_df["short_id"] = stations_df["uuid"].apply(shorten_id)

# Speichere die neue stations.csv-Datei
stations_df.to_csv(os.path.join(parent_dir, "stations.csv"), index=False)

# Erstelle ein Mapping von station_uuid zu shorter_id
mapping = stations_df.set_index("uuid")["short_id"]

# Durchlaufe alle Jahresordner
for year_dir in os.listdir(root_dir):
    if os.path.isdir(os.path.join(root_dir, year_dir)):
        try:
            os.mkdir(os.path.join(parent_dir, year_dir))
        except FileExistsError:
            pass
        # Durchlaufe alle Monatsordner
        for month_dir in os.listdir(os.path.join(root_dir, year_dir)):
            try:
                os.mkdir(os.path.join(parent_dir, year_dir, month_dir))
            except FileExistsError:
                pass
            if os.path.isdir(os.path.join(root_dir, year_dir, month_dir)):

                # Durchlaufe alle CSV-Dateien im Monatsordner
                for filename in os.listdir(os.path.join(root_dir, year_dir, month_dir)):
                    if filename.endswith(".csv"):
                        filepath = os.path.join(root_dir, year_dir, month_dir, filename)
                        print("Aktuelle Datei:", filename)

                        # Lade das CSV in ein DataFrame und füge es der Liste hinzu
                        df = pd.read_csv(filepath)
                        df = df.drop(df.columns[-5:], axis=1)
                        df['date'] = pd.to_datetime(df['date'])
                        df['date'] = df['date'].apply(lambda x: x.timestamp())

                        # Ersetze die station_uuids durch shorter_ids
                        df["station_uuid"] = df["station_uuid"].map(mapping)

                        # Erstelle den Pfad zur bereinigten CSV-Datei
                        cleaned_filepath = os.path.join(parent_dir, year_dir, month_dir, f"{filename}clean.csv")

                        # Speichere das bereinigte DataFrame als CSV-Datei im richtigen Verzeichnis
                        df.to_csv(cleaned_filepath, index=False)
