import os
import pandas as pd

# Verzeichnis mit den CSV-Dateien
root_dir = "/Users/elshoff/Documents/HAW/IS/Hausarbeit/Historic_Data/tankerkoenig-data/prices"

# Liste für alle DataFrames
df_list = []

# Durchlaufe alle Jahresordner
for year_dir in os.listdir(root_dir):
    if os.path.isdir(os.path.join(root_dir, year_dir)):

        # Liste für alle Monats-DataFrames
        year_df_list = []

        # Durchlaufe alle Monatsordner
        for month_dir in os.listdir(os.path.join(root_dir, year_dir)):
            if os.path.isdir(os.path.join(root_dir, year_dir, month_dir)):

                # Durchlaufe alle CSV-Dateien im Monatsordner
                for filename in os.listdir(os.path.join(root_dir, year_dir, month_dir)):
                    if filename.endswith(".csv"):
                        filepath = os.path.join(root_dir, year_dir, month_dir, filename)
                        print("Aktuelle Datei:", filename)

                        # Lade das CSV in ein DataFrame und füge es der Liste hinzu
                        df = pd.read_csv(filepath)
                        year_df_list.append(df)

        # Führe alle DataFrames im Jahr zusammen
        year_df = pd.concat(year_df_list, axis=0, ignore_index=True)

        # Speichere das Jahr-DataFrame als CSV-Datei
        year_df.to_csv(f"jahr_{year_dir}.csv", index=False, delimiter=",")

        # Lösche das Jahr-DataFrame aus dem Speicher, um Speicherplatz freizugeben
        del year_df

        # Füge das Jahr-DataFrame zur Liste aller DataFrames hinzu
        df_list.append(year_df)

# Führe alle DataFrames in der Liste zusammen
combined_df = pd.concat(df_list, axis=0, ignore_index=True)

# Gib das kombinierte DataFrame aus
print(combined_df.head())

# Speichere das kombinierte DataFrame als CSV-Datei
combined_df.to_csv("combined_data.csv", index=False)
