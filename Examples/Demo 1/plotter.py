import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Parancssori argumentumok beállítása
parser = argparse.ArgumentParser(description='Hőtérképes adatmegjelenítés')
parser.add_argument('csv_file', type=str, help='Az input CSV fájl elérési útvonala')
parser.add_argument('--title', type=str, default='Temperature Distribution', help='A plot címe')
parser.add_argument('--interpolation', type=str, help='Az interpoláció típusa (pl. nearest, bilinear, bicubic, stb.)')
args = parser.parse_args()

# CSV fájl betöltése az argumentumból
csv_file = args.csv_file
data = pd.read_csv(csv_file, header=None).replace('', np.nan).astype(float)

# Hiányzó adatok kitöltése az átlaggal
data.fillna(data.mean().mean(), inplace=True)

# Adatok numpy tömbbe való átalakítása
data_array = data.to_numpy()

# Hőtérkép megjelenítése
plt.imshow(data_array, cmap='viridis', interpolation=args.interpolation if args.interpolation else None)
plt.colorbar(label='Temperature (°C)')
plt.title(args.title)  # A plot címe az argumentumból
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()
