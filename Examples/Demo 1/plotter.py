import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv_file = "/home/wittmann/Programing/Algoritmusok ás Adatszerkezetek hatékony implementálása C nyelven/Házi Feladat/Nagy Hazi Feladat/results.csv"  # Replace with your CSV file path
data = pd.read_csv(csv_file, header=None).replace('', np.nan).astype(float)

data.fillna(data.mean().mean(), inplace=True)

data_array = data.to_numpy()

plt.imshow(data_array, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Temperature (°C)')  
plt.title('Temperature Distribution')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()