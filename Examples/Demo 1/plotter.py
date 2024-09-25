import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV data using pandas
csv_file = "/home/wittmann/Programing/Algoritmusok ás Adatszerkezetek hatékony implementálása C nyelven/Házi Feladat/Nagy Hazi Feladat/results.csv"  # Replace with your CSV file path
data = pd.read_csv(csv_file, header=None).replace('', np.nan).astype(float)

# Replace NaN values (missing data) with a suitable value, e.g., the mean temperature or 0
data.fillna(data.mean().mean(), inplace=True)

# Convert the pandas DataFrame to a NumPy array
data_array = data.to_numpy()

# Create a heatmap plot
plt.imshow(data_array, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Temperature (°C)')  # Add a color bar with label

# Add title and labels
plt.title('Temperature Distribution')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')

# Display the plot
plt.show()

