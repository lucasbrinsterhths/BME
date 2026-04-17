"""
Analyzes the data collected from the Arduino and generates heatmaps for each sensor.
The script reads data from the serial port, saves it to a CSV file, 
and then uses pandas and seaborn to create heatmaps of the average sensor values for each hole. 
Each heatmap represents one of the six sensors (A0 to A5) across a 5 x 6 grid of holes.
"""

import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import serial
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA

PORT = 'COM3'   # Arduino port
BAUD = 9600     # Baud rate

ser = serial.Serial(PORT, BAUD, timeout=1)  # Instantiate serial connection

time.sleep(2)  # Wait for the connection to initialize

with open('data.csv', 'w', encoding='utf-8') as f:
    while True:
        line = ser.readline().decode(
            'utf-8', errors = 'ignore').strip()   # Read a line from the serial port

        if line:
            print(line)

            if line == "END":  # Check for end signal
                print("Data collection complete.")
                break

            f.write(f"{time.time()},{line}\n")    # Write data to CSV file
            f.flush()               # Ensure data is written to file immediately

ser.close()  # Close the serial connection when done

grid = np.array([
    [6, 5, 4, 3, 2, 1],
    [7, 8, 9, 10, 11, 12],
    [18, 17, 16, 15, 14, 13],
    [19, 20, 21, 22, 23, 24],
    [30, 29, 28, 27, 26, 25]
])

hole_to_coord = {}
for row in range(grid.shape[0]):
    for col in range(grid.shape[1]):
        hole_to_coord[grid[row, col]] = (row, col)

# Analyze the collected data
data = pd.read_csv('data.csv', names=['timestamp', 'A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'hole'])

X = data[['A0', 'A1', 'A2', 'A3', 'A4', 'A5']]
y = data['hole']

# 1. Classification Accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

plt.figure()
plt.bar(['Accuracy'], [acc])
plt.title(f'Classification Accuracy: {acc:.3f}')
plt.ylabel('Accuracy')
plt.show()

# 2. Confusion Matrix
cm = confusion_matrix(y_test, preds)

plt.figure()
disp = ConfusionMatrixDisplay(confusion_matrix = cm)
disp.plot()
plt.title('Confusion Matrix')
plt.show()

# 3. F-ratio
f_ratios = []

for sensor in X.columns:
    overall_var = data[sensor].var()
    within_var = data.groupby('hole')[sensor].var().mean()
    between_var = data.groupby('hole')[sensor].mean().var()

    f_ratios.append(between_var / within_var)

plt.figure()
plt.bar(X.columns, f_ratios)
plt.title('F-ratio for Each Sensor')
plt.ylabel('F-ratio')
plt.show()

# 4. Singal-to-Noise Ratio (SNR)
snr_values = []

for sensor in X.columns:
    means = data.groupby('hole')[sensor].mean()
    stds = data.groupby('hole')[sensor].std()
    snr = (means / stds).mean()  # Average SNR across holes
    snr_values.append(snr)

plt.figure()
plt.bar(X.columns, snr_values)
plt.title('Signal-to-Noise Ratio (SNR) for Each Sensor')
plt.ylabel('SNR')
plt.show()

# 5. Spatial Contrast
spatial_contrast = []

for sensor in X.columns:
    contrast = data.groupby('hole')[sensor].mean().std()  # Standard deviation of means across holes
    spatial_contrast.append(contrast)

plt.figure()
plt.bar(X.columns, spatial_contrast)
plt.title('Spatial Contrast for Each Sensor')
plt.ylabel('Spatial Contrast')
plt.show()

# 6. Mutual Information
mi = mutual_info_classif(X, y)

plt.figure()
plt.bar(X.columns, mi)
plt.title('Mutual Information for Each Sensor')
plt.ylabel('Mutual Information')
plt.show()

# 7. Feature Importance from Random Forest
importances = model.feature_importances_

plt.figure()
plt.bar(X.columns, importances)
plt.title('Feature Importance from Random Forest')
plt.ylabel('Importance')
plt.show()

# 8. Correlation Matrix
corr = X.corr()

plt.figure()
sns.heatmap(corr, annot = True)
plt.title('Correlation Matrix of Sensor Readings')
plt.show()

# 9. PCA Explained Variance
pca = PCA()
pca.fit(X)

plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker = 'o')
plt.title('Cumulative Explained Variance by PCA Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()

# 10. Receptive Field Center
centers = []

for sensor in X.columns:
    means = data.groupby('hole')[sensor].mean()

    xs = []
    ys = []
    weights = []

    for hole, val in means.items():
        row, col = hole_to_coord[hole]
        xs.append(col)
        ys.append(row)
        weights.append(val)

    cx = np.sum(np.array(xs) * weights) / np.sum(weights)
    cy = np.sum(np.array(ys) * weights) / np.sum(weights)

    centers.append((cx, cy))

plt.figure()
for i, (cx, cy) in enumerate(centers):
    plt.scatter(cx, cy)
    plt.text(cx, cy, X.columns[i])

plt.gca().invert_yaxis()
plt.title('Receptive Field Centers')
plt.xlabel('Column')
plt.ylabel('Row')

# 11. Receptive Field Width
widths = []

for sensor in X.columns:
    means = data.groupby('hole')[sensor].mean()

    coords = np.array([hole_to_coord[h] for h in means.index])
    weights = means.values

    center = np.average(coords, axis = 0, weights = weights)

    dists = np.linalg.norm(coords - center, axis = 1)
    width = np.sqrt(np.average(dists ** 2, weights = weights))

    widths.append(width)

plt.figure()
plt.bar(X.columns, widths)
plt.title('Receptive Field Width')
plt.show()

# 12. Spatial Smoothness
smoothness = []

for sensor in X.columns:
    means = data.groupby('hole')[sensor].mean()

    total_diff = []

    for hole, val in means.items():
        row, col = hole_to_coord[hole]

        neighbors = [(row + 1, col), (row - 1, col), (row, col + 1), (row, col - 1)]

        for nr, nc in neighbors:
            if 0 <= nr < 5 and 0 <= nc < 6:
                neighbor_hole = grid[nr, nc]
                diff = abs(val - means[neighbor_hole])
                total_diff.append(diff)

    smoothness.append(np.mean(total_diff))

plt.figure()
plt.bar(X.columns, smoothness)
plt.title("2D Spatial Smoothness")
plt.show()

# 13. Temporal Stability
temporal_stability = []

for sensor in X.columns:
    var = data.groupby('hole')[sensor].var().mean()
    temporal_stability.append(var)

plt.figure()
plt.bar(X.columns, temporal_stability)
plt.title('Temporal Stability for Each Sensor')
plt.ylabel('Average Variance')
plt.show()

# 14. Nearest Neighbor Analysis
def grid_distance(h1, h2):
    """Calculate the Manhattan distance between two holes on the grid."""
    r1, c1 = hole_to_coord[h1]
    r2, c2 = hole_to_coord[h2]
    return abs(r1 - r2) + abs(c1 - c2)

TOLERANCE = 1  # Define a tolerance for "neighboring" holes

CORRECT = 0
for true, pred in zip(y_test, preds):
    if grid_distance(true, pred) <= TOLERANCE:
        CORRECT += 1

nn_acc = CORRECT / len(y_test)

plt.figure()
plt.bar(['Nearest Neighbor Accuracy'], [nn_acc])
plt.title(f'Nearest Neighbor Accuracy: {nn_acc:.3f}')
plt.show()

# 15. Spatial Error Distance
errors = []

for true, pred in zip(y_test, preds):
    errors.append(grid_distance(true, pred))

plt.figure()
plt.hist(errors, bins = 10)
plt.title('Spatial Erorr Distribution')
plt.xlabel('Grid Distance Error')
plt.ylabel('Count')
plt.show()

# 16. Heatmaps
for sensor in X.columns:
    means = data.groupby('hole')[sensor].mean()

    heatmap = np.zeroes((5, 6))

    for hole, val in means.items():
        row, col = hole_to_coord[hole]
        heatmap[row, col] = val

    plt.figure()
    sns.heatmap(heatmap, annot = True)
    plt.title(f'{sensor} Spatial Map')
    plt.show()

# 17. Gaussian Fits
def gaussian(coordinates, a, x_0, y_0, spread):
    """Fits a gaussian curve with standard parameters."""
    x_loc, y_loc = coordinates
    return a * np.exp(-((x_loc - x_0) ** 2 + (y_loc - y_0) ** 2) / (2 * spread ** 2))

fits = {}

for sensor in X.columns:
    means = data.groupby('hole')[sensor].mean()

    xs, ys, zs = [], [], []

    for hole, val in means.items():
        x, y = hole_to_coord[hole]
        xs.append(x)
        ys.append(y)
        zs.append(val)

    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)

    p0 = [np.max(zs), 3, 2, 1.5]

    popt, pcov = curve_fit(gaussian, (xs, ys), zs, p0 = p0) # pylint: disable=unpacking-non-sequence

    fits[sensor] = popt

xg = np.linspace(0, 5, 50)
yg = np.linspace(0, 4, 50)
X, Y = np.meshgrid(xg, yg)

for sensor in X.columns:
    A, x0, y0, sigma = fits[sensor]

    Z = gaussian((X, Y), A, x0, y0, sigma)

    plt.figure()
    plt.contourf(X, Y, Z, levels = 20)
    plt.colorbar()
    plt.scatter(*zip(*[(hole_to_coord[h]) for h in data['hole'].unique()]), c = 'red')
    plt.title(f'{sensor} Receptive Field Fit')
    plt.gca().invert_yaxis()
    plt.show()

for sensor, (A, x0, y0, sigma) in fits.items():
    print(sensor)
    print('Center: ', (x0, y0))
    print('Sigma: ', sigma)
    print('Gain: ', A)
    print()
