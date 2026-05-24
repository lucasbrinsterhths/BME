"""
heartValveDA.py
Analyzes stress-strain data from Shore 15A silicone tensile tests.
Shore 15A used for our prosthetic leaflets; imporant to understand its elasticity.
"""

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression

INITIAL_LENGTH_MM = 22.5  # Initial length of silicone sample in mm
CROSS_SECTIONAL_AREA_M2 = (17.7 / 1000) * (5 / 1000) # Cross-sectional area in m^2
GRAVITY = 9.80665  # Acceleration due to gravity in m/s^2

# Data dictionary: keys are masses in grams, values are silicone lengths (mm)
data = {
    0: 22.5,
    20: 23,
    100: 23.4,
    200: 24.1,
    500: 27.3,
    1000: 34.6,
    1500: 44.7
}

df = pd.DataFrame(list(data.items()), columns=['mass_g', 'length_mm'])

# Calculate force (N), stress (Pa), and strain
df['force_N'] = df['mass_g'] * GRAVITY / 1000  # Convert g to kg and multiply by gravity
df['stress_MPa'] = df['force_N'] / CROSS_SECTIONAL_AREA_M2 / 1e6  # Stress in MPa
df['strain'] = (df['length_mm'] - INITIAL_LENGTH_MM) / INITIAL_LENGTH_MM  # Strain

# Least-squares regression
X = df['strain'].values.reshape(-1, 1)
y = df['stress_MPa'].values

model = LinearRegression(fit_intercept=False)  # Force line through origin
model.fit(X, y)

np.random.seed(42)
x_data = np.random.uniform(-1, max(df['strain']), 1000)
y_data = model.coef_[0] * x_data + np.random.normal(0, 0.5, size=x_data.shape)

x_line = np.linspace(min(x_data), max(x_data), 1000)
y_line = model.coef_[0] * x_line

sns.scatterplot(x='strain', y='stress_MPa', data=df)
plt.plot(x_line, y_line, color='red', label='Linear Fit')
plt.title('Experimentally-Determined Stress-Strain Curve for Shore 15A Silicone '
'| Young\'s Modulus = ' + f"{model.coef_[0]:.2f}" + ' MPa')
plt.xlim(left = 0)
plt.ylim(bottom = 0)
plt.xlabel('Strain')
plt.ylabel('Stress (MPa)')
plt.show()
