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

# Heart Valve Diameter
diameters = [9.9568, 10.1854, 10.668, 9.2202, 11.2776, 10.5664, 10.541, 9.8298, 11.1252, 10.3632, 11.049, 10.3886]  # mm
average_diameter = np.mean(diameters)
flow_rates = [163.4329504, 87.55034978, 80.76628361] # mL/s
flow_velocities = [100 * flow_rate / (np.pi * (average_diameter / 2) ** 2) for flow_rate in flow_rates]

flow_rate_df = pd.DataFrame({
    'Metric': 'Flow Rate',
    'Flow Rate': flow_rates
})

flow_velocity_df = pd.DataFrame({
    'Metric': 'Flow Velocity',
    'Flow Velocity': flow_velocities
})

diameters_df = pd.DataFrame({
    'Metric': 'Diameter',
    'Diameter': diameters
})

print(f"Average Diameter of Heart Valve: {average_diameter:.4f} mm")
print(f"Flow Velocities: {flow_velocities}")

# Plots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
sns.barplot(data=flow_rate_df, ax=axes[0])
axes[0].set_title('Flow Rate through Aorta during Systole')
axes[0].set_ylabel('Flow Rate (mL/s)')
sns.barplot(data=flow_velocity_df, ax=axes[1])
axes[1].set_title('Flow Velocities through Aorta during Systole')
axes[1].set_ylabel('Flow Velocity (cm/s)')
plt.tight_layout()
plt.suptitle('Flow Rates and Velocities through Aorta during Systole')
plt.show()

fig, axes = plt.subplots(2, 1, figsize=(8, 12))
sns.barplot(data=diameters_df, orient='h', ax=axes[0])
axes[0].set_title('Diameter of Heart Valve')
axes[0].set_xlabel('Diameter (mm)')
axes[0].set_xlim(0, 12)
sns.histplot(data=diameters_df, x='Diameter', ax=axes[1], bins=5)
axes[1].set_title('Distribution of Heart Valve Diameters')
axes[1].set_xlabel('Diameter (mm)')
axes[1].set_xlim(0, 12)
max_y = int(axes[1].get_ylim()[1])
axes[1].set_yticks(range(0, max_y + 1, 1))
plt.tight_layout()
plt.show()

water_lost = [5.777164757, 5.400707906, 4.937376396, 3.706652075]   # mL lost 15 seconds after no beat

water_lost_df = pd.DataFrame({
    'Metric': 'Water Lost',
    'Water Lost (mL)': water_lost
})

regurgitation_rate_df = pd.DataFrame({
    'Metric': 'Regurgitation Rate',
    'Regurgitation Rate': [i / 15 for i in water_lost]  # mL/s regurgitation rate
})

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
sns.barplot(data=water_lost_df, ax=axes[0])
axes[0].set_title('Water Lost 15 Seconds After No Beat')
axes[0].set_ylabel('Water Lost (mL)')
sns.barplot(data=regurgitation_rate_df, ax=axes[1])
axes[1].set_title('Regurgitation Rate')
axes[1].set_ylabel('Regurgitation Rate (mL/s)')
plt.tight_layout()
plt.suptitle('Water Loss and Regurgitation Rate After No Beat')
plt.show()
