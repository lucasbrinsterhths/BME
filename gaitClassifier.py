import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Data manually entered as a list of dictionaries
data = [
    {"Label": "Adult", "AVG-Step_Time": 0.67, "SD-Step_Time": 0.05, "AVG-A_Gap-V": 0.40, "SD-A_Gap-V": 0.06,
     "AVG-A_Gap-L": 0.38, "SD-A_Gap-L": 0.06, "AVG-A_Gap-A": 0.36, "SD-A_Gap-A": 0.05,
     "AVG-AP_SD-V": 0.14, "SD-AP_SD-V": 0.01, "AVG-AP_SD-L": 0.11, "SD-AP_SD-L": 0.01,
     "AVG-AP_SD-A": 0.11, "SD-AP_SD-A": 0.01, "AVG-DP_SD-V": 0.30, "SD-DP_SD-V": 0.01,
     "AVG-DP_SD-L": 0.02, "SD-DP_SD-L": 0.01, "AVG-DP_SD-A": 0.03, "SD-DP_SD-A": 0.01},

    {"Label": "Adult", "AVG-Step_Time": 0.54, "SD-Step_Time": 0.03, "AVG-A_Gap-V": 0.43, "SD-A_Gap-V": 0.06,
     "AVG-A_Gap-L": 0.49, "SD-A_Gap-L": 0.08, "AVG-A_Gap-A": 0.46, "SD-A_Gap-A": 0.06,
     "AVG-AP_SD-V": 0.18, "SD-AP_SD-V": 0.01, "AVG-AP_SD-L": 0.15, "SD-AP_SD-L": 0.02,
     "AVG-AP_SD-A": 0.14, "SD-AP_SD-A": 0.02, "AVG-DP_SD-V": 0.28, "SD-DP_SD-V": 0.02,
     "AVG-DP_SD-L": 0.03, "SD-DP_SD-L": 0.01, "AVG-DP_SD-A": 0.05, "SD-DP_SD-A": 0.01},

    {"Label": "Adult", "AVG-Step_Time": 0.65, "SD-Step_Time": 0.03, "AVG-A_Gap-V": 0.45, "SD-A_Gap-V": 0.12,
     "AVG-A_Gap-L": 0.42, "SD-A_Gap-L": 0.11, "AVG-A_Gap-A": 0.39, "SD-A_Gap-A": 0.08,
     "AVG-AP_SD-V": 0.15, "SD-AP_SD-V": 0.02, "AVG-AP_SD-L": 0.12, "SD-AP_SD-L": 0.02,
     "AVG-AP_SD-A": 0.12, "SD-AP_SD-A": 0.02, "AVG-DP_SD-V": 0.29, "SD-DP_SD-V": 0.02,
     "AVG-DP_SD-L": 0.03, "SD-DP_SD-L": 0.01, "AVG-DP_SD-A": 0.04, "SD-DP_SD-A": 0.01},

    {"Label": "Adult", "AVG-Step_Time": 0.52, "SD-Step_Time": 0.07, "AVG-A_Gap-V": 0.54, "SD-A_Gap-V": 0.12,
     "AVG-A_Gap-L": 0.53, "SD-A_Gap-L": 0.17, "AVG-A_Gap-A": 0.51, "SD-A_Gap-A": 0.11,
     "AVG-AP_SD-V": 0.19, "SD-AP_SD-V": 0.03, "AVG-AP_SD-L": 0.16, "SD-AP_SD-L": 0.04,
     "AVG-AP_SD-A": 0.15, "SD-AP_SD-A": 0.03, "AVG-DP_SD-V": 0.29, "SD-DP_SD-V": 0.02,
     "AVG-DP_SD-L": 0.04, "SD-DP_SD-L": 0.03, "AVG-DP_SD-A": 0.05, "SD-DP_SD-A": 0.02},

    {"Label": "Child", "AVG-Step_Time": 0.55, "SD-Step_Time": 0.05, "AVG-A_Gap-V": 0.62, "SD-A_Gap-V": 0.21,
     "AVG-A_Gap-L": 0.52, "SD-A_Gap-L": 0.14, "AVG-A_Gap-A": 0.53, "SD-A_Gap-A": 0.18,
     "AVG-AP_SD-V": 0.20, "SD-AP_SD-V": 0.05, "AVG-AP_SD-L": 0.15, "SD-AP_SD-L": 0.04,
     "AVG-AP_SD-A": 0.16, "SD-AP_SD-A": 0.05, "AVG-DP_SD-V": 0.31, "SD-DP_SD-V": 0.02,
     "AVG-DP_SD-L": 0.04, "SD-DP_SD-L": 0.02, "AVG-DP_SD-A": 0.05, "SD-DP_SD-A": 0.01},

    {"Label": "Child", "AVG-Step_Time": 0.64, "SD-Step_Time": 0.09, "AVG-A_Gap-V": 0.89, "SD-A_Gap-V": 0.43,
     "AVG-A_Gap-L": 0.52, "SD-A_Gap-L": 0.17, "AVG-A_Gap-A": 0.80, "SD-A_Gap-A": 0.29,
     "AVG-AP_SD-V": 0.25, "SD-AP_SD-V": 0.08, "AVG-AP_SD-L": 0.16, "SD-AP_SD-L": 0.04,
     "AVG-AP_SD-A": 0.22, "SD-AP_SD-A": 0.05, "AVG-DP_SD-V": 0.29, "SD-DP_SD-V": 0.03,
     "AVG-DP_SD-L": 0.08, "SD-DP_SD-L": 0.03, "AVG-DP_SD-A": 0.07, "SD-DP_SD-A": 0.03},

    {"Label": "Child", "AVG-Step_Time": 0.50, "SD-Step_Time": 0.13, "AVG-A_Gap-V": 1.28, "SD-A_Gap-V": 0.52,
     "AVG-A_Gap-L": 1.05, "SD-A_Gap-L": 0.23, "AVG-A_Gap-A": 0.85, "SD-A_Gap-A": 0.26,
     "AVG-AP_SD-V": 0.36, "SD-AP_SD-V": 0.13, "AVG-AP_SD-L": 0.32, "SD-AP_SD-L": 0.06,
     "AVG-AP_SD-A": 0.28, "SD-AP_SD-A": 0.07, "AVG-DP_SD-V": 0.33, "SD-DP_SD-V": 0.06,
     "AVG-DP_SD-L": 0.15, "SD-DP_SD-L": 0.06, "AVG-DP_SD-A": 0.08, "SD-DP_SD-A": 0.07},

    {"Label": "Child", "AVG-Step_Time": 0.61, "SD-Step_Time": 0.07, "AVG-A_Gap-V": 0.58, "SD-A_Gap-V": 0.12,
     "AVG-A_Gap-L": 0.38, "SD-A_Gap-L": 0.18, "AVG-A_Gap-A": 0.44, "SD-A_Gap-A": 0.07,
     "AVG-AP_SD-V": 0.18, "SD-AP_SD-V": 0.03, "AVG-AP_SD-L": 0.11, "SD-AP_SD-L": 0.04,
     "AVG-AP_SD-A": 0.13, "SD-AP_SD-A": 0.02, "AVG-DP_SD-V": 0.30, "SD-DP_SD-V": 0.02,
     "AVG-DP_SD-L": 0.04, "SD-DP_SD-L": 0.01, "AVG-DP_SD-A": 0.05, "SD-DP_SD-A": 0.01},
]

unknown = [
    {"Label": "Unknown 1", "AVG-Step_Time": 0.52, "SD-Step_Time": 0.13, "AVG-A_Gap-V": 0.65, "SD-A_Gap-V": 0.24,
     "AVG-A_Gap-L": 0.69, "SD-A_Gap-L": 0.23, "AVG-A_Gap-A": 0.76, "SD-A_Gap-A": 0.31,
     "AVG-AP_SD-V": 0.21, "SD-AP_SD-V": 0.06, "AVG-AP_SD-L": 0.19, "SD-AP_SD-L": 0.06,
     "AVG-AP_SD-A": 0.21, "SD-AP_SD-A": 0.08, "AVG-DP_SD-V": 0.30, "SD-DP_SD-V": 0.02,
     "AVG-DP_SD-L": 0.11, "SD-DP_SD-L": 0.05, "AVG-DP_SD-A": 0.07, "SD-DP_SD-A": 0.02},

    {"Label": "Unknown 2", "AVG-Step_Time": 0.59, "SD-Step_Time": 0.06, "AVG-A_Gap-V": 0.39, "SD-A_Gap-V": 0.09,
     "AVG-A_Gap-L": 0.37, "SD-A_Gap-L": 0.10, "AVG-A_Gap-A": 0.48, "SD-A_Gap-A": 0.06,
     "AVG-AP_SD-V": 0.15, "SD-AP_SD-V": 0.02, "AVG-AP_SD-L": 0.11, "SD-AP_SD-L": 0.03,
     "AVG-AP_SD-A": 0.15, "SD-AP_SD-A": 0.01, "AVG-DP_SD-V": 0.28, "SD-DP_SD-V": 0.02,
     "AVG-DP_SD-L": 0.05, "SD-DP_SD-L": 0.02, "AVG-DP_SD-A": 0.05, "SD-DP_SD-A": 0.01},

    {"Label": "Unknown 3", "AVG-Step_Time": 0.52, "SD-Step_Time": 0.15, "AVG-A_Gap-V": 0.69, "SD-A_Gap-V": 0.19,
     "AVG-A_Gap-L": 0.73, "SD-A_Gap-L": 0.20, "AVG-A_Gap-A": 0.98, "SD-A_Gap-A": 0.30,
     "AVG-AP_SD-V": 0.22, "SD-AP_SD-V": 0.05, "AVG-AP_SD-L": 0.20, "SD-AP_SD-L": 0.04,
     "AVG-AP_SD-A": 0.29, "SD-AP_SD-A": 0.08, "AVG-DP_SD-V": 0.29, "SD-DP_SD-V": 0.02,
     "AVG-DP_SD-L": 0.11, "SD-DP_SD-L": 0.05, "AVG-DP_SD-A": 0.11, "SD-DP_SD-A": 0.02},
]

# Convert to DataFrames
df = pd.DataFrame(data)
df_unknown = pd.DataFrame(unknown)

# Split features and labels
X = df.drop(columns=["Label"])
y= df["Label"].map({"Adult": 0, "Child": 1})  # Encode labels

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_scaled, y)

# Prepare unknown data for prediction
X_unknown = df_unknown.drop(columns=["Label"])
X_unknown_scaled = scaler.transform(X_unknown)

# Predict unknown samples
predictions = model.predict(X_unknown_scaled)
probabilities = model.predict_proba(X_unknown_scaled)

# Output predictions
results = df_unknown[["Label"]].copy()
results["Predicted_Label"] = ["Adult" if pred == 0 else "Child" for pred in predictions]
results["Child Probability"] = probabilities[:, 1]

print(results)

# Get coefficients and feature names
coefficients = model.coef_[0]
features = X.columns

# Sort by absolute importance
sorted_idx = np.argsort(np.abs(coefficients))[::-1]
sorted_features = features[sorted_idx]
sorted_coefficients = coefficients[sorted_idx]

# Create color coding: positive = child, negative = adult
colors = ['blue' if coef > 0 else 'red' for coef in sorted_coefficients]

plt.figure(figsize=(12, 6))
plt.barh(sorted_features, sorted_coefficients, color=colors)
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.title('Feature Importance in Gait Classification')
plt.gca().invert_yaxis()  # Highest importance on top
plt.show()