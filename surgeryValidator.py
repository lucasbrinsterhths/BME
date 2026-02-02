import numpy as np

def generate_normal_data(mean=0.0, std_dev=1.0, size=100000000):
    """Generate synthetic data following a normal distribution.

    Args:
        mean (float): The mean of the normal distribution.
        std_dev (float): The standard deviation of the normal distribution.
        size (int): The number of data points to generate.

    Returns:
        np.ndarray: An array of synthetic data points.
    """
    data = np.random.normal(loc=mean, scale=std_dev, size=size)
    return data

healthy = generate_normal_data(mean=1.09, std_dev=0.07)
injured = generate_normal_data(mean=1.14, std_dev=0.12)

true_positive, false_positive, true_negative, false_nagative = 0, 0, 0, 0

for patient in healthy:
    if patient <= 1.115:
        true_negative += 1
    else:
        false_positive += 1

for patient in injured:
    if patient > 1.115:
        true_positive += 1
    else:
        false_nagative += 1

print(f"True Positives: {true_positive}")
print(f"False Positives: {false_positive}")
print(f"True Negatives: {true_negative}")
print(f"False Negatives: {false_nagative}")

print(f"Total Accuracy: {(true_positive + true_negative) / (true_positive + true_negative + false_positive + false_nagative):.4f}")