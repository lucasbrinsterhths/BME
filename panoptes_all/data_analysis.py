import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import serial
import time

PORT = 'COM3'   # Arduino port
BAUD = 9600     # Baud rate

ser = serial.Serial(PORT, BAUD, timeout=1)  # Instantiate serial connection

time.sleep(2)  # Wait for the connection to initialize

with open('data.csv', 'w') as f:
    while True:
        line = ser.readline().decode('utf-8', errors = 'ignore').strip()   # Read a line from the serial port

        if line:
            print(line)

            if line == "END":  # Check for end signal
                print("Data collection complete.")
                break

            f.write(f"{time.time()},{line}\n")    # Write data to CSV file
            f.flush()               # Ensure data is written to file immediately

ser.close()  # Close the serial connection when done

# Analyze the collected data
data = pd.read_csv('data.csv', names=['timestamp', 'A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'hole'])

# For each sensor, determine the average value for each hole and put it into separate 5 x 6 heatmaps
for sensor in ['A0', 'A1', 'A2', 'A3', 'A4', 'A5']:
    pivot_table = data.pivot_table(index='hole', values=sensor, aggfunc='mean')
    pivot_table = pivot_table.reindex(range(30))  # Ensure all holes are represented
    heatmap_data = pivot_table.values.reshape(5, 6)  # Reshape to 5 x 6

    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap='viridis')
    plt.title(f'Average {sensor} Values by Hole')
    plt.xlabel('Hole Column')
    plt.ylabel('Hole Row')
    plt.show()