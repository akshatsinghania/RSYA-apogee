import numpy as np
import pandas as pd

# Load the data from the CSV file, skipping any comment lines
csv_file_path = 'altdata.csv'  # Path to your CSV file
df = pd.read_csv(csv_file_path, comment='#')

# Display the column names to verify
print(df.columns)

# Assuming the columns are 'time' and 'altitude'
time_data = df['time'].values
altitude_data = df['altitude'].values

# Introduce random variation of ±5% to the altitude data
variation = np.random.uniform(-0.01, 0.01, altitude_data.shape)
modified_altitude_data = altitude_data * (1 + variation)

# Convert the modified data back to a DataFrame
modified_df = pd.DataFrame({
    'time': time_data,
    'altitude': modified_altitude_data
})

# Save the modified data to a new CSV file
modified_df.to_csv('data1.csv', index=False)

# Optionally, print or display the first few rows of the modified data
print(modified_df.head())  # Print the first few rows of the modified data
