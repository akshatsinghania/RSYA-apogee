import csv

# Define the file path
file_path = 'data.csv'

# Read the original data from the CSV file
with open(file_path, mode='r', newline='') as file:
    reader = csv.reader(file)
    data = list(reader)

# Generate new first column with a constant increment of 0.002
new_data = []
start_value = float(data[0][0])  # Get the starting value from the first row
increment = 0.002

# Loop through the data and create the new first column values
for i, row in enumerate(data):
    new_first_column_value = round(start_value + i * increment, 6)  # Ensure precision
    new_row = [new_first_column_value] + row[1:]  # Replace the first column, keep the second column
    new_data.append(new_row)

# Write the new data back to the CSV file, overwriting the original
with open(file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(new_data)

print("CSV file has been updated with consistent increments of 0.002.")
