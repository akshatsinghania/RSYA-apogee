import random

def add_deviation_to_altitude(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    with open(file_path, 'w') as file:
        for line in lines:
            if line.strip():  # Check if the line is not empty
                parts = line.split(',')
                if len(parts) == 2:
                    altitude = float(parts[1])
                    # Calculate a random deviation between -5% and 5%
                    deviation_percentage = random.uniform(-0.05, 0.05)  # -5% to 5%
                    deviation = altitude * deviation_percentage
                    new_altitude = altitude + deviation
                    file.write(f"{parts[0]},{new_altitude:.3f}\n")
                else:
                    file.write(line)  # Write the line as is if it doesn't match the expected format
            else:
                file.write(line)  # Write empty lines as is

# Call the function with the path to your data.csv
add_deviation_to_altitude('data.csv')
# ... existing code ...