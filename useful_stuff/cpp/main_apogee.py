import subprocess
import matplotlib.pyplot as plt

def run_command_and_store_output():
    # Define the command to be executed
    command = "g++ -o kapogee main_apogee.cpp -lm && ./kapogee < data.txt"
    
    # Run the command and capture the output
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    # Check if the command was successful
    if result.returncode == 0:
        # Process the output
        lines = result.stdout.splitlines()
        if len(lines) >= 2:
            takeoff = float(lines[0])
            apogee = float(lines[1])
            print(f"Takeoff: {takeoff}, Apogee: {apogee}")
            return takeoff, apogee
        else:
            print("Output does not contain enough lines.")
            return None, None
    else:
        print("An error occurred:", result.stderr)
        return None, None



def plot_pressure_vs_time(takeoff, apogee):
    # Read data from file
    time = []
    pressure = []
    
    with open('data.txt', 'r') as file:
        for line in file:
            parts = line.split()
            if len(parts) == 3:
                time.append(float(parts[0]))
                pressure.append(5000-float(parts[2]))
    
    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(time, pressure, label='Pressure')
    
    # Plot takeoff and apogee times
    if takeoff is not None:
        plt.axvline(x=takeoff, color='r', linestyle='--', label='Takeoff')
    if apogee is not None:
        plt.axvline(x=apogee, color='g', linestyle='--', label='Apogee')
    
    plt.title('Pressure vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Pressure')
    plt.legend()
    plt.grid(True)
    plt.show()

# Call the functions
takeoff, apogee = run_command_and_store_output()
plot_pressure_vs_time(takeoff, apogee)
