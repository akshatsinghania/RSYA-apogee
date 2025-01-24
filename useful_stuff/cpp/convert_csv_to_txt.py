import csv

def convert_csv_to_txt(csv_file, txt_file):
    with open(csv_file, 'r') as csvfile, open(txt_file, 'w') as txtfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip the header row

        for row in csvreader:
            time = float(row[0])
            altitude = float(row[1])
            # Write to txt file in the format: <time> <acceleration> <pressure>
            # Assuming acceleration is ignored and using altitude as pressure
            txtfile.write(f"{time} -2.000 {altitude}\n")

# Convert data1.csv to data.txt
convert_csv_to_txt('data1.csv', 'data.txt') 