import csv

overwrite=""
def convert_csv_to_txt(csv_file, txt_file):
    with open(csv_file, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip the header row
        for row in csvreader:
            if not row[0].startswith('#'):
                # print(row)
                time = float(row[0])
                altitude = 5000-float(row[1])
                # Write to txt file in the format: <time> <acceleration> <pressure>
                # Assuming acceleration is ignored and using altitude as pressure
                overwrite+=f"{time},{altitude}\n"
    with open(txt_file, 'w') as txtfile:
        txtfile.write(overwrite)

# Convert data1.csv to data.txt
convert_csv_to_txt('data.csv', 'data.csv') 