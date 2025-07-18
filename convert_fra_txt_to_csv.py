import csv

with open("fra.txt", "r", encoding="utf-8") as infile, open("dataset.csv", "w", newline="", encoding="utf-8") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["en", "fr"])  # header
    for line in infile:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            en, fr = parts[0], parts[1]
            writer.writerow([en, fr])
