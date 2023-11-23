import ast
import csv


def read_csv(file_name):
    """
    Reads a CSV file and returns a list of dictionaries representing each row in the file.
    """
    with open(file_name, 'r', errors='ignore') as csv_file:
        # Use the DictReader class to read the CSV file and create a list of dictionaries
        reader = csv.DictReader(csv_file)
        data = list(reader)
        return data


def read_word_corpus(file_name):
    """
    Reads a CSV file and returns a set of words.
    """
    with open(file_name, 'r', errors='ignore') as csv_file:
        # Use the csv.reader class to read the CSV file and create a set of words
        reader = csv.reader(csv_file, delimiter='\n')
        data = []
        for row in reader:
            data.append(*row)
        return data


def read_preprocessed_csv(file_name):
    """
    Reads a CSV file and returns a list of dictionaries representing each row in the file. The difference here is that the
    values of the 'title' and 'text' keys in the dictionaries are stored as lists of words instead of strings.
    """
    with open(file_name, 'r', errors='ignore') as csv_file:
        # Use the DictReader class to read the CSV file and create a list of dictionaries
        reader = csv.DictReader(csv_file)
        data = list(reader)
        # Convert the 'title' and 'text' values from strings to lists of words
        for x in data:
            x['title'] = ast.literal_eval(x['title'])
            x['text'] = ast.literal_eval(x['text'])
        return data


def write_to_csv(data, name):
    """
    Writes a list of dictionaries to a CSV file.
    """
    with open('assets/' + name + '.csv', 'w') as f:
        writer = csv.writer(f, delimiter=',')
        headers = list(data[0].keys())
        # Write the headers of the CSV file
        writer.writerow(headers)
        # Write each row of data to the CSV file
        for x in range(len(data)):
            writer.writerow(list(data[x].values()))


def save_set_to_csv(my_set, name):
    """
    Writes a set to a CSV file.
    """
    with open('assets/' + name + '.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write each item in the set to a row in the CSV file
        for item in my_set:
            writer.writerow([item])
