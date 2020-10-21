import csv
import re


def wiki_data_to_csv(input_path, output_path):
    with open(input_path) as input_f, open(output_path, 'w') as output_f:
        writer = csv.writer(output_f, delimiter=';')
        writer.writerow(['Source', 'Target'])

        content = input_f.readlines()
        for line in content:
            line = line.strip()
            if not line.startswith('#'):
                elements = list(map(lambda s: re.sub('\\D', '', s), line.split('\t')))
                writer.writerow(elements)


if __name__ == '__main__':
    wiki_data_to_csv('C:\\Projects\\algorithms-8-tasks\\resources\\task-7\\Wiki-Vote.txt', 'wiki.csv')
