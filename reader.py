"""
Module to read CSV Files
Will default search data.csv 
"""
#from https://docs.python.org/3/library/csv.html
import csv
matrix=[]
def read():
    with open('data.csv') as csvfile:
        spamreader = csv.reader(csvfile, quotechar='|')
        for row in spamreader:
            Elements=[]
            for element in row:
                Elements.append(element)
            matrix.append(Elements)
        print("File loaded")
read()


