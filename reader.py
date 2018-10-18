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
            elements=[]
            for element in row:
                elements.append(element)
            matrix.append(elements)
        print('File loaded')
read()

def get_matrix():
    return matrix
