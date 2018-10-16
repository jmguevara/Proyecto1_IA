"""
Module to read CSV Files
Will default search data.csv 
"""
#from https://docs.python.org/3/library/csv.html
import csv

tags=[]
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

#returns n column from the matrix, without the heading
def get_col(col_number):
    array =[]
    for row in matrix:
        array.append(row[col_number])
    return array[1:]
        
def mean(array):
    total=0
    for i in array:
        total=total + float(i)
    return total/len(array)

def standardDeviation(array, mean):
    total=0
    for i in array:
        total += (i- mean)*(i - mean)

    total=total/(len(array)-1)
    total=math.sqrt(total)
    return total




