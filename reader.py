"""
Module to read CSV Files
Will default search data.csv 
"""
#from https://docs.python.org/3/library/csv.html
import csv
import math
matrix=[]
z_matrix=[]

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

#get mean from an array
def mean(array):
    total=0
    for i in array:
        total=total + float(i)
    return total/len(array)

def standardDeviation(array, mean):
    total=0
    for i in array:
        total += (float(i)- float(mean))*(float(i) - float(mean))

    total=total/(len(array)-1)
    total=math.sqrt(total)
    return total

def zscore(element,mean,deviation):
    return (float(element)-float(mean))/float(deviation)

#create a new 'z_matrix' with each value normalized
def normalization():
    z_matrix.append(matrix[0])
    for i in range(1,len(matrix)):
        row=[]
        for j in range(2,len(matrix[0])):
            row.append(matrix[0][j])
            row.append(matrix[1][j])
            col=get_col(j)
            x=matrix[i][j]            
            m=mean(col)
            d=standardDeviation(col,m)
            row.append(round(zscore(x,m,d),2))
        z_matrix.append(row)
    print('Done')
