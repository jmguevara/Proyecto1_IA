"""
Module to get standarized data set
"""
#https://github.com/danielpang/decision-trees/blob/master/learn.py
import reader
import math
import pandas as pd

z_matrix=[]
matrix = reader.get_matrix()

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
        row.append(matrix[i][0])
        row.append(matrix[i][1])
        for j in range(2,len(matrix[0])):    
            col=get_col(j)
            x=matrix[i][j]            
            m=mean(col)
            d=standardDeviation(col,m)
            row.append(round(zscore(x,m,d),2))
        z_matrix.append(row)
    print('z_matrix created')

def get_z_matrix():
    return z_matrix

def to_csv():
    df = pd.DataFrame(z_matrix)
    df.to_csv("z_matrix.csv", header=None, index=None)
    print('z_matrix.csv created!')






