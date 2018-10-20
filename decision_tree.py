"""
Module to create a decision tree
"""
#https://github.com/danielpang/decision-trees/blob/master/learn.py
import standardization
import sys
import math
import random
import pandas as pd
import standardization as st


z_matrix = pd.read_csv('z_matrix.csv')
headers = st.get_z_matrix()[0][1:]



class Node(object):
	def __init__(self, attribute, threshold):
		self.attr = attribute
		self.thres = threshold
		self.left = None
		self.right = None
		self.leaf = False
		self.predict = None

# First select the threshold of the attribute to split set of test data on
# The threshold chosen splits the test data such that information gain is maximized
def select_threshold(df, attribute, predict_attr):
	# Convert dataframe column to a list and round each value
	values = df[attribute].tolist()
	values = [ float(x) for x in values]
	# Remove duplicate values by converting the list to a set, then sort the set
	values = set(values)
	values = list(values)
	values.sort()
	max_ig = float("-inf")
	thres_val = 0
	# try all threshold values that are half-way between successive values in this sorted list
	for i in range(0, len(values) - 1):
		thres = (values[i] + values[i+1])/2
		ig = info_gain(df, attribute, predict_attr, thres)
		if ig > max_ig:
			max_ig = ig
			thres_val = thres
	# Return the threshold value that maximizes information gained
	return thres_val

# Calculate info content (entropy) of the test data
def info_entropy(df, predict_attr):
	# Dataframe and number of positive/negatives examples in the data
	p_df = df[df[predict_attr] == 1]
	n_df = df[df[predict_attr] == 0]
	p = float(p_df.shape[0])
	n = float(n_df.shape[0])
	# Calculate entropy
	if p  == 0 or n == 0:
		I = 0
	else:
		I = ((-1*p)/(p + n))*math.log(p/(p+n), 2) + ((-1*n)/(p + n))*math.log(n/(p+n), 2)
	return I

# Calculates the weighted average of the entropy after an attribute test
def remainder(df, df_subsets, predict_attr):
	# number of test data
	num_data = df.shape[0]
	remainder = float(0)
	for df_sub in df_subsets:
		if df_sub.shape[0] > 1:
			remainder += float(df_sub.shape[0]/num_data)*info_entropy(df_sub, predict_attr)
	return remainder

# Calculates the information gain from the attribute test based on a given threshold
# Note: thresholds can change for the same attribute over time
def info_gain(df, attribute, predict_attr, threshold):
	sub_1 = df[df[attribute] < threshold]
	sub_2 = df[df[attribute] > threshold]
	# Determine information content, and subract remainder of attributes from it
	ig = info_entropy(df, predict_attr) - remainder(df, [sub_1, sub_2], predict_attr)
	return ig

# Returns the number of positive and negative data
def num_class(df, predict_attr):
	p_df = df[df[predict_attr] == 1]
	print('-13.1----------n')
	n_df = df[df[predict_attr] == 0]
	print('-13.1----------p')
	return p_df.shape[0], n_df.shape[0]

# Chooses the attribute and its threshold with the highest info gain
# from the set of attributes
def choose_attr(df, attributes, predict_attr):
	max_info_gain = float("-inf")
	best_attr = None
	threshold = 0
	# Test each attribute (note attributes maybe be chosen more than once)
	for attr in attributes:
		thres = select_threshold(df, attr, predict_attr)
		ig = info_gain(df, attr, predict_attr, thres)
		if ig > max_info_gain:
			max_info_gain = ig
			best_attr = attr
			threshold = thres
	return best_attr, threshold

# Builds the Decision Tree based on training data, attributes to train on,
# and a prediction attribute
def build_tree(df, cols, predict_attr):
	# Get the number of positive and negative examples in the training data
	print('-13.1----------')
	print(df)
	print('-13.1----------')
	print(cols)
	print('-13.1----------')
	print(predict_attr)
	p, n = num_class(df, predict_attr)
	print(p, n)
	# If train data has all positive or all negative values
	# then we have reached the end of our tree
	if p == 0 or n == 0:
		# Create a leaf node indicating it's prediction
		leaf = Node(None,None)
		leaf.leaf = True
		if p > n:
			leaf.predict = 1
		else:
			leaf.predict = 0
		return leaf
	else:
		# Determine attribute and its threshold value with the highest
		# information gain
		best_attr, threshold = choose_attr(df, cols, predict_attr)
		# Create internal tree node based on attribute and it's threshold
		tree = Node(best_attr, threshold)
		sub_1 = df[df[best_attr] < threshold]
		sub_2 = df[df[best_attr] > threshold]
		# Recursively build left and right subtree
		tree.left = build_tree(sub_1, cols, predict_attr)
		tree.right = build_tree(sub_2, cols, predict_attr)
		return tree

# Given a instance of a training data, make a prediction of healthy or colic
# based on the Decision Tree
# Assumes all data has been cleaned (i.e. no NULL data)
def predict(node, row_df):
	# If we are at a leaf node, return the prediction of the leaf node
	if node.leaf:
		return node.predict
	# Traverse left or right subtree based on instance's data
	if row_df[node.attr] <= node.thres:
		return predict(node.left, row_df)
	elif row_df[node.attr] > node.thres:
		return predict(node.right, row_df)

# 

		

# Prints the tree level starting at given level
def print_tree(root, level):
	print('--------------')
	if root.leaf:
		print('prediccion: ',root.predict,' nivel arbol: ',level)
	else:
		print('atributo: ',root.attr,' nivel arbol: ',level)
	if root.left:
		print_tree(root.left, level + 1)
	if root.right:
		print_tree(root.right, level + 1)

# Cleans the input data, removes 'Diagnosis' column and adds 'Outcome' column
# where 0 means healthy and 1 means colic
def clean(csv_file_name):
	df = pd.read_csv(csv_file_name, header=None)
	print('-1----------')
	print(df)
	print('-2----------')
	print(headers)
	print('-3----------')
	print(print(df.columns))
	print('-4----------')
	df.columns = headers
	print('-5----------')
	# Create new column 'Outcome' that assigns healthy horses a value of 0 (negative case) and
	# horses with colic a value of 1 (positive case), this makes creating our decision tree easier
	print('-6----------')
	df['Outcome'] = 0
	print('-7----------')
	df.loc[df['diagnosis'] == 'M', 'Outcome'] = 1
	print('-8----------')
	df.drop(['diagnosis'], axis=1 )
	print('-9----------')
	cols = df.columns
	print('-10----------')
	df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
	print('-11----------')
	return df
def x_aux(x):
        new_x=[]
        for row in x:
                new_x.append(row[2:])
        return new_x
        
def random_data_sets():
        x=st.get_z_matrix()[1:]#delete headers
        x=x_aux(x)#delete id, result columns
        datasets=[]
        dataset_1=[]
        dataset_2=[]
        dataset_3=[]
        dataset_4=[]
        dataset_5=[]
        for i in range(0,99):
                rnd=random.randint(0,len(x)-1)
                dataset_1.append(x[rnd])
                x.remove(x[rnd])
        for i in range(0,99):
                rnd=random.randint(0,len(x)-1)
                dataset_2.append(x[rnd])
                x.remove(x[rnd])
        for i in range(0,99):
                rnd=random.randint(0,len(x)-1)
                dataset_3.append(x[rnd])
                x.remove(x[rnd])
        for i in range(0,99):
                rnd=random.randint(0,len(x)-1)
                dataset_4.append(x[rnd])
                x.remove(x[rnd])        
        for i in range(0,99):
                rnd=random.randint(0,len(x)-1)
                dataset_5.append(x[rnd])
                x.remove(x[rnd])
        datasets.append(dataset_1)
        #print(dataset_1)
        datasets.append(dataset_2)
        datasets.append(dataset_3)
        datasets.append(dataset_4)
        datasets.append(dataset_5)
        
        return datasets



def trainingForest(sub_data):
        print('-13.1----------')
        forest = []
        print('-13.2----------')
        for index in range (0,5):
                print('-13.3----------')
                print(index)
                root = build_tree(pd.DataFrame(sub_data[index]), headers[1:], 'Outcome')
                print(root)
                print('-13.4----------')
                forest.append(root)
                print('-13.5----------')
        print('-13.6----------')

        return forest


def predictForest(forest, ejemplo):

        predictions=[]
        
        for index in forest:

                prediction=predict(index,ejemplo)
                predictions.append(prediction)


        count1=0
        count0=0

        for index in predictions:
                if index == 1:
                        count1+=1
                else:
                        count0+=1

        if count1 > count0:
                print('M')

        else:
                print('B')


def test_predictions(df_test):
        print('-12----------')
        r_data = random_data_sets()
        print('-13----------')
        forest = trainingForest(r_data)
        print('-14----------')
        for index,row in df_test.iterrows():
                predictForest(forest,row)
                print('-15----------')

        print('-16----------')

def main():
	
	
	df_test = clean('daframe_prueba.csv')
	test_predictions(df_test)


if __name__ == '__main__':
	main()


