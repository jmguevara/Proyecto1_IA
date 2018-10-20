from decision_tree import Node
from decision_tree import select_threshold
from decision_tree import info_entropy
from decision_tree import remainder
from decision_tree import info_gain
from decision_tree import num_class
from decision_tree import choose_attr
from decision_tree import build_tree
from decision_tree import predict
from decision_tree import print_tree
from decision_tree import clean
from decision_tree import x_aux
from decision_tree import random_data_sets
from decision_tree import trainingForest
from decision_tree import predictForest



def test_node():
    nodeT = Node('positivo', None)
    assert nodeT.attr == 'positivo'
    assert nodeT.thres == None


