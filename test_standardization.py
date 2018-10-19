from standardization import get_col
from standardization import mean
from standardization import standardDeviation
from standardization import zscore
from standardization import normalization
from standardization import get_z_matrix


def test_get_col():
    x = get_col(3)
    y = get_col(4)
    z = get_col(5)
    assert get_col(3) == x
    assert get_col(4) == y
    assert get_col(5) == z


def test_mean():
    array=['1','2','3','4','5']
    array2=['6','7','8','9','10']
    array3=['11','12','13','14','15']
    assert mean(array) == 3
    assert mean(array2) == 8
    assert mean(array3) == 13
    

def test_standardDeviation():
     array=['1','2','3','4','5']
     array2=['6','7','8','9','10']
     array3=['11','12','13','14','15']
     mean=3
     mean2=8
     mean3=13
     assert standardDeviation(array,mean) == 1.5811388300841898
     assert standardDeviation(array2,mean2) == 1.5811388300841898
     assert standardDeviation(array,mean) == 1.5811388300841898

def test_zscore():
    mean = 3
    deviation = 1.5811388300841898
    element = '1'
    assert zscore(element,mean,deviation) == -1.2649110640673518

def test_normalization():
    normalization()
    z_matrix=get_z_matrix()
    
    assert z_matrix[1][2] == 1.1

def test_z_matrix():
    normalization()
    z_matrix=get_z_matrix()

    assert z_matrix[2][2] == 1.83
