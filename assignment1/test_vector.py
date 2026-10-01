import math
import pytest
from vec import Vec

#test for mean
def test_mean():
    v=Vec([2,4,6,8])
    assert v.mean()==5

def test_mean_with_negative_values():
    v=Vec([-1,-2,3])
    assert v.mean()==0

def test_mean_with_decimal_values():
    v=Vec([1.1,2.2,3.3])
    assert v.mean()==pytest.approx(2.2)

def test_mean_constant_vector():
    v=Vec([5,5,5,5,5])
    assert v.mean()==pytest.approx(5)

def test_mean_single_element():
    v=Vec([10])
    assert v.mean()==10


#test the demean 

def test_demean_basic():
    v=Vec([2, 4, 6, 8])
    result=v.demean()
    assert result.elements==(-3, -1, 1, 3)


def test_demean_mean_is_zero():
    v=Vec([2, 4, 6, 8])
    result=v.demean()
    assert result.mean()==pytest.approx(0)


def test_demean_preserves_length():
    v=Vec([10,20,30,40,50])
    result=v.demean()
    assert len(result.elements)==len(v.elements)


def test_demean_does_not_modify_original():
    v=Vec([2,4,6,8])
    original=v.elements
    v.demean()
    assert v.elements==original


def test_demean_constant_vector():
    v=Vec([5,5,5,5])
    result=v.demean()
    assert result.elements==(0,0,0,0)


def test_demean_negative_values():
    v=Vec([-4,-2,0,2])
    result=v.demean()
    assert result.mean()==pytest.approx(0)

#test std

def test_std_known_value():
    v=Vec([2,4,6,8])
    assert v.std()==pytest.approx(math.sqrt(5))


def test_std_constant_vector_is_zero():
    v=Vec([5,5,5,5])
    assert v.std()==0


def test_std_single_element_is_zero():
    v=Vec([10])
    assert v.std()==0


def test_std_is_non_negative():
    v=Vec([-10,0,10])
    assert v.std()>=0


def test_std_independent_of_order():
    v1=Vec([1,2,3,4])
    v2=Vec([4,3,2,1])
    assert v1.std()==pytest.approx(v2.std())


def test_std_unchanged_by_translation():
    v1=Vec([1,2,3,4])
    v2=Vec([101,102,103,104])
    assert v1.std()==pytest.approx(v2.std())


def test_std_with_negative_values():
    v=Vec([-2,0,2])
    assert v.std()==pytest.approx(math.sqrt(8/3))



def test_demean_values_are_original_minus_mean():
    v = Vec([10,20,30])
    result = v.demean()
    expected_mean=20
    expected=(10-expected_mean,20-expected_mean,30-expected_mean)
    assert result.elements == expected


def test_mean_of_demeaned_vector_is_zero():
    v = Vec([10,20,30,40,50])
    assert v.demean().mean() == pytest.approx(0)


def test_demean_returns_new_vector():
    v = Vec([2,4,6])
    result = v.demean()
    assert isinstance(result, Vec)
    assert result is not v