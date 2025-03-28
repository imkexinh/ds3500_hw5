import pytest
import numpy as np
from assignta import overallocation, conflicts, undersupport, unavailable, unpreferred

# Load test solutions
test1 = np.loadtxt('test1.csv', delimiter=',', dtype=int)
test2 = np.loadtxt('test2.csv', delimiter=',', dtype=int)
test3 = np.loadtxt('test3.csv', delimiter=',', dtype=int)

def test_overallocation():
    assert overallocation(test1) == 34
    assert overallocation(test2) == 37
    assert overallocation(test3) == 19

def test_conflicts():
    assert conflicts(test1) == 7
    assert conflicts(test2) == 5
    assert conflicts(test3) == 2

def test_undersupport():
    assert undersupport(test1) == 1
    assert undersupport(test2) == 0
    assert undersupport(test3) == 11

def test_unavailable():
    assert unavailable(test1) == 59
    assert unavailable(test2) == 57
    assert unavailable(test3) == 34

def test_unpreferred():
    assert unpreferred(test1) == 10
    assert unpreferred(test2) == 16
    assert unpreferred(test3) == 17