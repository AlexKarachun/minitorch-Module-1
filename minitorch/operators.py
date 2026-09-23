"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$





mul = lambda x, y: x * y

id = lambda x: x

add = lambda x, y: x + y

neg = lambda x: - x

lt = lambda x, y: x < y

eq = lambda x, y: x == y

max = lambda x, y: x if lt(y, x) else y

is_close = lambda x, y: -1e-2 < x - y < 1e-2

sigmoid = lambda x:  1.0 / (1.0 + math.exp(-x)) if x >=0 else math.exp(x) / (1.0 + math.exp(x))

relu = lambda x: max(x, 0)

log = lambda x: math.log(x)

exp = lambda x: math.exp(x)

inv = lambda x: 1 / x

log_back = lambda x, y: y / x 

inv_back = lambda x, y: - y / (x ** 2)

relu_back = lambda x, y: y if x >= 0 else 0












# TODO: Implement for Task 0.1.


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists

def map(f: Callable):
    def map_(li: Iterable):
        return [f(val) for val in li]
    return map_
    
    
def zipWith(f: Callable):
    def zipWith_(li1: Iterable, li2: Iterable):
        return [f(li1[i], li2[i]) for i in range(len(li1))]
    return zipWith_


def reduce(f: Callable, alpha: float):

    def reduce_(li):
        acc = alpha
        for val in li:
            acc = f(acc, val)
        return acc
    return reduce_
    

def negList(li: list):
    return map(neg)(li)
    

def addLists(li1: Iterable, li2: Iterable):
    return zipWith(add)(li1, li2)


def sum(li: list):
    return reduce(add, 0)(li)


def prod(li: list):
    return reduce(mul, 1)(li)

# TODO: Implement for Task 0.3.
