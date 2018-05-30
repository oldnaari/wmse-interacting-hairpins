"""
This file contains functions that describe different regions.
"""


import numpy as np

# Body 
def get_matrix_hb_on_body(energy, temperature):
    return np.array([[np.math.exp(energy / temperature), 0],
                     [0, 1]])

def get_matrix_hb_off_body():
    return np.array([[0, 0],
                     [1, 1]])

def get_matrix_wdw_on_body(energy, temperature):
    return get_matrix_hb_on_body(energy, temperature)

def get_matrix_wdw_off_body():
    return get_matrix_hb_off_body().T

# Tail - Head
def get_matrix_hb_on_head(energy, temperature):
    return np.array([[np.math.exp(energy / temperature), 0],
                     [0, 0]])

def get_matrix_hb_off_head():
    return np.array([[0, 0],
                     [0, 1]])

def get_matrix_wdw_on_head(energy, temperature):
    return get_matrix_hb_on_head(energy, temperature)

def get_matrix_wdw_off_head():
    return get_matrix_hb_off_head()

# Outer
def get_matrix_hb_on_outer(energy, temperature):
    return np.eye(2)

def get_matrix_hb_off_outer():
    return np.eye(2)

def get_matrix_wdw_on_outer(energy, temperature):
    return np.eye(2)

def get_matrix_wdw_off_outer():
    return np.eye(2)


# CELL 3
def body():
    return get_matrix_hb_on_body, get_matrix_hb_off_body, \
           get_matrix_wdw_on_body, get_matrix_wdw_off_body

def head():
    return get_matrix_hb_on_head, get_matrix_hb_off_head, \
           get_matrix_wdw_on_body, get_matrix_wdw_off_body

def tail():
    return get_matrix_hb_on_body, get_matrix_hb_off_body, \
           get_matrix_wdw_on_head, get_matrix_wdw_off_head

def outer():
    return get_matrix_hb_on_outer, get_matrix_hb_off_outer, \
           get_matrix_wdw_on_outer, get_matrix_wdw_off_outer
