import numpy as np
import matplotlib.pyplot as plot

from IPython.display import display, clear_output
import ipywidgets as widgets
from ipywidgets import interact, interactive

from moviepy.editor import VideoClip

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
    return get_matrix_hb_on_body, get_matrix_hb_off_body, get_matrix_wdw_on_body, get_matrix_wdw_off_body

def head():
    return get_matrix_hb_on_head, get_matrix_hb_off_head, get_matrix_wdw_on_body, get_matrix_wdw_off_body

def tail():
    return get_matrix_hb_on_body, get_matrix_hb_off_body, get_matrix_wdw_on_head, get_matrix_wdw_off_head

def outer():
    return get_matrix_hb_on_outer, get_matrix_hb_off_outer, get_matrix_wdw_on_outer, get_matrix_wdw_off_outer

# CELL 4
class WMSESegment:
    def __init__(self, segment_length, energy_hb, energy_wdw, entropy_loss, category):
        self.length = segment_length
        self.energy_hb = energy_hb
        self.energy_wdw = energy_wdw
        self.entropy_loss = entropy_loss
        self.category = category

    def get_set_of_transfer_matrices(self, temperature, num_of_vertices):
    
        get_matrix_hb_on, get_matrix_hb_off, get_matrix_wdw_on, get_matrix_wdw_off = self.category()
        
        hb_matrix_on = get_matrix_hb_on(self.energy_hb, temperature)
        hb_matrix_off = get_matrix_hb_off()
        wdw_matrix_on = get_matrix_wdw_on(self.energy_wdw, temperature)
        wdw_matrix_off = get_matrix_wdw_off()    

        hb_matrix = [hb_matrix_off, hb_matrix_on]
        wdw_matrix = [wdw_matrix_off, wdw_matrix_on]

        matrices = np.zeros([2]*(num_of_vertices), dtype=object)
        
        for i in np.ndindex(matrices.shape):
            matrix = hb_matrix[i[0]]
            for j0,j1 in zip(i[1::2], i[2::2]):
                matrix = np.kron(matrix, wdw_matrix[j0])
                matrix = np.kron(matrix, hb_matrix[j1])
            matrices[i] = matrix
            
        return matrices
    
    def get_number_of_conformations_table(self, n_vertices):
        Q = np.math.exp(self.entropy_loss)
        N = n_vertices + 1
        hidden_state_weights = np.zeros([2]*N)
        for i in np.ndindex(hidden_state_weights.shape):
            hidden_state_weights[i] = Q ** (N - sum(i))
                    
        visible_state_weights = np.zeros([2]*(N - 1))
        for i in np.ndindex(hidden_state_weights.shape):
            j = tuple([x*y for x,y in zip(i[:-1], i[1:])])
            
            visible_state_weights[j] += hidden_state_weights[i]
            
        return visible_state_weights
    
    @staticmethod
    def _get_transfer_matrix(state_matrices, state_weights):
        transfer_matrix = 0
        for i in np.ndindex(state_matrices.shape):
            transfer_matrix += state_matrices[i]*state_weights[i]
            
        return transfer_matrix
    @staticmethod
    def _get_per_line_transfer_matrix(state_matrices, state_weights):
        per_line_transfer_matrices = []
        for line, _ in enumerate(state_matrices.shape):
            per_line_transfer_matrices.append([])
            for state in [0, 1]:
                t_matrix = 0
                for i in np.ndindex(state_matrices.shape):
                    if i[line] == state:
                        t_matrix += state_matrices[i]*state_weights[i]
                per_line_transfer_matrices[line].append(t_matrix)
        return per_line_transfer_matrices
    

# CELL 5
class WMSEBetaWave:
    def __init__(self, segments, n_hairpins):
        if type(segments) not in [list, tuple]:
            segments = (segments,)
        self.segments = segments
        self.hairpin_length = sum([x.length for x in segments])
        self.n_hairpins = n_hairpins
    
    def get_bond_probability_map(self, temperature):
        state_matrices = [s.get_set_of_transfer_matrices(temperature, self.n_hairpins) for s in self.segments]
        state_weights = [s.get_number_of_conformations_table(self.n_hairpins) for s in self.segments]
        segment_index_nested = [[i]*s.length for i,s in enumerate(self.segments)]
        segment_index = sum(segment_index_nested, list())
        
        transfer_matrix = [WMSESegment._get_transfer_matrix(state_matrices[s], state_weights[s]) for s,_ in enumerate(self.segments)]
        per_line_transfer_matrices = [WMSESegment._get_per_line_transfer_matrix(state_matrices[s], state_weights[s]) for s,_ in enumerate(self.segments)]
        
        probability_map = np.zeros((self.n_hairpins, self.hairpin_length))
        partition_function_agg = np.eye(transfer_matrix[0].shape[0])
        for i in range(self.hairpin_length):
            partition_function_agg = partition_function_agg.dot(transfer_matrix[segment_index[i]])
        partition_function = np.sum(partition_function_agg)
        for line in range(self.n_hairpins):
            for row in range(self.hairpin_length):
                
                state_agg = np.eye(transfer_matrix[0].shape[0])
                for i in range(self.hairpin_length):
                    if i == row or (i > row) == (line%2==0):
                        state_agg = state_agg.dot(per_line_transfer_matrices[segment_index[i]][line][1])
                    else:
                        state_agg = state_agg.dot(transfer_matrix[segment_index[i]])
                probability_map[line, row] = np.sum(state_agg) / partition_function
        
        return probability_map