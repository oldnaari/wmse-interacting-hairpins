import numpy as np

from WMSESegment import WMSESegment

class WMSEBetaWave:
    def __init__(self, segments, n_hairpins):
        """[summary]
        
        Arguments:
            segments {[type]} -- [description]
            n_hairpins {[type]} -- [description]
        """

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