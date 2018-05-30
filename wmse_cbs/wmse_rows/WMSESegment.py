
import numpy as np

class WMSESegment:
    def __init__(self, segment_length, energy_hb, energy_wdw, entropy_loss, category):
        """This object describes a number of lines in complex beta structure.
        
        Parameters
        ----------
        segment_length : int
            number of lines in segment
        energy_hb : float 
            energy of bond for odd connections
        energy_wdw : float
            energy of bond for even connections
        entropy_loss : float
            entropy loss for single element native conformation
        category : enum
            use wmse_rows .body/ .head/ .tail/ .outer
        """
        self.length = segment_length
        self.energy_hb = energy_hb
        self.energy_wdw = energy_wdw
        self.entropy_loss = entropy_loss
        self.category = category

    def get_set_of_transfer_matrices(self, temperature, num_of_vertices):
        """Calculates transfer for the given temperature, and number of segments
        0.      *               *
        1.  #########       #########
        2.  #       #       #       #
        3.  #       #       #       #
        4.  #       #       #       #
        5.  #       #########       #
        6.              *
        On the image above each asterix corresponds to one vertex. In this num_of_vertices == 3.

        Parameters
        ----------
        temperature : float
            the temperature in KT (energy) values.
        num_of_vertices : int
            number of vertices in complex beta structure.
        
        Returns
        -------
        ndarray of matrices:
            returns an numpy array of shape [2, 2, ...(num_of_vertices times), 2]. Each element of the array is a transfer matrix.
        """
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
        """[summary]
        
        Parameters
        ----------
        n_vertices : [type]
            [description]
        Returns
        -------
        [type]
            [description]
        """

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