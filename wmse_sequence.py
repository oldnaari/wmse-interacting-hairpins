import numpy as np
from collections import OrderedDict

from wmse import *

class dir:
    left = 0
    right = 1

    @staticmethod
    def invert(direction):
        return 1 - direction

class WMSESequencialModelBuilder():

    def __init__(self, entropy_loss):
        self.seq = list()
        self.direction = dir.right
        self.pikes = [0]
        self.default_entropy = entropy_loss

    def assert_invalid_values(self):
        Warning('asserting values for WMSESequencialModelBuilder not is not implemented\
                 please be careful')
        return None

    def add_pike(self, length):
        sign = 1 if self.direction == dir.right else -1
        self.pikes.append(self.pikes[-1] + length*sign)
        self.direction = dir.invert(self.direction)
        self.assert_invalid_values()
        return self

    def add_connection(self, pike_ind, start, width, energy, entropy, mark=None):
        
        if type(pike_ind) != int:
            raise ValueError('pike_ind must be an integer')

        if pike_ind < 0:
            raise ValueError('pike_ind must be positive')

        if pike_ind >= len(self.pikes):
            raise ValueError('pike of index %d [pike_ind] not defined. Try using add_pike' % pike_ind)

        if pike_ind == 0 or pike_ind==len(self.pikes)-1:
            raise ValueError('''pike_ind must correspond to inner pikes of the sequence. Sequence edges are not valid''')

        self.seq.append((pike_ind, start, width, energy, entropy, mark))
        self.assert_invalid_values()
        return self

class _WMSESequentialSegment(WMSESegment):
    
    def __init__(self, segment_length, line_e, line_s, list_of_types):
        if len(np.unique(line_s)) > 1:
            raise NotImplementedError("The functionality of several entropies in a row yet is not implemented")
        self.length = segment_length
        self.entropy_loss = line_s[0]
        self.line_e = line_e.tolist()
        self.list_of_types = list_of_types

    def get_set_of_transfer_matrices(self, temperature, num_of_vertices):
        
        # get_matrix_hb_on, get_matrix_hb_off, get_matrix_wdw_on, get_matrix_wdw_off = body()
        Warning('''The edge effects are yet beeing ignored, planned to add  them in the future''')
        
        matrix_on = [_type()[0] if i % 2 == 0 else _type()[2] for i, _type in enumerate(self.list_of_types)]
        matrix_off = [_type()[1] if i % 2 == 0 else _type()[3] for i, _type in enumerate(self.list_of_types)]
       
        matrix_on = [matrix_on[i](e, temperature) for i, e in enumerate(self.line_e)]
        matrix_off = [matrix_off[i]() for i, e in enumerate(self.line_e)]
        
        matrix = [(m_off, m_on) for m_on, m_off in zip(matrix_on, matrix_off)]

        matrices = np.zeros([2]*(num_of_vertices), dtype=object)
        for i in np.ndindex(matrices.shape):
            m = matrix[0][i[0]]
            for column_number, column_state in enumerate(i[1:]):
                m = np.kron(m, matrix[column_number + 1][column_state])
            matrices[i] = m
            
        return matrices


        
class WMSESequencialModel():

    @staticmethod
    def builder(*args, **kwargs):
        return WMSESequencialModelBuilder(*args, **kwargs)

    def __init__(self, builder):
        cursor_pos = [0]
        min_h = min(builder.pikes)
        max_h = max(builder.pikes)
        self.energy_values = np.zeros((max_h - min_h, len(builder.pikes) - 2))
        self.entropy_values = np.ones((max_h - min_h, len(builder.pikes) - 2))*builder.default_entropy
        self.existance_values = np.zeros((max_h - min_h, len(builder.pikes) - 2))
        self.mark_values = np.full((max_h - min_h, len(builder.pikes) - 2), None)  
        self.type_values = np.full((max_h - min_h, len(builder.pikes) - 2), outer, dtype=object)

        for i in range(len(builder.pikes) - 2):
            c0, c1, c2 = builder.pikes[i], builder.pikes[i + 1], builder.pikes[i + 2]
            if c1 > c0:
                max_i = c1 - min_h
                min_i = max(c0, c2) - min_h
            else:
                min_i = c1 - min_h
                max_i = min(c0, c2) - min_h
            self.existance_values[min_i:max_i, i] = True

        for pike_ind, start, width, energy, entropy, mark in builder.seq:
            if pike_ind % 2 == 1:
                min_limit = builder.pikes[pike_ind] - start - width - 1
                max_limit = builder.pikes[pike_ind] - start
            else:
                min_limit = builder.pikes[pike_ind] + start
                max_limit = builder.pikes[pike_ind] + start + width + 1

            min_limit -= min_h
            max_limit -= min_h
            self.energy_values[min_limit:max_limit, pike_ind - 1] = energy
            self.entropy_values[min_limit:max_limit, pike_ind - 1] = entropy
            self.existance_values[min_limit:max_limit, pike_ind - 1] = True
            self.mark_values[min_limit:max_limit, pike_ind - 1] = mark
        
        for i, j in np.ndindex(self.existance_values.shape):
            if self.existance_values[i, j]:
                if i == 0:
                    self.type_values[i, j] = tail
                    continue
                if i == self.existance_values.shape[0] - 1:
                    self.type_values[i, j] = head
                    continue
                if not self.existance_values[i + 1, j]:
                    self.type_values[i, j] = head
                if not self.existance_values[i - 1, j]:
                    self.type_values[i, j] = tail

                self.type_values[i, j] = body
            # else:
            #     self.entropy_values[i, j] = 1

        self.layers = list()

        # First layer
        line_s, line_e, line_t = self.entropy_values[0, :], self.energy_values[0, :], self.type_values[0, :]
        line_n_prev = 0

        # Push a layer if met a change
        for line_n in range(1, self.entropy_values.shape[0]):

            has_change = (np.any(line_s != self.entropy_values[line_n, :] ) or
                          np.any(line_e != self.energy_values[line_n, :]) or
                          np.any(line_t != self.type_values[line_n, :]))

            if has_change:
                self.layers.append(_WMSESequentialSegment(line_n - line_n_prev,
                                   line_e, line_s, line_t))
                line_n_prev = line_n
                line_s = self.entropy_values[line_n, :]
                line_e = self.energy_values[line_n, :]
                line_t = self.type_values[line_n, :]
                continue

        # Push the remainder
        self.layers.append(_WMSESequentialSegment(self.entropy_values.shape[0] - line_n_prev,
                    line_e, line_s, line_t))
        self.calculator = WMSEBetaWave(self.layers, len(line_e))


    def get_probabilty_map(self, temperature):
        return self.calculator.get_bond_probability_map(temperature)

    def get_link_probability(self, temperature):
        probability_map = self.get_probabilty_map(temperature)

        results = {}
        for j in np.unique(self.mark_values):
            if j is None:
                continue
            mask = (self.mark_values == j)
            results[j] = (probability_map.transpose()*mask).sum() / mask.sum()

        return results

    def get_condition_temperature(self, bond_strength, bond_mark, t1 = 0.1, t2 = 1.0, t_eps = 0.01):
        bond_t = lambda t: self.get_link_probability(t)[bond_mark]

        bond1 = bond_t(t1)
        bond2 = bond_t(t2)

        assert bond1 > bond_strength > bond2, '''Out of range of temperatures [%f, %f]. Try a wider range (current values = { %f, %f })''' % (t1, t2, bond1, bond2)

        while t2 - t1 > t_eps:
            t_12 = (t1 + t2)/2
            bond_12 = bond_t(t_12)
            if bond_12 > bond_strength:
                bond1 = bond_12
                t1 = t_12
            else:
                bond2 = bond_12
                t2 = t_12

        return (t1 + t2)/2

    def visualize(self, temperature, auto_display=True):
        import matplotlib.pyplot as plot        
        pm = self.get_probabilty_map(temperature).T

        draw_image = np.ones((pm.shape[0]*3, pm.shape[1]*2 + 1, 3))

        mark_colors = [np.array([0.0, 1.0, 0.0]),
                       np.array([0.0, 0.5, 1.0]),
                       np.array([1.0, 0.5, 0.0]),
                       np.array([1.0, 0.0, 0.0]),
                       np.array([0.0, 0.0, 1.0]),
                       np.array([1.0, 1.0, 0.0])]

        color_black = np.array([0.2, 0.2, 0.2])
        color_white = np.array([1.0, 1.0, 1.0])

        markers = list(np.unique(self.mark_values))
        for i, j in np.ndindex(self.existance_values.shape):
            if self.existance_values[i, j]:
                right_cup = True if (j%2 == 0 and ((i == self.existance_values.shape[0] - 1) or 
                                                    not self.existance_values[i + 1, j])) else False
                left_cup = True if (j%2 == 1 and ((i == 0) or not self.existance_values[i - 1, j])) else False 


                medium_x = 3*i + 1
                medium_y = 2*j + 1
                draw_image[medium_x - 1, medium_y - 1, :] = color_black
                draw_image[medium_x    , medium_y - 1, :] = color_black
                draw_image[medium_x + 1, medium_y - 1, :] = color_black
                draw_image[medium_x - 1, medium_y + 1, :] = color_black
                draw_image[medium_x    , medium_y + 1, :] = color_black
                draw_image[medium_x + 1, medium_y + 1, :] = color_black

                if left_cup: draw_image[medium_x - 1, medium_y, :] = color_black
                if right_cup: draw_image[medium_x + 1, medium_y, :] = color_black
                
                mark = self.mark_values[i,j]
                if mark is None: continue
                draw_image[medium_x, medium_y, :] = pm[i, j]*mark_colors[markers.index(mark)] + (1.0 - pm[i,j])*color_white

        
        fig = plot.imshow(draw_image[::-1,:,:])
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        if auto_display:
            plot.show()
        return fig



