"""
This python file contains tools for creating line based models.

1.      #########       #########
2.      #       #       #       #
3.      #-OH--N-#  ---  #-OH--N-#
4.      #-N--HO-#  ---  #-N--HO-#
5.      #       #       #       #
6.      #       #########       #

The model is defined by a list of lines. Each line is described with tuple of energies.
I.e line 3 can be described with a tuple (E, 0, E)
"""

import numpy as np
import matplotlib.pyplot as plot

from WMSESegment import *
from WMSEBetaWave import *

from wmse_cbs import _tmatrix 

"""
Use this functions to define the category of the line 
Usually
    1-st line is head
    Last line is tail
    Inner lines are body
    Everything else is outer.
"""

body = _tmatrix.body
head = _tmatrix.head
tail = _tmatrix.tail
outer = _tmatrix.outer