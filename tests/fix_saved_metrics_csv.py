"""
Script to add columns to saved metrics csv files, for:
    1. the dimension of the system
    2. the number of samples per subdirectory, e.g.
        Lorenz_Rossler could have 100 systems (unique dynamical systems, different parameters)
        whereas Aizawa_Blasius could have 50 systems (unique dynamical systems, different parameters)
        and so on... But currently we aggregate all samples of a subdirectory (skew pair or base dyst name) into a single data point in the csv.
"""

import os

WORK_DIR = os.environ.get("WORK", "")
