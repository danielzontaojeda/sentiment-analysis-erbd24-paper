import numpy as np

# Define which counters to use, the options are:
# CC
# ACC
# PCC
# PACC
# X
# MAX
# T50
# MS
# MS2
# HDy
# DyS
# SORD
# SMM
COUNTERS = (
    "CC",
    "ACC",
    "X",
    "MAX",
    "T50",
    "MS",
    "MS2",
    "HDy",
    "DyS",
    "SORD",
    "SMM"
)

# Define how many iterations each sample from the test portion of the dataset will run. This is to make sure a quantifier doesn't get 'lucky' and is tested with a easier subset
N_ITERATIONS = 10

# Define different test sizes for samples
BATCH_SIZES = [10, 20, 30, 40, 50, 100, 200]

# Define different positive proportions for the samples
ALPHA_VALUES = [round(x, 2) for x in np.linspace(0, 1, 21)]

# Method to measure distance, options are:
# topsoe
# hellinger
# probsymm
MEASURE = "topsoe"
