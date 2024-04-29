#from typing import Tuple
from glass_utilities import *
from neural import *
from sklearn.model_selection import train_test_split

with open ("glass.data", "r") as f:
    training_data = [parse_line(line) for line in f.readlines() if len(line) > 7]

print(training_data)