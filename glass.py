# from typing import Tuple
from glass_utilities import *
from neural import *
from sklearn.model_selection import train_test_split

with open ("glass.data", "r") as f:
    training_data = [parse_line(line) for line in f.readlines() if len(line) > 8]

# print(training_data)
trd = normalize(training_data)
# print(trd)

train, test = train_test_split(trd)

nn = NeuralNet(10, 6, 1)
nn.train(train, iters = 10000, print_interval = 1000, learning_rate = 0.2)

print(" ")
print("Types of glass: 0 = Building Windows(Float Processed), 0.2 = Building Windows(Non-Float Processed), 0.4 = Vehicle Windows(Float Processed), 0.6 = Containers, 0.8 = Tableware, 1 = Headlamps")
print(" ")

for i in nn.test_with_expected(test):
    diff = round(abs(i[1][0] - i[2][0]), 6)
    print(f"Desired: {i[1]}, Actual: {i[2]}, Diff: {diff}")