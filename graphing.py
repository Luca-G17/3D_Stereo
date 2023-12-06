import matplotlib.pyplot as plt
import numpy as np

def bar_graph_data(x, y, w):
    plt.bar(x, y, width=w)
    plt.xlabel("Translation Noise")
    plt.ylabel("Average Radius Error")
    plt.show()

def read_log(filename):
    data = []
    with open(filename, "r") as file:
        for line in file:
            data.append([float(num.strip()) for num in line.split(",")])
    return np.array(data)
#bar_graph_data(data[:,0], data[:,3], 0.4)

data = read_log('logs/log1')
print(data)
bar_graph_data(data[:,0], data[:,3], 0.4)