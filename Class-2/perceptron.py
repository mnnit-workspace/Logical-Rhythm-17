import numpy as np
import matplotlib.pyplot as plt

# Make a prediction with weights
def predict(row, weights):
    activation = weights[0]
    for i in range(len(row)-1):
        activation += weights[i + 1] * row[i]
    return 1.0 if activation >= 0.0 else -1

# Estimate Perceptron weights using stochastic gradient descent
# n_epoch is number of times to iterate over dataset
# l_rate is the amount by which we will modify the weights in each iteration
def train_weights(train, l_rate, n_epoch):
    weights = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
        sum_error = 0.0
        for row in train:
            prediction = predict(row, weights)
            error = row[-1] - prediction
            sum_error += error**2 # Add error to sum
            weights[0] = weights[0] + l_rate * error
            for i in range(len(row)-1):
                weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
            # For plotting
            # Generating points for showing the line
            if weights[2] != 0:
                y  = []
                for i in dataset :
                    # Line equation in 2d ax + by + c = 0 or y = -(ax + c)/b
                    y.append(-(weights[0] + weights[1]*i[0])/weights[2])
                plt.cla()
                plt.hold(True)
                plt.scatter(data_array[:,0],data_array[:,1],marker="o",c=data_array[:,2])
                plt.plot(data_array[:,0], y)
                plt.hold(False)
                plt.pause(0.25)
        print('epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
    return weights

# Calculate weights
dataset = [[1, 38, 1], [2, -63, -1], [3, 84, 1], [4, -39, -1], [5, 73, 1], [6, -12, -1], [7, 68, 1], [8, -21, -1], [9, 57, 1], [10, 1, -1], [11, 78, 1], [12, -16, -1], [13, 101, 1], [14, -27, -1], [15, 79, 1], [16, -10, -1], [17, 86, 1], [18, -1, -1], [19, 98, 1], [20, 19, -1], [21, 109, 1], [22, 40, -1], [23, 135, 1], [24, 22, -1], [25, 147, 1], [26, 41, -1], [27, 130, 1], [28, 27, -1], [29, 130, 1], [30, 33, -1], [31, 164, 1], [32, 65, -1], [33, 155, 1], [34, 76, -1], [35, 155, 1], [36, 45, -1], [37, 165, 1], [38, 65, -1]]

data_array = np.array(dataset)
plt.ion()
plt.figure()
plt.scatter(data_array[:,0],data_array[:,1],marker="o",c=data_array[:,2])
l_rate = 0.01
n_epoch = 5
weights = train_weights(dataset, l_rate, n_epoch)
print("---------Results---------")
print("Final weights:", weights)

