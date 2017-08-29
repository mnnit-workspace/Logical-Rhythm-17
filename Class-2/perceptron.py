import numpy as np
import matplotlib.pyplot as plt

# Make a prediction with weights
def predict(row, weights):
    activation = weights[0]
    for i in range(len(row)-1):
        activation += weights[i + 1] * row[i]
    return 1.0 if activation >=0.0 else -1

# Estimate Perceptron weights using stochastic gradient descent
# n_epoch is number of times to iterate over dataset
# l_rate is the amount by which we will modify the weights in each iteration
def train_weights(train, l_rate, n_epoch):
    weights = [1.0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
        sum_error = 0.0
        for row in train:
            prediction = predict(row, weights)
            error = row[-1] - prediction
            sum_error += error**2
            # Threshold updation
            weights[0] = weights[0] + l_rate * error
            for i in range(len(row)-1):
                weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
            # For plotting
            # Generating points for showing the line
            if weights[2]!=0:
                y = []
                for i in dataset :
                    y.append(-(weights[0] + weights[1]*i[0])/weights[2])
                plt.cla()
                plt.hold(True)
                plt.scatter(data_array[:,0],data_array[:,1],marker="o",c=data_array[:,2])
                plt.plot(data_array[:,0], y)
                plt.hold(False)
                plt.pause(0.3)
        print('epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
    return weights

# Calculate weights
dataset = [[1, 43, 1], [2, -43, -1], [3, 58, 1], [4, -37, -1], [5, 73, 1], [6, -34, -1], [7, 74, 1], [8, -30, -1], [9, 85, 1], [10, -15, -1], [11, 73, 1], [12, -11, -1], [13, 83, 1], [14, -14, -1], [15, 95, 1], [16, 5, -1], [17, 110, 1], [18, 1, -1], [19, 115, 1], [20, 16, -1], [21, 111, 1], [22, 25, -1], [23, 126, 1], [24, 23, -1], [25, 133, 1]]

data_array = np.array(dataset)
plt.ion()
plt.figure()
plt.scatter(data_array[:,0],data_array[:,1],marker="o",c=data_array[:,2])
l_rate = 0.01
n_epoch = 2
weights = train_weights(dataset, l_rate, n_epoch)

plt.pause(1000)

print("---------Results---------")
print("Final weights:", weights)

