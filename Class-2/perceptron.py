import numpy as np
import matplotlib.pyplot as plt

# Make a prediction with weights
def predict(row, weights):
    activation = weights[0]
    for i in range(len(row)-1):
        activation += weights[i + 1] * row[i]
    return 1.0 if activation >=0.0 else 0.0

# Estimate Perceptron weights using stochastic gradient descent
# n_epoch is number of times to iterate over dataset
# l_rate is the amount by which we will modify the weights in each iteration
def train_weights(train, l_rate, n_epoch):
    weights = [1.0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
        sum_error = 0.0
        for row in train:
            #print (row)
            plt.cla()
            prediction = predict(row, weights)
            error = row[-1] - prediction
            sum_error += error**2
            weights[0] = weights[0] + l_rate * error
            for i in range(len(row)-1):
                weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
            # For plotting
            # Generating points for showing the line
            if weights[2]!=0:
	            y = []
	            for i in dataset :
	                y.append(-(weights[0] + weights[1]*i[0])/weights[2])
	            plt.hold(True)
	            plt.scatter(data_array[:,0],data_array[:,1],marker="o",c=data_array[:,2])
	            plt.plot(data_array[:,0], y)
	            plt.hold(False)
	            plt.pause(0.5)
        print('epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
    return weights

# Calculate weights
dataset = [[1, 43, 1], [2, -43, 0], [3, 58, 1], [4, -37, 0], [5, 73, 1], [6, -34, 0], [7, 74, 1], [8, -30, 0], [9, 85, 1], [10, -15, 0], [11, 73, 1], [12, -11, 0], [13, 83, 1], [14, -14, 0], [15, 95, 1], [16, 5, 0], [17, 110, 1], [18, 1, 0], [19, 115, 1], [20, 16, 0], [21, 111, 1], [22, 25, 0], [23, 126, 1], [24, 23, 0], [25, 133, 1]]

data_array = np.array(dataset)
plt.ion()
plt.figure()
plt.scatter(data_array[:,0],data_array[:,1],marker="o",c=data_array[:,2])
l_rate = 0.01
n_epoch = 2
weights = train_weights(dataset, l_rate, n_epoch)

plt.pause(10)

print("---------Results---------")
print("Final weights:", weights)

