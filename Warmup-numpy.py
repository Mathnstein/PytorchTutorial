'''
A pure numpy fully connected relu network to take random input data and ouput data and fit it with L2 loss.
'''

import numpy as np
import time

# N is batch size; D_in is input dimension; H is hidden dimension; D_out is output dimension
N, D_in, H, D_out = 64, 1000, 100, 10

steps = 501

# Create random input and output data
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# Randomly initialize
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

start = time.time()
learning_rate = 1e-6
for t in range(steps):
    # Forward pass: compute predicted y
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    # Compute and print loss
    loss = np.square(y_pred - y).sum()
    if t % 100 == 0:
        print(t, loss)

    # Backprop to compute gradients of w1 and w2 w.r.t loss
    grad_y_pred = 2.0 * (y_pred-y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    # Update weights
    w1 -= learning_rate*grad_w1
    w2 -= learning_rate*grad_w2

end = time.time()
# Final results
print(f'The best L2 loss is {loss}')
print(f'Ellapsed time: {end-start}s')
