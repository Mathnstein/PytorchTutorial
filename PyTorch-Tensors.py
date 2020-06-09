'''
A pure pytorch implementation of a fully connected relu network using either cpu or gpu.
'''
import torch
import time

dtype = torch.float
# device = torch.device("cpu")
device = torch.device("cuda:0")

# N is batch size; D_in is input dimension; H is hidden dimension; D_out is output dimension
N, D_in, H, D_out = 64, 1000, 100, 10

steps = 501

# Create random input and output data
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# Randomly initialize weights
w1 = torch.randn(D_in, H, device=device, dtype=dtype)
w2 = torch.randn(H, D_out, device=device, dtype=dtype)

start = time.time()
learning_rate = 1e-6
for t in range(steps):
    h = x.mm(w1) # mm = matrix multiplication
    h_relu = h.clamp(min=0) # a linear function with a truncated min, optionally can truncate max
    y_pred = h_relu.mm(w2)

    loss = (y_pred - y).pow(2).sum().item() # get the L2 norm and then extract it with .item()
    if t % 100 == 0:
        print(t, loss)
    
    # Backprop to compute gradients of w1 and w2 w.r.t loss
    grad_y_pred = 2.0*(y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred) # .t() is the transpose of a 2D tensor
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    # Update weights using gradient descent
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
    
end = time.time()
# Final results
print(f'The best L2 loss is {loss}')
print(f'Ellapsed time for {device}: {end-start}s')

