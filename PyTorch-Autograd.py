'''
Using the autograd approach, we can have the partial derivatives computed by an automatic chainrule, thus saving us memory allocation for interim vars.
'''

import torch
import time

dtype = torch.float
# device = torch.device("cpu")
device = torch.device("cuda:0") # Uncomment this to run on GPU

# N is batch size; D_in is input dimension; H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

steps = 501

# Create random Tensors to hold input and outputs.
# Setting requires_grad=False indicates that we do not need to compute gradients
# with respect to these Tensors during the backward pass.
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# Create random Tensors for weights.
# Setting requires_grad=True indicates that we want to compute gradients with
# respect to these Tensors during the backward pass.
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

start = time.time()
learning_rate = 1e-6
for t in range(steps):
    # Forward pass: compute predicted y using operations on Tensors; these are the same as before but we just dont need interim values
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # Compute the loss which will be a (1,) tensor
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 0:
        print(t, loss.item())

    # Use autograd to compute the backward pass.
    # This call will compute the gradient of loss with respect to all Tensors with requires_grad=True
    # After this call we can get the gradients with X.grad for each differentiable X
    loss.backward()

    # Manually update weights using gradient descent. Must use torch.no_grad() since the weights had requires_grad=True
    # but we don't need to track these operations with autograd
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # Manually zero the gradients back out
        w1.grad.zero_()
        w2.grad.zero_()

end = time.time()
# Final results
print(f'The best L2 loss is {loss}')
print(f'Ellapsed time for {device}: {end-start}s')