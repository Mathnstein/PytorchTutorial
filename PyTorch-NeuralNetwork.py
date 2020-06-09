'''
The neural network framwork is used here to do most of the work for us. Notice that the learning rate is higher here because we actually have
two new parameters that are being optimized (w1,w2) were the former weights between input->hidden and hidden->output. The new params
are the bias inside of a linear term xA.T + b, since we have two linear terms then there should be two 1D parm sets mapping the bias.
'''
import torch
import time

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

steps = 501

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Use the nn package to define our model as a sequence of layers. nn.sequential is a module which contains other modules,
# and applies them in sequence to produce its output. Each Linear module computes output from input using a linear function, 
# and holds internal Tensors for its weight and bias.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H,D_out),
)

# The nn package also contains definitions of popular loss functions; in this case we will use the MSE with reduction by sum, which is equiv to L2
loss_fn = torch.nn.MSELoss(reduction='sum')

start = time.time()
learning_rate = 1e-4
for t in range(steps):
    # Forward pass: compute predicted y by passing x to the model. Module objects override the __call__ operator so you can call them like functions.
    # When doing so you pass a Tensor of input data to the module and it produces a Tensor of output data.
    y_pred = model(x)

    # Compute loss.
    loss = loss_fn(y_pred, y)
    if t % 100 == 0:
        print(t, loss.item())

    # Zero the gradients before running the backward pass
    model.zero_grad()

    # Backward pass: compute the gradient of the loss w.r.t all the learnable parameters of the model. Internally, the parameters of each module
    # are stored in Tensors with requires_grad=True, so this call will compute gradients for all learnable parameters.
    loss.backward()

    # Update the weights using gradient descent. Each parameter is a Tensor, so we can access the gradients as before.
    with torch.no_grad():
        if t == 0:
            # Here I determine how many parameters we are calculating derivatives for
            params = list(model.parameters())
            print(len(params))
        for param in model.parameters():
            param -= learning_rate * param.grad
            if t == steps-1:
                # Here is how i determine the params, w1 should be size [100,1000], w2 is size [10, 100] and so the 1D parms must be the bias
                print(param.size())

end = time.time()
# Final results
print(f'The best L2 loss is {loss}')
print(f'Ellapsed time for cpu: {end-start}s')