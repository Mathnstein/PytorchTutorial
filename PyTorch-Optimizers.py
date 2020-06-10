'''
Adam is a great state of the art optimizer that uses 2nd moment estimates to do momentum-like updates to parameters instead of basic grad desc.
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

# Use the nn package to define our model and loss function.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)
loss_fn = torch.nn.MSELoss(reduction='sum')

# Use the optim package to define an optimizer that will update weights in a model for us. Here we use Adam.
start = time.time()
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(steps):
    # Forward pass: compute predicted y by passing x into the model
    y_pred = model(x)

    # Compute loss
    loss = loss_fn(y_pred, y)
    if t % 100 == 0:
        print(t, loss.item())

    # Before the backward pass, use the optimizer object to zero all gradients that are marked as learnable.
    # This is because by default, gradients are accumulated in buffers (i.e not overwritten) whenever .backward() is called.
    optimizer.zero_grad()

    # Backward pass
    loss.backward()

    # Calling the step function on an optimizer makes the update in the parameter space
    optimizer.step()

end = time.time()
# Final results
print(f'The best L2 loss is {loss}')
print(f'Ellapsed time for cpu: {end-start}s')