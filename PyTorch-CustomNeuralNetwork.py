'''
We may also define our own module as long as we provide how the forward direction works, we may also specify the backward direction if we want to ignore params
'''
import torch
import time

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

steps = 501

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Construct our model by instantiating the class defined above
model = TwoLayerNet(D_in, H, D_out)

# Construct our loss function and an optimizer. The call to model.parameters() in the SGD constructor will contain the learnable params
# of the above model we defined, which consists of two nn.Linear modules.
loss_fn = torch.nn.MSELoss(reduction='sum')

start = time.time()
learning_rate = 1e-4
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # SDG Stochastic Gradient descent
for t in range(steps):
    # Forward pass: Pass the data through the model
    y_pred = model(x)

    # Compute loss
    loss = loss_fn(y_pred, y)
    if t % 100 == 0:
        print(t, loss.item())

    # Zero grads, backward pass and update weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

end = time.time()
# Final results
print(f'The best L2 loss is {loss}')
print(f'Ellapsed time for cpu: {end-start}s')

    