# PytorchTutorial
(Recall that pipenv is used and all dependencies now sit in the pipfile.lock)

This is just a collection of the demos found on [PyTorch](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html).

## Demos
All demos do the same thing, they vary on what features they use, we start from scratch and manually do everything until we have automated the entire
process. We are randomly generating observed data **y**, we have random feature data **x** and we fit a fully connected ReLU network with structure
<img src="https://latex.codecogs.com/gif.latex?X_{N,D_{in}}\rightarrow H \rightarrow Y_{N,D_{out}} " /> 
- Pure Numpy 
- PyTorch Tensors
- PyTorch AutoGradients
- PyTorch Custom AutoGradients
- PyTorch Neural Network (*nn*) package
- PyTorch Optimizer (*optim*) package
- PyTorch Custom Neural Network modules
- PyTorch Dynamic Neural Networks