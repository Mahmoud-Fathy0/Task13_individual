# Task13_individual

# **Neural Network from Scratch:**
## Implementation:
In this approach, we manually define all the components of the neural network, including back propagation, weight updates, and activation functions using Python and NumPy.


## Network Architecture:
- Input Layer: 784 nodes (28x28 pixels).
- Hidden Layer: Configurable number of nodes (e.g., 30).
- Output Layer: 10 nodes (digits 0-9).
- Activation Function: Sigmoid activation for hidden and output layers.
- Cost Function: Quadratic cost (mean squared error).
- Backpropagation: Implement backpropagation to calculate gradients and update weights.
  ```
  def backprop(self, x, y):
    """Return a tuple (nabla_b, nabla_w) representing the gradient."""
    # Feedforward
    activation = x
    activations = [x]
    zs = []
    for b, w in zip(self.biases, self.weights):
        z = np.dot(w, activation) + b
        zs.append(z)
        activation = sigmoid(z)
        activations.append(activation)
    # Backward pass
    # Compute the cost gradient and backpropagate the error
    ...
   ```
- Training: Train the network using stochastic gradient descent (SGD).
- Testing: Test the model on unseen MNIST images to compute the accuracy.
## **What I learned:**
-  Building from scratch offers complete control over the implementation and is an excellent learning exercise.
-  Manually implementing backpropagation and managing weight updates can be error-prone and tedious for more complex models.
  
  # **Neural Network using PyTorch:**
  
## Implementation:
In this approach, we use PyTorch, a powerful deep learning library, to build the same neural network architecture. PyTorch abstracts away much of the manual work involved in neural networks (e.g., backpropagation) and provides efficient GPU support.

Key Steps:
- Data Preparation: Use torchvision.datasets to directly load the MNIST dataset.
- Network Architecture: Define a feedforward neural network using torch.nn.Module.
- Loss Function: Use torch.nn.CrossEntropyLoss() which combines softmax and cross-entropy loss.
- Optimizer: Use stochastic gradient descent (SGD) or other optimizers (e.g., Adam).
- Training: Train the network using PyTorch’s built-in training loop.
- Testing: Test the model using PyTorch’s evaluation mode and compute accuracy.
  ```
  import torch
  import torch.nn as nn
  import torch.optim as optim
  import torchvision
  import torchvision.transforms as transforms

  class NeuralNet(nn.Module):
      def __init__(self):
          super(NeuralNet, self).__init__()
          self.fc1 = nn.Linear(28*28, 30)
          self.fc2 = nn.Linear(30, 10)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)  # Output layer (no activation since we'll use CrossEntropyLoss)
        return x

  # Initialize model, loss, and optimizer
  model = NeuralNet()
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=0.1)

  # Training Loop
  for epoch in range(num_epochs):
      for images, labels in train_loader:
          images = images.view(-1, 28*28)  # Flatten images
          optimizer.zero_grad()
          outputs = model(images)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()

  ```
## **What I learned**:
- PyTorch simplifies model building, training, and evaluation with high-level abstractions.
- Using a frmaework lets you load datasets easier.
- It would be easier and more effecient to use a framework for more complex network requirements.

## **Comparison: Scratch vs. Framework**

| Aspect                 | Neural Network from Scratch                          | Neural Network with PyTorch                |
|------------------------|------------------------------------------------------|--------------------------------------------|
| **Implementation**      | Requires manual coding of everything (forward/backward pass, gradient updates, etc.) | High-level API simplifies implementation  |
| **Learning Experience** | Excellent for understanding core concepts like backpropagation, activation functions, etc. | Easier to get started, but concepts are abstracted |
| **Performance**         | Slower, especially for large datasets and models     | Optimized for speed, especially with GPU support |
| **Ease of Use**         | More complex, requires careful debugging             | Much simpler and faster to implement       |
| **Flexibility**         | Full control over the network and learning process   | Flexible, but some control is abstracted   |
| **Error-Prone**         | Higher chance of errors when manually implementing everything | Less error-prone due to built-in features and optimizations |


