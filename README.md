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
  
- 
