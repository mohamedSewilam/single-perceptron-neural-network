import numpy as np

# Define a single perceptron
class Perceptron():

    def __init__(self, n_features, weights, bias):
        self.weights = np.array(weights)
        self.bias = bias
        self.n_features = n_features

    def predict(self, inputs):
        # Calculate the dot product of the inputs and weights
        dot_product = np.dot(inputs, self.weights)

        # Add the bias
        dot_product += self.bias

        # Apply the activation function ( a step function)
        if dot_product >= 0:
            return 1
        else:
            return 0

# Define the input values
n_features = int(input("enter number of features : "))

weights = []
for i in range(n_features):
    weight = float(input(f"Enter weight {i+1}: "))
    weights.append(weight) 
    
bias = float(input("enter bias value : "))

# Take user input for patterns and labels
patterns = []
n_patterns = int(input("Enter the number of patterns: "))
for i in range(n_patterns):
    pattern = []
    for j in range(n_features):
        feature = float(input(f"Enter feature {j+1} of pattern {i+1}: "))
        pattern.append(feature)
    label = int(input("Enter label for pattern {i+1} (0 or 1): "))
    patterns.append((pattern, label))
    
"""
patterns = [
    ([0, 0], 0),
    ([0, 1], 0),
    ([1, 0], 0),
    ([1, 1], 1)
]
"""
# Create a single perceptron with the input values
perceptron = Perceptron(n_features, weights, bias)

# Use the perceptron to predict the labels for each pattern and compare with the true labels
for pattern in patterns:
    inputs, true_label = pattern
    predicted_label = perceptron.predict(inputs)
    if predicted_label == true_label:
        print(f"Input pattern {inputs} is correctly classified.")
    else:
        print(f"Input pattern {inputs} is misclassified.")

