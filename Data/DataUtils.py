import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def non_linear_case_1(input_data, output_dim, noise):
    # f(x) = 1 / (1 + exp(-x)) + noise
    # Ensure input_data has the correct shape
    input_dim = input_data.shape[-1]
    assert input_dim == output_dim, "Input data must be the same as output_dim."
    # Apply sigmoid function to the last output_dim dimensions
    transformed_data = sigmoid(input_data)
    transformed_data += noise
    
    return transformed_data

def non_linear_case_2(input_data, output_dim, noise_std=0.1):
    """
    Transform input data through a weighted sum (mimicking a simple neural layer) followed by a ReLU activation,
    and add isotropic Gaussian noise to the output. This function changes the dimensionality from input_dim to output_dim.

    Parameters:
    - input_data: Input array of shape (N, input_dim) where N is the number of samples.
    - input_dim: The dimensionality of the input data.
    - output_dim: The desired dimensionality of the output data.
    - noise_std: Standard deviation of the isotropic Gaussian noise to be added to the output.

    Returns:
    - transformed_data: Output array of shape (N, output_dim) with the transformed data.
    """
    # Randomly initialize weights for dimensionality change
    input_dim = input_data.shape[-1]    
    weights = np.random.randn(input_dim, output_dim)
    
    # Apply transformation: Weighted sum followed by ReLU activation
    transformed_data = relu(np.dot(input_data, weights))
    
    # Add isotropic Gaussian noise
    noise = np.random.normal(0, noise_std, transformed_data.shape)
    transformed_data += noise
    
    return transformed_data