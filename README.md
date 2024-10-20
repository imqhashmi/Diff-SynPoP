# A differentiable approach for modelling multi-layer Synthetic Populations

### Table of Contents
- [Summary](#summary)
- [Main Benefits](#main-benefits)
- [Future Improvements](#future-improvements)
- [Installation](#installation)
- [Contributing](#contributing)
- [License](#license)

### Summary
This codebase provides a differentiable approach for generating synthetic demographic data, employing a combination of neural networks, tensor operations, and data aggregation techniques.

### Differentiable Approach:
1. **Gumbel-Softmax for Discrete Data Generation**: Typically, generating discrete data (like categorical demographic characteristics) is challenging in gradient-based optimization because the process is non-differentiable. The Gumbel-Softmax function is a clever use to approximate categorical distributions in a way that is continuous and differentiable, allowing standard backpropagation techniques to be used for training neural networks. This method bridges the gap between the need for discrete outputs and the requirement for differentiable operations in neural network training.
The Gumbel-Softmax technique provides a way to sample from a categorical distribution in a manner that is differentiable and therefore compatible with gradient-based optimization methods. It's particularly useful when dealing with attributes like 'sex' in neural network models where you need a differentiable approximation of discrete variables. Let's break down how this process works using the 'sex' attribute as an example:

![img.png](img.png)

This Gumbel-Softmax approach allows the 'sex' attribute to be encoded in a way that's compatible with the requirements of neural network training and optimization, particularly for handling discrete data categories.

### 1. **System Design**
   - **PyTorch Libraries**: For neural network operations and tensor manipulation.
   - **Custom Modules (`InputData`, `InputCrossTables`)**: For UK Census data handling and cross-table data structures.
   - **Plotly and Pandas**: For data visualization and manipulation.
   - **OS and Time Modules**: For directory operations and time tracking.

### 2. **Neural Network Definition (FFNetwork)**
   - A custom class defining a feedforward neural network.
   - **Initialization**: Creates layers dynamically based on input dimensions and hidden layer configurations.
   - **Forward Method**: Defines how data passes through the network.

   - Multiple instances of the `FFNetwork` for different demographics.
   - **Input Dimension**: Calculated as the sum of categories across demographic characteristics.
   - **Hidden Layers**: Two hidden layers with specified neuron counts.
   - Networks are moved to GPU if available.

### 3. **Gumbel-Softmax Function**
   - A function to sample from logits using the Gumbel-Softmax distribution, useful for generating discrete data.

### 4. **Population Generation Function**
   - Generates synthetic data by processing an input tensor through each neural network.
   - Applies the Gumbel-Softmax function to each set of logits to obtain a probability distribution.

### 5. **Data Encoding and Aggregation**
   - **`aggregate` Function**: Aggregates encoded tensors based on cross-tables.
   - **`decode_tensor` Function**: Decodes the tensor into human-readable categories.

### 6. **Tensor Category Selection**
   - The `keep_categories` function selects specific categories from the encoded tensor, used in the training loop.

### 7. **Visualization with Plotly**
   - The `plot` function creates bar plots to compare target and computed data.

### 8. **Accuracy and Loss Calculation**
   - **RMSE Accuracy**: Calculates Root Mean Square Error (RMSE) accuracy.
   - **RMSE Loss**: Loss function used during training.

### 9. **Training Loop**
   - **Optimizer Setup**: Uses Adam optimizer for training the neural networks.
   - **Epochs and Loss Computation**: Iterative training process.
   - **Data Aggregation**: Aggregates data for different demographic combinations using cross-tables.
   - **Backpropagation**: Adjusts network weights based on the computed loss.

### 10. **Final Data Processing and Output**
   - Decodes the tensor into readable records.
   - Converts records into a DataFrame and saves as CSV.

### 11. **Execution Time Tracking**
   - Records and prints the total duration of the script execution.

### Main Benefits:
1. **Enables Gradient-Based Optimization**: The differentiability of the entire process allows the use of gradient descent and backpropagation, which are fundamental techniques in training deep neural networks. This leads to more efficient and effective learning.

2. **Facilitates Discrete Data Modeling**: By employing Gumbel-Softmax, the script can model discrete data categories (like age groups or ethnicities) while still leveraging the power of neural networks. This is particularly important in demographic studies where data is inherently categorical.

3. **Improves Model Accuracy and Flexibility**: Differentiable operations enable the networks to adjust their weights in a fine-grained manner, potentially leading to higher accuracy in the synthetic data generated. It also provides flexibility in model design and optimization, allowing for various architectures and loss functions to be employed effectively.

4. **Scalability and Adaptability**: The modular and differentiable nature of the architecture makes it easier to scale the model to accommodate more demographic categories or to adapt it to different datasets and requirements.


### Future Improvements:
1. **Expansion of Demographic Characteristics**: Including more varied demographic data for a comprehensive view.
2. **Advanced Neural Architectures**: Exploring state-of-the-art neural network designs could improve data quality.
3. **Improved Sampling Methods**: Enhancing the data sampling process for more accurate or diverse results.
4. **Adaptation to Dynamic Cross-Tables**: Making the script more flexible to changes in cross-table structures.
5. **Complex Loss Functions and Regularization**: To capture demographic nuances more effectively.
6. **Scalability and Performance**: Ensuring efficiency for larger datasets.
7. **Ethical and Bias Considerations**: Incorporating measures to mitigate biases and ensure ethical use.
8. **Interactive Visualization**: Integrating tools for better data analysis and interpretation.
9. **User-Friendly Interface**: Developing a GUI for broader user accessibility.
10. **Real-World Testing and Validation**: Extensive validation with real-world data to ensure reliability.

In summary, this script is a significant step forward in the realm of synthetic demographic data generation, offering a nuanced and multifaceted approach to modeling complex data distributions. However, there's potential for further enhancements, particularly in terms of complexity, user accessibility, ethical considerations, and scalability, to increase its utility and applicability in diverse real-world scenarios.

### Installation
To use this project, clone the repository and install the necessary dependencies. The list of libraries needed to run this project have been extracted along with their respective versions into the requirements.txt file.

```bash
git clone https://github.com/imqhashmi/Diff-SynPoP.git
```
```bash
cd Diff-SynPoP
```
```bash
pip install -r requirements.txt
```
```bash
python run_model.py
```

### Contributing
We welcome contributions to this project and appreciate your efforts to improve it. To keep everything organized and to ensure quality, please follow these guidelines when contributing:

- Fork the repository to your own GitHub account. This will create a copy of the repository in your account.

- Clone the forked repository to your local machine.

```bash
git clone https://github.com/imqhashmi/Diff-SynPoP.git
```
```bash
cd Diff-SynPoP
```

- Create a new branch for your feature or bug fix. Use a descriptive name for your branch.

```bash
git checkout -b feature/your-feature-name
```

- Make the necessary changes to the codebase. Ensure your code follows the existing code style and conventions.

- If applicable, write tests for your changes to ensure they work as expected and do not break existing functionality.

- Commit your changes with a clear and concise commit message.

```bash
git add .
```
```bash
git commit -m "Add feature: Description of your feature"
```

- Push your changes to the forked repository.

```bash
git push origin feature/your-feature-name
```

- Lastly, create a pull request (PR) from your branch to the main branch of the original repository. Include a detailed description of your changes in the PR.

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
