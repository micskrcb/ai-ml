# ai-ml

the framework contains 

1. **Linear Layer Class**:
   - Implements a fully connected layer.
   - Methods: `forward`, `backward`.

2. **ReLU Activation Class**:
   - Implements the ReLU activation function.
   - Methods: `forward`, `backward`.

3. **Sigmoid Activation Class**:
   - Implements the Sigmoid activation function.
   - Methods: `forward`, `backward`.

4. **Tanh Activation Class**:
   - Implements the Tanh activation function.
   - Methods: `forward`, `backward`.

5. **Softmax Activation Class**:
   - Implements the Softmax activation function.
   - Methods: `forward`, `backward`.

6. **Cross-Entropy Loss Class**:
   - Implements the cross-entropy loss function. You can use the fusion method described in the PDF as well. See how `nn.CrossEntropyLoss` in PyTorch works.
   - Methods: `forward`, `backward`.

7. **Mean Squared Error (MSE) Loss Class**:
   - Implements the MSE loss function.
   - Methods: `forward`, `backward`.

8. **SGD Optimizer Class**:
   - Implements the stochastic gradient descent optimizer.
   - Methods: `step`.

9. **Model Class**:
   - Wraps everything into a cohesive model.
   - Methods: `add_layer`, `compile`, `train`, `predict`, `evaluate`, `save`, and `load`.
