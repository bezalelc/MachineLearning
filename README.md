# MachineLearning
A module of machine learning algorithms


#Code Structure

- layer.py - representation of different layers: 
    - WeightLayer - simple layer with weights 
    - ActLayer - layer with activation gate like ReLU, Tanh, etc...
    - Dropout
    - Conv - convolution layer for CNN.
    - MaxPooling
    - LstmRNN - lstm implementation of RNN layer.
    - VanillaRNN - simple implementation of RNN layer.
    - NormLayer - normalization layer with support in batch norm, layer norm and group norm.
- model.py - representation of different machine learning models:
  - NN - neural network
  - RNN - recurrent neural network
  - KNearestNeighbor
  - SVM
- optimizer.py - optimizers for the regression:
  - Vanilla
  - Momentum
  - NesterovMomentum
  - AdaGrad
  - RMSProp
  - Adam
- activation.py - activation function that support in gradient and loss for each type of activation:
  - Sigmoid
  - Linear
  - Softmax
  - Hinge
  - Tanh
  - ReLU
- norm.py - norm data function with support in gradient for each norm type:
  - Norm1 - sum(|X|)
  - Norm2 - (sum(X)**2)**0.5
  - Norm12 - Norm1+Norm2
- metrics.py - metrics for evaluate the performance of the model:
  - accuracy
  - null_accuracy
  - confusion_matrix
  - recall
  - precision
  - F_score
  - false_positive_rate
  - roc
  - auc
- norm_data.py - normalizers for prepare the data before the train:
  - StdNorm - move the distribution to normal distribution
  - ZeroCenter - move center of the data to zero
  - ZeroOneRange - normalize the range of the data to  [0,1] range
- data_generator.py - generators that help the train prosses to move on data with different sizes of batches:
  - DataGen
  - DataGenRNN
- weights_init.py - different method for init the weights:
  - zerosInit - init weights to zeroes
  - xavierScale
  - stdScale - W=random between [-epsilon,+epsilon]
  - heEtAlScale
- examples.py - examples how to use this MachineLearning lib 



