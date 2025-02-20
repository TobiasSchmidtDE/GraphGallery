from tensorflow.keras import activations, constraints, initializers, regularizers
from tensorflow.keras.layers import Layer

import tensorflow as tf


class DAGNNConv(Layer):
    """
        Basic graph convolution layer as in: 
        [Towards Deeper Graph Neural Networks](https://arxiv.org/abs/2007.09296)
        Pytorch implementation: <https://github.com/mengliu1998/DeeperGNN>

        `DAGNNConv` implements the operation:

        `propagations = Stack(\sum_k^K adj^k @ x)
        output = activation(propagations @ kernel + bias).transpose([0, 2, 1])
        output = (output @ propagationsoutput).squeeze()`

        where `x` is the node attribute matrix, `adj` is the adjacency matrix,
        K is the propagation steps of adjacency matrix.
        `activation` is the element-wise activation function
        passed as the `activation` argument, `kernel` is a weights matrix
        created by the layer, and `bias` is a bias vector created by the layer
        (only applicable if `use_bias` is `True`).


        Parameters:
          K: Propagation steps of adjacency matrix.
          activation: Activation function to use.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
          use_bias: bool, whether the layer uses a bias vector.
          kernel_initializer: Initializer for the `kernel` weights matrix.
          bias_initializer: Initializer for the bias vector.
          kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
          bias_regularizer: Regularizer function applied to the bias vector.
          activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation")..
          kernel_constraint: Constraint function applied to
            the `kernel` weights matrix.
          bias_constraint: Constraint function applied to the bias vector.

        Input shape:
          tuple/list with two 2-D tensor: Tensor `x` and SparseTensor `adj`: `[(num_nodes, num_node_attrs), (num_nodes, num_nodes)]`.
          The former one is the node attribute matrix (Tensor) and the last is adjacency matrix (SparseTensor).

        Output shape:
          2-D tensor with the same shape as `x`: `(num_nodes, num_node_attrs)`.       
    """

    def __init__(self,
                 K=10,
                 use_bias=False,
                 activation=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        super().__init__(**kwargs)
        self.units = 1  # units must be 1
        self.K = K
        self.use_bias = use_bias

        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shapes):
        self.kernel = self.add_weight(shape=(input_shapes[0][-1], self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        super().build(input_shapes)

    def call(self, inputs):

        x, adj = inputs

        propagations = [x]
        for _ in range(self.K):
            x = tf.sparse.sparse_dense_matmul(adj, x)
            propagations.append(x)

        h = tf.stack(propagations, axis=1)

        retrain_score = h @ self.kernel

        if self.use_bias:
            retrain_score += self.bias

        retrain_score = self.activation(retrain_score)
        retrain_score = tf.transpose(retrain_score, [0, 2, 1])

        output = tf.matmul(retrain_score, h)
        output = tf.squeeze(output, axis=1)

        return output

    def get_config(self):
        config = {'units': self.units,
                  'K': self.K,
                  'use_bias': self.use_bias,
                  'activation': activations.serialize(self.activation),
                  'kernel_initializer': initializers.serialize(
                      self.kernel_initializer),
                  'bias_initializer': initializers.serialize(
                      self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(
                      self.kernel_regularizer),
                  'bias_regularizer': regularizers.serialize(
                      self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(
                      self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(
                      self.kernel_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint)
                  }

        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shapes):
        attributes_shape = input_shapes[0]
        output_shape = (attributes_shape[0], self.units)
        return tf.TensorShape(output_shape)  # (num_nodes, output_dim)
