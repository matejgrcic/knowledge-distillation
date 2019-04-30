import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np

class DeepModel():
    def __init__(self,inputs, num_classes):
        config = {
            'weight_decay': 1e-3,
            'conv1_size': 16,
            'mp1_size': 3,
            'conv2_size': 32,
            'mp2_size': 3,
            'fc3_size': 256,
            'fc4_size': 128,
            'out_size': 10
        }

        optimization_config = {
            'learning_rate': 0.01,
            'decay_steps': 10,
            'decay_rate': 0.96,
        }

        _, H, W, C = inputs.shape
        self.X = tf.placeholder(tf.float32, [None, H, W, C], name='X_placeholder')
        self.Y_oh = tf.placeholder(tf.float32, [None, num_classes], name='Y_placeholder')

        net = self._create_5x5_conv_layers(config)
        self.logits = self._create_fully_connected_layers(net, config)

        self.loss = tf.losses.softmax_cross_entropy(self.Y_oh, self.logits)
        self.optimization = self._create_optimization(optimization_config)
    
    def _create_5x5_conv_layers(self, config):
        net = None
        with tf.contrib.framework.arg_scope([layers.convolution2d],
            kernel_size=5, stride=1, padding='SAME', activation_fn=tf.nn.relu,
            weights_initializer=layers.variance_scaling_initializer(),
            weights_regularizer=layers.l2_regularizer(config['weight_decay'])):

            net = layers.convolution2d(self.X, config['conv1_size'], scope='te-conv1', data_format='NHWC')
            net = tf.layers.batch_normalization(net)
            net = layers.max_pool2d(net, config['mp1_size'], scope='te-mp1', data_format='NHWC')
            net = layers.convolution2d(net, config['conv2_size'], scope='te-conv2', data_format='NHWC')
            net = tf.layers.batch_normalization(net)
            net = layers.max_pool2d(net, config['mp2_size'], scope='te-mp2', data_format='NHWC')
        return net

    def _create_fully_connected_layers(self, net, config):
        with tf.contrib.framework.arg_scope([layers.fully_connected],
            activation_fn=tf.nn.relu,
            weights_initializer=layers.variance_scaling_initializer(),
            weights_regularizer=layers.l2_regularizer(config['weight_decay'])):
            net = layers.flatten(net)
            net = layers.fully_connected(net, config['fc3_size'], scope='te-fc3')
            net = tf.layers.batch_normalization(net)
            net = layers.fully_connected(net, config['fc4_size'], scope='te-fc4')
            net = tf.layers.batch_normalization(net)
            net = layers.fully_connected(net, config['out_size'], activation_fn=None, scope='te-logits')
        return net
    def _create_optimization(self, config):
        self.global_step = tf.placeholder(tf.int32, [])
        self.learning_rate = tf.train.exponential_decay(
            config['learning_rate'], self.global_step, config['decay_steps'], config['decay_rate'])
        optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        return optimizer.minimize(self.loss)

    def train(self, train_x, train_y, session, config):
        batch_size = config['batch_size']
        max_epochs = config['max_epochs']
        num_examples = train_x.shape[0]
        assert num_examples % batch_size == 0
        num_batches = num_examples // batch_size

        for epoch in range(1, max_epochs + 1):
            cnt_correct = 0
            permutation_idx = np.random.permutation(num_examples)
            train_x = train_x[permutation_idx]
            train_y = train_y[permutation_idx]
            for i in range(num_batches):

                batch_x = train_x[i*batch_size:(i+1)*batch_size, ...]
                batch_y = train_y[i*batch_size:(i+1)*batch_size, ...]

                logits_val, loss_val, lr_val = session.run(
                    [self.logits, self.loss, self.optimization, ],
                    feed_dict={self.X: batch_x, self.Y_oh: batch_y, self.global_step: epoch}
                )

                yp = np.argmax(logits_val, 1)
                yt = np.argmax(batch_y, 1)
                cnt_correct += (yp == yt).sum()

                if i % 5 == 0:
                    print("epoch %d, step %d/%d, batch loss = %.2f" % (epoch, i*batch_size, num_examples, loss_val))
                if i > 0 and i % 50 == 0:
                    print("Train accuracy = %.2f" % (cnt_correct / ((i+1)*batch_size) * 100))
            print("Train accuracy = %.2f" % (cnt_correct / num_examples * 100))
