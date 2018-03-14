
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import convenience_utils as utils
import tiff_utils


# In[2]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# In[6]:


class VariationalAutoEncoder(object):

    def __init__(self, encoder_arch, decoder_arch,
                 learning_rate=1e-4, nonlinearity=tf.nn.relu):

        self.encoder = None
        self.decoder = None

        self.learning_rate = learning_rate
        self.nonlinearity = nonlinearity

        self._construct_encoder(encoder_arch)
        self._construct_decoder(decoder_arch)

        self.loss_function = self._get_loss_function(self.input, self.output)
        self.optimizer = self._get_optimizer(learning_rate, self.loss_function)

        self._create_internal_session()
        self._create_internal_saver()

    
    def train(self, batch_size, data_stream, epochs, to_save_path=None, verbose=False, every=1000):
        '''
        The train method allows you to train your autoencoder on different datasets fairly hands-free
        It does demand as input a data_stream object, which implements a method called next_batch()
        that returns a tensor of dimension [input_size, batch_size], or [x, y, pixel_depth, batch_size] for the
        convolutional case.
        '''

        self.internal_session.run(tf.global_variables_initializer())
        test_data = data_stream.test_batch(1)

        for i in range(epochs):
            image_batch = data_stream.next_batch(batch_size=batch_size)
#             image_batch = image_batch - np.mean(image_batch, 0)
#             image_batch = image_batch / np.var(image_batch, 0)
#             image_batch = image_batch / np.max(image_batch) * 10

            _, cost, result = self.internal_session.run(
                [self.optimizer, self.loss_function, self.output],
                feed_dict={self.input: image_batch}
            )

            if verbose and (i % every == 0):
                print 'Round: %d' % (i + 1), 'cost = %f' % cost

            if to_save_path is not None and (i % every == 0):
                self._save_model(to_save_path)


    def predict(self, new_data):
        return self.internal_session.run(self.output,
                feed_dict={self.input: new_data})
    
    
    def predict_from_embedding(self, new_data_mean, new_data_stddev):
        return self.internal_session.run(self.output,
                                        feed_dict={self.encoder: (new_data_mean, new_data_stddev)})
    

    def get_embedding(self, input_batch):
        return self.internal_session.run(self.encoder, feed_dict={self.input: input_batch})


    def _save_model(self, to_save_path):
        self.saved_path = self.saver.save(self.internal_session, to_save_path)
        print 'Model saved at location: ', self.saved_path
        
    
    def _restore_model(self, saved_path):
        self.saver.restore(self.internal_session, saved_path)


    def _get_weights_and_biases(self, widths):
        weights = []
        biases = []

        initializer = tf.contrib.layers.xavier_initializer()
        for layer in range(len(widths)-1):
            weights.append(tf.Variable(initializer((widths[layer], widths[layer+1]))))
            biases.append(tf.Variable(tf.constant(0.1, shape=(widths[layer+1], ))))

        return weights, biases


    def _get_network(self, weights, biases, inp):

        h = inp
        for layer in range(len(weights)-1):
            h = self.nonlinearity(tf.add(tf.matmul(h, weights[layer]), biases[layer]))

        return tf.add(tf.matmul(h, weights[-1]), biases[-1])


    def _construct_encoder(self, encoder_architecture):
        '''
        Construct an encoder architecture from the list of sizes given in
        the encoder_architecture list.
        '''

        self.input = tf.placeholder(tf.float32, shape=[None, encoder_architecture[0]])
        self.encoder_weights, self.encoder_biases = self._get_weights_and_biases(encoder_architecture[:-1])

        # should we put activation at this layer
        network = self._get_network(self.encoder_weights,
                                         self.encoder_biases,
                                         self.input)

        w1, b1 = self._get_weights_and_biases([encoder_architecture[-2], encoder_architecture[-1]])
        self.mean = tf.add(tf.matmul(network, w1[0]), b1[0])
        
        # how to account for the non-negative sigma?
        w2, b2 = self._get_weights_and_biases([encoder_architecture[-2], encoder_architecture[-1]])
        self.stddev = tf.add(tf.matmul(network, w2[0]), b2[0])
        
        self.encoder = (self.mean, self.stddev)


    def _construct_decoder(self, decoder_architecture):
        '''
        Construct a decoder architecture from the list of sizes given in
        the decoder_architecture list.
        '''

        if self.encoder is None:
            raise Exception('Encoder architecture not defined')
            
#         samples_ = tf.random_normal([self.batchsize, decoder_architecture[0]], 0, 1, dtype=tf.float32)
        samples_ = tf.random_normal(tf.shape(self.mean), 0, 1, dtype=tf.float32)
        z = self.mean + (self.stddev * samples_)
        
        self.decoder_weights, self.decoder_biases = self._get_weights_and_biases(decoder_architecture)

        self.decoder = self._get_network(self.decoder_weights,
                                         self.decoder_biases,
                                         z)

        self.output = self.decoder
        
        
    def _kl_divergence(self):
        return tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(self.mean) + tf.square(self.stddev) - tf.log(tf.square(self.stddev)) - 1, 1))


    def _get_loss_function(self, inp, output):
        return tf.losses.mean_squared_error(inp, output) + self._kl_divergence()


    def _get_optimizer(self, learning_rate, loss_function):
        return tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_function)


    def _create_internal_session(self):
        self.internal_session = tf.Session()

        
    def _create_internal_saver(self):
        self.saver = tf.train.Saver()


# In[4]:


enc_architecture = [28*28, int(28*28/4), int(28*28/16)]
dec_architecture = [int(28*28/16), int(28*28/4), 28*28]
transfer_fct = tf.nn.relu
learning_rate = 0.00001
batch_size = 32

data_stream = utils.MNIST()
vae = VariationalAutoEncoder(encoder_arch=enc_architecture, decoder_arch=dec_architecture, learning_rate=learning_rate, nonlinearity=transfer_fct)
vae.train(data_stream=data_stream, batch_size=batch_size, epochs=1000000, verbose=True, to_save_path='./model.ckpt')


# In[5]:


rows, cols = 10, 10
_, axes1 = plt.subplots(rows, cols, figsize=(5, 5))
_, axes2 = plt.subplots(rows, cols, figsize=(5, 5))

test_sample = data_stream.test_batch(rows*cols)
prediction = vae.predict(test_sample)

for i in range(rows):
    for j in range(cols):
        axes1[i][j].set_axis_off()
        axes1[i][j].imshow(np.reshape(test_sample[cols*i+j, :], [28, 28]), cmap='gray')

for i in range(rows):
    for j in range(cols):
        axes2[i][j].set_axis_off()
        axes2[i][j].imshow(np.reshape(prediction[cols*i+j,:], [28, 28]), cmap='gray')

plt.show()

