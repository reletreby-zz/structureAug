'''
Based off of example autoencoder from https://github.com/aymericdamien/TensorFlow-Examples/
Reference:
	Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
	learning applied to document recognition." Proceedings of the IEEE,
	86(11):2278-2324, November 1998.
'''

#import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
tf.merge_all_summaries = tf.summary.merge_all
tf.train.SummaryWriter = tf.summary.FileWriter
import os, sys
from sklearn.decomposition import PCA
# from sklearn.decomposition import SparsePCA
from sklearn.preprocessing import StandardScaler as Scaler

display_step = 200

class PCAm():
	# Note: n_components can be int, None, or 'mle'
	def __init__(self, n_components=10, center_data=True):
		self.n_components = n_components
		self.center_data = center_data
		self.pca = PCA(n_components=self.n_components)
		if self.center_data:
			self.scaler = Scaler()
		else:
			self.scaler = None

	# Fit scaler and PCA to data:
	def fit_and_decompose(self, X):
		if self.center_data:
			X_scaled = self.scaler.fit_transform(X)
		else:
			X_scaled = X
		return self.pca.fit_transform(X_scaled)


	def decompose(self, X):
		if self.center_data:
			X_scaled = self.scaler.transform(X)
		else:
			X_scaled = X
		self.pca.transform(X_scaled)

	def recompose(self, X_decomp):
		X_recomp = self.pca.inverse_transform(X_decomp)
		if self.center_data:
			return self.scaler.inverse_transform(X_recomp)
		else:
			return X_recomp

	def unscale(self, X):
		if self.center_data:
			return self.scaler.inverse_transform(X)
		else:
			return X


# Autoencoder class:
# hidden layers is a list with the size of each hidden layer (exluding
# input layer) for the encoder half of the AE. Decoder layers are sized
# identically, but ordered in reverse.
# Tying weights means that the weights of the decoder stage are
# the transpose of the weights of the encoder stage.
# Options for activation are: 'sigmoid', 'tanh', 'relu', 'elu'
class AutoEncoder():
	def __init__(self, \
		input_dim, hidden_layers, \
		model_name="AutoEncoderV3", \
		use_disk=False, \
		tie_weights=True, \
		activation='sigmoid',\
		status=False,
		num_steps=5000,
		learning_rate=0.001,
		batch_size=100 ):

		self.num_input = input_dim
		self.num_hidden = len(hidden_layers)
		self.layer_sizes = [input_dim] + hidden_layers
		self.num_steps = num_steps
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		# print(self.layer_sizes)

		# Arrays that will hold TF vars for each layer:
		self.tf_enc_weights = []
		self.tf_dec_weights = []
		self.tf_enc_biases = []
		self.tf_dec_biases = []

		# placeholder for input data:
		self.X_in = tf.placeholder("float", [None, self.num_input])

		# Create TF weights and biases:
		L = len(self.layer_sizes)
		for i in range(1,L):
			# Add encoder weight vars:
			self.tf_enc_weights.append( tf.Variable(tf.random_normal(\
											[self.layer_sizes[i-1], \
											self.layer_sizes[i]])))
			# Encoder bias vars:
			self.tf_enc_biases.append( tf.Variable(tf.random_normal(\
											[self.layer_sizes[i]])))

		for i in range(len(self.tf_enc_weights)-1, -1, -1):
			# Decoder weight vars:
			if tie_weights:
				self.tf_dec_weights.append( tf.transpose(\
											self.tf_enc_weights[i]) )
			else:
				self.tf_dec_weights.append( tf.Variable(tf.random_normal(\
											[self.layer_sizes[i+1], \
											self.layer_sizes[i]])))
			# Decoder bias vars:
			self.tf_dec_biases.append( tf.Variable(tf.random_normal(\
											[self.layer_sizes[i]])))

		# print([s.shape for s in self.tf_enc_weights])
		# print([s.shape for s in self.tf_dec_weights])

		self.enc_layers = []
		self.dec_layers = []
		if activation == 'sigmoid':
			self.activation = tf.nn.sigmoid
		elif activation == 'tanh':
			self.activation = tf.nn.tanh
		elif activation == 'relu':
			self.activation = tf.nn.relu
		elif activation == 'elu':
			self.activation = tf.nn.elu
		else:
			print("ERROR: unknown activation: {}".format(activation))
			sys.exit(1)

		# Build the encoder from each layer of weights and biases:
		def encoder(x):
			# Build L-1 layers of encoder:
			for l in range(L-1):
				w = self.tf_enc_weights[l]
				b = self.tf_enc_biases[l]
				inp = x if l == 0 else self.enc_layers[l-1]
				# print("{} {} {} {}".format(l, w.shape, b.shape, inp.shape))
				self.enc_layers.append(self.activation(tf.add(tf.matmul(inp, w), b)))
			return self.enc_layers[-1]

		def decoder(x):
			# Build L-1 layers of encoder:
			for l in range(L-1):
				w = self.tf_dec_weights[l]
				b = self.tf_dec_biases[l]
				inp = x if l == 0 else self.dec_layers[l-1]
				# print("{} {} {} {}".format(l, w.shape, b.shape, inp.shape))
				self.dec_layers.append(self.activation(tf.add(tf.matmul(inp, w), b)))
			return self.dec_layers[-1]

		# Construct the entire autoencoder model:
		self.encoder_operator = encoder(self.X_in)
		self.decoder_operator = decoder(self.encoder_operator)

		# 'Prediction' for training of the entire system:
		self.y_pred = self.decoder_operator
		# Target values (input or noisy inputs)
		self.y_true = self.X_in

		# Define loss function and iterative optimizer to use:
		# (Mean Squared Error)
		self.loss = tf.reduce_mean(tf.square(self.y_pred - self.y_true))
		self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
		# self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
		# self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss)

		# State saving vars:
		self.main_dir = os.getcwd()
		self.model_name = model_name
		if use_disk:
			self.use_disk = True
			self.models_dir, self.tf_summary_dir = self._create_dirs()
			self.model_path = self.models_dir + self.model_name
		else:
			self.use_disk = False
			self.models_dir, self.tf_summary_dir = (None, None)
			self.model_path = None
		self.tf_session = None
		self.tf_merged_summaries = None
		self.tf_summary_writer = None
		self.tf_saver = None
		self.print_status = status

	# Create dirs for models and tensorflow summaries
	def _create_dirs(self):
		assert(self.use_disk)
		# Making sure we have a valid main dir:
		self.main_dir = self.main_dir + '/' if self.main_dir[-1] != '/' else self.main_dir

		models_dir = self.main_dir + "model"
		if not os.path.isdir(models_dir):
			os.mkdir(models_dir)

		summaries_dir = self.main_dir + "tf_summaries"
		if not os.path.isdir(summaries_dir):
			os.mkdir(summaries_dir)

		return models_dir, summaries_dir



	# Initialize model:
	def _init_tf(self, init_op, restore):
		self.tf_merged_summaries = tf.merge_all_summaries()
		self.tf_saver = tf.train.Saver()

		self.tf_session.run(init_op)

		if restore and self.use_disk:
			self.tf_saver.restore(self.tf_session, self.model_path)

		if self.use_disk:
			self.tf_summary_writer = tf.train.SummaryWriter(self.tf_summary_dir, \
														self.tf_session.graph)

	# Restore model from disk:
	# def load_model(self, model_path=self.model_path):
	# 	init_op = tf.initialze_all_variables()
	# 	self.tf_saver = tf.train.Saver()

	# 	with tf.Session() as self.tf_session:
	# 		self.tf_session.run(init_op)
	# 		self.tf_saver.restore(self.tf_session, model_path)


	# Supply input data X to train autoencoder - note that the input should be a matrix
	def fit(self, X, restore_model=False):
		num_examples = X.shape[0]
		batch_sz = min(self.batch_size, num_examples)
		# With the entire model constructed, initialize all vars to their defaults:
		self.init = tf.global_variables_initializer()

		# Start training with new tensor flow session:
		with tf.Session() as self.tf_session:
			self._init_tf(self.init, restore_model)

			# Perform training on X:
			for step in range(1, self.num_steps + 1):
				# Prepare data batch (shuffle the data and select a subset)
				perm = np.arange(num_examples)
				np.random.shuffle(perm)
				X_tmp = X[perm]
				start = (batch_sz	* (step-1)) % num_examples
				end = (batch_sz * step) % num_examples
				X_batch = X_tmp[start:end]

				# Optimize over the current batch:
				_, l = self.tf_session.run([self.optimizer, self.loss], feed_dict={self.X_in: X_batch})
				# Show logs per step:
				if self.print_status and step % display_step == 0:
					print("Step %i: Minibatch Loss: %f" % (step, l))

			# Save model:
			if self.use_disk:
				self.tf_saver.save(self.tf_session, self.models_dir + self.model_name)

			# Return encoded data:
			X_enc = self.tf_session.run(self.encoder_operator, feed_dict={self.X_in: X})
			return X_enc

	def encode_data(self, X):
		# Create session in which to run:
		if self.use_disk:
			with tf.Session() as self.tf_session:
				self.tf_saver.restore(self.tf_session, self.models_dir + self.model_name)
				X_enc = self.tf_session.run(self.encoder_operator, feed_dict={self.X_in: X})
				return X_enc
		else:
			X_enc = self.tf_session.run(self.encoder_operator, feed_dict={self.X_in: X})
			return X_enc


	def decode_data(self, X):
		# Create session in which to run:
		if self.use_disk:
			with tf.Session() as self.tf_session:
				self.tf_saver.restore(self.tf_session, self.models_dir + self.model_name)
				X_dec = self.tf_session.run(self.decoder_operator, feed_dict={self.X_in: X})
				return X_dec
		else:
			X_dec = self.tf_session.run(self.decoder_operator, feed_dict={self.X_in: X})
			return X_dec

if __name__ == "__main__":
	print("This is the autoencoder module. Include in main code to use as a class.")
	print("Commencing test...")

	import numpy.random as rnd

	rnd.seed(4)
	m = 200
	w1, w2 = 0.1, 0.3
	noise = 0.1

	angles = rnd.rand(m) * 3 * np.pi / 2 - 0.5
	data = np.empty((m, 3))
	data[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * rnd.randn(m) / 2
	data[:, 1] = np.sin(angles) * 0.7 + noise * rnd.randn(m) / 2
	data[:, 2] = data[:, 0] * w1 + data[:, 1] * w2 + noise * rnd.randn(m)

	input_dim = 3
	layer_dims = [2]

	AE = AutoEncoder(input_dim, layer_dims, \
		use_disk=False, \
		tie_weights=True, \
		activation='sigmoid',\
		status=True )

	enc_data = AE.fit(data)
	# for i in range(len(AE.tf_enc_weights)):
	# 	print(AE.tf_enc_weights[i].values)
