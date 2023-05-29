import tensorflow as tf

def log_t(u, t):
	if t == 1.0: return tf.math.log(u)
	else: return (u ** (1.0 - t) - 1.0) / (1.0 - t)

def exp_t(u, t):
	if t == 1.0: return tf.math.exp(u)
	else: return tf.math.maximum(0.0, 1.0 + (1.0 - t) * u) ** (1.0 / (1.0 - t))

def compute_normalization_fixed_point(y_pred, t, num_iters=5):
	mu = tf.math.reduce_max(y_pred, -1, keepdims=True)
	normalized_y_pred_step_0 = y_pred - mu
	normalized_y_pred = normalized_y_pred_step_0
	i = 0
	while i < num_iters:
		i += 1
		logt_partition = tf.math.reduce_sum(exp_t(normalized_y_pred, t), -1, keepdims=True)
		normalized_y_pred = normalized_y_pred_step_0 * (logt_partition ** (1.0 - t))
	logt_partition = tf.math.reduce_sum(exp_t(normalized_y_pred, t), -1, keepdims=True)
	return -log_t(1.0 / logt_partition, t) + mu

def compute_normalization(y_pred, t, num_iters=5):
	if t < 1.0:
		return None
	else:
		return compute_normalization_fixed_point(y_pred, t, num_iters)

class BiTemperedLogisticLoss(tf.keras.losses.Loss):

	def __init__(self, t1=0.8, t2=1.4, n_iter=5, from_logits=False, label_smoothing=0.0, name='loss'):
		super(BiTemperedLogisticLoss, self).__init__(name=name)
		self.t1 = t1
		self.t2 = t2
		self.n_iter = n_iter
		self.label_smoothing = label_smoothing
		self.from_logits = from_logits

	@staticmethod
	def tempered_softmax(y_pred, t, num_iters=5):
		if t == 1.0:
			normalization_constants = tf.math.log(tf.math.reduce_sum(tf.math.exp(y_pred), -1, keepdims=True))
		else:
			normalization_constants = compute_normalization(y_pred, t, num_iters)
		return exp_t(y_pred - normalization_constants, t)

	@staticmethod
	def loss_fn(y_pred, y_true, t1, t2, num_iters=5, label_smoothing=0.0):
		y_pred = tf.cast(y_pred, tf.float32)
		y_true = tf.cast(y_true, tf.float32)

		if label_smoothing > 0.0:
			num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
			y_true = (1 - num_classes / (num_classes - 1) * label_smoothing) * y_true + label_smoothing / (num_classes - 1)

		probabilities = __class__.tempered_softmax(y_pred, t2, num_iters)
		temp1 = (log_t(y_true + 1e-10, t1) - log_t(probabilities, t1)) * y_true
		temp2 = (1 / (2 - t1)) * (tf.math.pow(y_true, 2 - t1) - tf.math.pow(probabilities, 2 - t1))
		loss_values = temp1 - temp2
		return tf.math.reduce_sum(loss_values, -1)


	def call(self, y_true, y_pred):
		return __class__.loss_fn(y_pred, y_true, self.t1, self.t2, self.n_iter, self.label_smoothing)