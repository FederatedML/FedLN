import tensorflow as tf

class Distiller(tf.keras.Model):

	def __init__(self, model, features, idxs):
		super(Distiller, self).__init__()
		self.model = model
		self.features = tf.convert_to_tensor(features, dtype=tf.float32)
		self.idxs = tf.cast(tf.constant(idxs), dtype=tf.int32)
		self.table = tf.lookup.experimental.DenseHashTable(key_dtype=tf.int32, value_dtype=tf.float32, default_value=tf.zeros(shape=(features.shape[1],), dtype=tf.float32), empty_key=-2, deleted_key=-3, name='table')
		self.table.insert(self.idxs, self.features)

	def compile(self, optimizer, loss, metrics, alpha=10.0, temperature=4.0,):
		super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
		self.student_loss_fn = loss
		self.distillation_loss_fn = tf.keras.losses.MeanAbsoluteError(name='mae_loss', reduction=tf.keras.losses.Reduction.AUTO)
		self.alpha = alpha
		self.temperature = temperature

	def train_step(self, data):
		((x, y), idxs) = data
		supervision_signal = self.table.lookup(tf.cast(idxs, dtype=tf.int32))
		with tf.GradientTape() as tape:
			preds, embeddings = self.model(x, training=True)
			loss = self.student_loss_fn(y, preds)
			distillation_loss = self.distillation_loss_fn(supervision_signal,embeddings)
			r_loss = tf.add_n(self.model.losses)
			loss  = loss + (self.alpha * distillation_loss) + r_loss
		# Compute gradients
		trainable_vars = self.model.trainable_variables
		gradients = tape.gradient(loss, trainable_vars)
		# Update weights
		self.optimizer.apply_gradients(zip(gradients, trainable_vars))
		# Update the metrics configured in `compile()`.
		self.compiled_metrics.update_state(y, preds)
		# Return a dict of performance
		return {"loss": loss, "accuracy": self.metrics[0].result(), "distil_loss": distillation_loss,}

	def test_step(self, data):
		x, y = data
		y_prediction = self.model(x, training=False)
		student_loss = self.student_loss_fn(y, y_prediction)
		self.compiled_metrics.update_state(y, y_prediction)
		results = {m.name: m.result() for m in self.metrics}
		results.update({"loss": student_loss})
		return results



	



