import tensorflow as tf
import tensorflow_addons as tfa


def _conv_block(x, num_features, l2_reg, dropout_rate=0.1, add_max_pool=True,):
	x_t = tf.keras.layers.Conv2D(num_features, (1, 4), padding = "same", kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
	x_t = tfa.layers.GroupNormalization(groups=4)(x_t)
	x_t = tf.keras.layers.Activation("relu")(x_t)
	x_f = tf.keras.layers.Conv2D(num_features, (4, 1), padding = "same", kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
	x_f = tfa.layers.GroupNormalization(groups=4)(x_f)
	x_f = tf.keras.layers.Activation("relu")(x_f)
	x = tf.keras.layers.Concatenate(axis=-1)([x_t, x_f])
	x = tf.keras.layers.Conv2D(num_features, (1, 1), padding = "same", kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
	x = tfa.layers.GroupNormalization(groups=4)(x)
	x = tf.keras.layers.Activation("relu")(x)
	if add_max_pool:
		x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
	x = tf.keras.layers.SpatialDropout2D(dropout_rate)(x)
	return x

def CNN_Model(input_shape, n_classes, l2_reg=1e-4, dropout_rate=0.1, add_max_pool=True, features=(24, 32, 64,128), embeddings_dim=None):
	inputs = tf.keras.layers.Input(shape=input_shape, name="audio_inputs")
	for block_idx, num_features in enumerate(features):
		flow = _conv_block(flow if block_idx!=0 else inputs, num_features, l2_reg=l2_reg, dropout_rate=dropout_rate, add_max_pool=add_max_pool,)
	flow = tf.keras.layers.GlobalMaxPool2D()(flow)

	outputs = tf.keras.layers.Dense(n_classes)(flow)

	if embeddings_dim is not None: 
		embeddings =  tf.keras.layers.Dense(embeddings_dim)(flow)
		return tf.keras.models.Model(inputs, [outputs,embeddings], name='audio_classifier')
  
	model = tf.keras.models.Model(inputs, outputs, name='audio_classifier')
	return model

def load_model(input_shape, num_classes, dropout_rate=0.1, l2_reg=1e-5, add_max_pool=True, features=(24, 32, 64,128), embeddings_dim=None):
	return CNN_Model(input_shape=input_shape, n_classes=num_classes, l2_reg=l2_reg, dropout_rate=dropout_rate, add_max_pool=add_max_pool, features=features, embeddings_dim=embeddings_dim)

def load_initial_model_weights(input_shape, num_classes, dropout_rate=0.1, l2_reg=1e-5):
	return load_model(input_shape=input_shape, num_classes=num_classes, dropout_rate=dropout_rate, l2_reg=l2_reg).get_weights()