import tensorflow as tf

class ResNet20:

	@staticmethod
	def regularized_padded_conv(*args, **kwargs):
		return tf.keras.layers.Conv2D(*args, **kwargs, padding="same", kernel_regularizer=_regularizer, kernel_initializer="he_normal", use_bias=False)

	@staticmethod
	def bn_relu(x):
		x = tf.keras.layers.BatchNormalization()(x)
		return tf.keras.layers.ReLU()(x)

	@staticmethod
	def shortcut(x, filters, stride, mode):
		if x.shape[-1] == filters:
			return x
		elif mode == "B":
			return __class__.regularized_padded_conv(filters, 1, strides=stride)(x)
		elif mode == "B_original":
			x = __class__.regularized_padded_conv(filters, 1, strides=stride)(x)
			return tf.keras.layers.BatchNormalization()(x)
		elif mode == "A":
			return tf.pad(tf.keras.layers.MaxPool2D(1, stride)(x) if stride>1 else x, paddings=[(0, 0), (0, 0), (0, 0), (0, filters - x.shape[-1])])
		else:
			raise KeyError("Parameter shortcut_type not recognized!")

	@staticmethod
	def original_block(x, filters, stride=1, **kwargs):
		c1 = __class__.regularized_padded_conv(filters, 3, strides=stride)(x)
		c2 = __class__.regularized_padded_conv(filters, 3)(__class__.bn_relu(c1))
		c2 = tf.keras.layers.BatchNormalization()(c2)

		mode = "B_original" if _shortcut_type == "B" else _shortcut_type
		x = __class__.shortcut(x, filters, stride, mode=mode)
		x = tf.keras.layers.Add()([x, c2])
		return tf.keras.layers.ReLU()(x)

	@staticmethod
	def preactivation_block(x, filters, stride=1, preact_block=False):
		flow = __class__.bn_relu(x)
		if preact_block:
			x = flow
		c1 = __class__.regularized_padded_conv(filters, 3, strides=stride)(flow)
		if _dropout:
			c1 = tf.keras.layers.Dropout(_dropout)(c1)
		c2 = __class__.regularized_padded_conv(filters, 3)(__class__.bn_relu(c1))
		x = __class__.shortcut(x, filters, stride, mode=_shortcut_type)
		return x + c2

	@staticmethod
	def bootleneck_block(x, filters, stride=1, preact_block=False):
		flow = __class__.bn_relu(x)
		if preact_block:
			x = flow
		c1 = __class__.regularized_padded_conv(filters//_bootleneck_width, 1)(flow)
		c2 = __class__.regularized_padded_conv(filters//_bootleneck_width, 3, strides=stride)(__class__.bn_relu(c1))
		c3 = __class__.regularized_padded_conv(filters, 1)(__class__.bn_relu(c2))
		x = __class__.shortcut(x, filters, stride, mode=_shortcut_type)
		return x + c3

	@staticmethod
	def group_of_blocks(x, block_type, num_blocks, filters, stride, block_idx=0):
		global _preact_shortcuts
		preact_block = True if _preact_shortcuts or block_idx == 0 else False

		x = block_type(x, filters, stride, preact_block=preact_block)
		for i in range(num_blocks-1):
			x = block_type(x, filters)
		return x

	@staticmethod
	def Resnet(input_shape, n_classes, l2_reg=1e-4, group_sizes=(2, 2, 2), features=(16, 32, 64), strides=(1, 2, 2),
		shortcut_type="B", block_type="preactivated", first_conv={"filters": 16, "kernel_size": 3, "strides": 1},
		dropout=0, cardinality=1, bootleneck_width=4, preact_shortcuts=True, embeddings_dim=None):

		global _regularizer, _shortcut_type, _preact_projection, _dropout, _cardinality, _bootleneck_width, _preact_shortcuts
		_bootleneck_width = bootleneck_width
		_regularizer = tf.keras.regularizers.l2(l2_reg)
		_shortcut_type = shortcut_type
		_cardinality = cardinality
		_dropout = dropout
		_preact_shortcuts = preact_shortcuts

		block_types = {"preactivated": __class__.preactivation_block,
					"bootleneck": __class__.bootleneck_block,
					"original": __class__.original_block}

		selected_block = block_types[block_type]
		inputs = tf.keras.layers.Input(shape=input_shape)
		flow = __class__.regularized_padded_conv(**first_conv)(inputs)

		if block_type == "original":
			flow = __class__.bn_relu(flow)

		for block_idx, (group_size, feature, stride) in enumerate(zip(group_sizes, features, strides)):
			flow = __class__.group_of_blocks(flow, block_type=selected_block, num_blocks=group_size, block_idx=block_idx, filters=feature, stride=stride)

		if block_type != "original":
			flow = __class__.bn_relu(flow)

		flow = tf.keras.layers.GlobalAveragePooling2D()(flow)
		outputs = tf.keras.layers.Dense(n_classes, kernel_regularizer=_regularizer)(flow)

		if embeddings_dim is not None: 
			embeddings =  tf.keras.layers.Dense(embeddings_dim)(flow)
			return tf.keras.models.Model(inputs, [outputs, embeddings], name='audio_classifier')

		model = tf.keras.models.Model(inputs, outputs)
		return model


def load_model(input_shape, num_classes, l2_reg=1e-4, shortcut_type="A", block_type="original", embeddings_dim=None):
	return ResNet20.Resnet(input_shape=input_shape, n_classes=num_classes, l2_reg=l2_reg, embeddings_dim=embeddings_dim,
							group_sizes=(3, 3, 3), features=(16, 32, 64), strides=(1, 2, 2),
							first_conv={"filters": 16, "kernel_size": 3, "strides": 1},
							shortcut_type=shortcut_type, block_type=block_type, preact_shortcuts=False)

def load_initial_model_weights(input_shape, num_classes):
	return ResNet20.load_model(input_shape=input_shape, num_classes=num_classes).get_weights()