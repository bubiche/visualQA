import tf.contrib.layers.xavier_initializer as xavier

def xavier_var(name, shape):
	return tf.get_variable(name = name, shape = shape,
		dtype = tf.float32, initializer = xavier())

def _log(*msgs):
	for msg in list(msgs):
		print(msg)