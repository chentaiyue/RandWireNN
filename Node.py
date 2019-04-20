import tensorflow as tf
from sep_conv import Sep_conv
from utils.unique_name import get_unique_name
from config import cfg

@get_unique_name
def NodeOp(input_tensor,in_degree,in_channel,out_channel,stride):
	single = (in_degree==1)
	'''
	when in-degree is bigger than 1, we will play linear combination
	on input-tensor.
	we assume the input tensor shape is [batch,img_h,img_w,c,in_degree]
	in this case.
	when in-degree is equal to 1, this kind of nodes are input nodes. so,
	we needn't to do such linear transformation and just to keep original
	shape.
	'''
	if not single:
		if len(input_tensor.get_shape()) != 5:
			raise ValueError("input tensor must be 5 dims.")
		coeff = tf.get_variable(
			name="coeff",
			shape=[in_degree],
			dtype=tf.float32,
			initializer=\
				tf.random_normal_initializer(stddev=cfg.random_initializer_stddev))
		
		input = tf.reduce_sum(tf.sigmoid(coeff)*input_tensor,axis=-1)
	else:
		
		input = tf.squeeze(input_tensor)
	
	_relu = tf.nn.relu(input)
	_depth_conv = Sep_conv(_relu,in_channel,out_channel,kernel_size=3,padding="SAME",stride=[stride,stride])
	
	_bn = tf.layers.batch_normalization(
		_depth_conv,-1,0.997,1e-5,training=True
	)
	
	return _bn

if __name__ == "__main__":
	pass