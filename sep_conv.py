import tensorflow as tf
from utils.unique_name import get_unique_name
from config import cfg

@get_unique_name
def Sep_conv(input_tensor,in_channels,out_channels,kernel_size=3,stride=[1,1],padding="VALID"
	             ,dilation=[1,1]):
	
	DeepConvFilter = tf.get_variable(
		name='DeepConv',
		shape=[kernel_size, kernel_size, in_channels, 1],
		dtype=tf.float32,
		initializer=\
			tf.truncated_normal_initializer(stddev=cfg.random_initializer_stddev))

	PointConvFilter = tf.get_variable(
		name='PointConv',
		shape=[1, 1, in_channels, out_channels],
		dtype=tf.float32,
		initializer=\
			tf.truncated_normal_initializer(stddev=cfg.random_initializer_stddev))
	
	sep_conv = tf.nn.separable_conv2d(
		input_tensor,
		depthwise_filter=DeepConvFilter,
		pointwise_filter=PointConvFilter, strides=[1] + stride + [1],
		padding=padding,
		rate=[1] + dilation + [1],
		name="sep_conv" ,
		data_format="NHWC")
	
	return sep_conv



if __name__ == '__main__':
	x = tf.constant(1.0,shape=[2,200,200,30])
	conv = Sep_conv(x,30,100,3)
	with tf.Session() as sess:
		tf.global_variables_initializer().run()
		print(sess.run(conv).shape)