import tensorflow as tf

flags = tf.flags

flags.DEFINE_float('random_initializer_stddev',0.1,'set the para stddev of random initializer.')

flags.DEFINE_string('random_graph','ws','you also can choose ba or er.')
flags.DEFINE_float('prob',0.4,'define the prob of random graph.')
flags.DEFINE_integer('M',5,'the para of ba.')
flags.DEFINE_integer('K',2,'the para of ws.')
flags.DEFINE_integer('num_nodes',5,'the num of each layer.')
flags.DEFINE_integer('channel',200,'the num of channels.')

flags.DEFINE_float('learning_rate',0.01,'define the initial learning rate.')
flags.DEFINE_integer('decay_steps',1000,'the interval of decay.')
flags.DEFINE_float('decay_rate',0.96,'the rate of decay.')
flags.DEFINE_bool('staircase',False,'control decay shape.')
flags.DEFINE_bool('training',True,'set the mode.')
flags.DEFINE_integer('img_size',28,'the height or width of img.')
flags.DEFINE_integer('img_channel',1,'the size of img channel.')
flags.DEFINE_integer('num_labels',10,'the size of image labels')
cfg = tf.flags.FLAGS