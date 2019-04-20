import functools
import tensorflow as tf
record = {}
'''
there has a drawback. When i try to call func without
specifying para name, it will have an error.
Now I change my strategy.
'''
# def get_unique_name(func):
# 	@functools.wraps(func)
# 	def tran(*args,**kwargs):
# 		if func.__name__ not in record:
# 			record[func.__name__] = 1
# 		else:
# 			record[func.__name__] += 1
# 		kwargs.pop('id')
# 		return func(*args,**kwargs,id=record[func.__name__])
# 	return tran

'''
we throw away the para called id.
'''
def get_unique_name(func):
	@functools.wraps(func)
	def wrapper(*args,**kwargs):
		if func.__name__ not in record:
			record[func.__name__] = 1
		else:
			record[func.__name__] += 1
			
		with tf.variable_scope(func.__name__+'_%d' % record[func.__name__]):
			result = func(*args,**kwargs)
		return result
	return wrapper

