import tensorflow as tf
from Node import NodeOp
from functools import partial
from math import ceil
def DAGLayer(input_tensor, in_channel,
             out_channel, num_nodes, edges):
	
	adjlist = [[] for _ in range(num_nodes)]
	rev_adjlist = [[] for _ in range(num_nodes)]
	
	in_degree = [0 for _ in range(num_nodes)]
	out_degree = [0 for _ in range(num_nodes)]
	
	# initialize graph structure.
	for s, e in edges:
		in_degree[e] += 1
		out_degree[s] += 1
		adjlist[s].append(e)
		rev_adjlist[e].append(s)
	
	inputs_nodes = [x for x in range(num_nodes)
	                if in_degree[x] == 0]
	output_nodes = [x for x in range(num_nodes)
	                if out_degree[x] == 0]
	
	
	if len(inputs_nodes) == 0 or \
		len(output_nodes) == 0 :
		raise RuntimeError("No input nodes or output nodes")
	
	for node in inputs_nodes:
		rev_adjlist[node].append(-1)
	
	# specify each node op.
	node_op = []
	
	for x in range(num_nodes):
		
		node_op.append(
			partial(
				NodeOp,
				in_degree=max(1,in_degree[x]),
				in_channel=in_channel,
				out_channel=out_channel if x in output_nodes else in_channel,
				stride=2 if x in inputs_nodes else 1))
	
	outputs = [None for _ in range(num_nodes)] + [input_tensor]
	
	# topological sort
	while len(inputs_nodes)>0:
		
		now = inputs_nodes[0]
		inputs_nodes.remove(now)
		# get all predecessor.
		input_list = [outputs[x] for x in rev_adjlist[now]]
		# for such node that has more than one predecessor,
		# we just stack it on the last dim.
		feed = tf.stack(input_list,axis=-1)
		outputs[now] = node_op[now](feed)
		
		for v in adjlist[now]:
			in_degree[v] -= 1
			if in_degree[v] == 0:
				inputs_nodes.append(v)
		
	out_list = [outputs[x] for x in output_nodes]
	result =  tf.reduce_mean(tf.stack(out_list,-1),-1)
	input_shape = input_tensor.get_shape().as_list()
	
	result.set_shape([None,int(ceil(input_shape[1]/2.0)),int(ceil(input_shape[2]/2.0)),out_channel])
	return result

if __name__ == '__main__':
	x = tf.constant(1.0, shape=[2, 200, 200, 30])
	y = tf.reduce_mean(tf.stack([x],axis=-1),-1)
	print(y.get_shape())