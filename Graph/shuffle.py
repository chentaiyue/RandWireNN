import numpy as np
'''
In this module, we spirit from https://github.com/seungwonpark/RandWireNN.
just to make graph generator to have more random quality.
'''
def shuffle(num_nodes,edges):
	map = np.random.permutation(range(num_nodes))
	shuffled = []
	
	for edge in edges:
		s,e = edge
		shuffled.append(sorted((map[s],map[e])))
	
	return sorted(shuffled)