'''
ER model is a random graph generator.
It has a para P that defines the probability
between two nodes.
To generate dag, we decide the direction from
the lower index node to the higher index node.
'''

import numpy as np
from Graph.shuffle import shuffle


def er(nodes,prob):
	
	rand_matrix = np.random.uniform(0.0,1.0,size=(nodes,nodes))
	edges = []
	for x in range(nodes):
		for y in range(x+1,nodes):
			if rand_matrix[x,y]<=prob:
				edges.append((x,y))
	
	edges = shuffle(nodes,edges)
	return edges

if __name__ == '__main__':
	pass