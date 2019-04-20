import numpy as np
from Graph.shuffle import shuffle


def ba(num_nodes, M):
	degree = np.zeros([num_nodes])
	if num_nodes <= M:
		raise ValueError("N should be bigger than M, otherwise ,there are no"
		                 "edge")
	edges = []
	for x in range(M):
		degree[x] = 1.0
		degree[M] += 1.0
		edges.append((x,M))
	# degree[M] = M
	for x in range(M+1,num_nodes):
		
		choice = np.random.choice(range(num_nodes),
		                          M,
		                          replace=False,
		                          p=degree/(2.0*M*(x-M)))
		for c in choice:
			edges.append((c,x))
			degree[c] += 1.0
			degree[x] += 1.0
	edges = shuffle(num_nodes,edges)
	return edges

if __name__ == '__main__':
	print(ba(10,4))