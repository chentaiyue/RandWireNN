import numpy as np
from Graph.shuffle import shuffle

def ba(nodes,M):
	degree = np.zeros([nodes])
	if nodes<=M:
		raise ValueError("N should be bigger than M, otherwise ,there are no"
		                 "edge")
	edges = []
	for x in range(M):
		degree[x] = 1.0
		degree[M] += 1.0
		edges.append((x,M))
	#degree[M] = M
	for x in range(M+1,nodes):
		
		choice = np.random.choice(range(nodes),M,replace=False,p=degree/(2.0*M*(x-M)))
		for c in choice:
			edges.append((c,x))
			degree[c] += 1.0
			degree[x] += 1.0
	edges = shuffle(nodes,edges)
	return edges

if __name__ == '__main__':
	print(ba(10,4))