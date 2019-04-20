import numpy as np
from Graph.shuffle import shuffle

def ws(nodes,K,P):
	if K%2 != 0:
		raise ValueError("K must be even.")
	adj_mat = np.zeros((nodes,nodes))
	
	for i in range(nodes):
		for j in range(i-K//2, i+K//2):
			real_j = j%nodes
			if real_j == i:
				continue
			adj_mat[real_j,i] = adj_mat[i,real_j] = 1
	re_wire = np.random.uniform(0.0,1.0,(nodes,nodes))
	for i in range(nodes):
		for j in range(i+1,i+1+K//2):
			real_j = j%nodes
			if re_wire[i,real_j]<=P:
				unoccupied = [x for x in range(nodes) if not adj_mat[i][x]]
				choice = np.random.choice(unoccupied,1,replace=False)
				adj_mat[i,choice]=adj_mat[choice,i]=1
				adj_mat[i, real_j] = adj_mat[real_j, i] = 1
	
	edges = []
	for i in range(nodes):
		for j in range(i+1,nodes):
			if adj_mat[i,j]==1:
				edges.append((i,j))
	
	return shuffle(nodes,edges)

if __name__ == '__main__':
	print(ws(10,4,0.5))