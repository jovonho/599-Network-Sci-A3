import json
from os import replace
import time
from igraph import Graph
import numpy as np
from numpy.lib.shape_base import split
import gcn
from networkx.algorithms.node_classification import local_and_global_consistency, harmonic_function
from networkx.classes.function import is_directed, to_undirected, to_directed
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score
import matplotlib.pyplot as plt
import gcn

def load_dataset(dataset):

	real_classic = ["karate", "football", "polblogs", "polbooks", "strike"]
	real_node_label = ["citeseer", "cora", "pubmed"]

	print(f"\nDataset {dataset}")

	if dataset in real_classic:
		G = Graph.Read_GML(f"./data/real-classic/{dataset}.gml")
		G = G.to_networkx()

		# Polbooks dataset has string labels which throws a ValueError when we 
		# try the first line. We convert it to int using ord().
		if dataset == 'polbooks':
			polbooks_map = {
				'l': 1,
				'c': 2,
				'n': 3
			}
			for node in G.nodes:
				prev = G.nodes[node]['value']
				G.nodes[node]['value'] = polbooks_map[prev]

		try:
			labels_true = [ int(G.nodes[node]['value']) for node in G.nodes ]
		except ValueError:
			labels_true = [ ord(G.nodes[node]['value']) for node in G.nodes ]

		n_communities = len(np.unique(labels_true))

	# Requested a GCN graph
	elif dataset in real_node_label:
		G, labels_true, n_communities = gcn.load_data(dataset)

	print(f"Real number of communities: {n_communities}")

	if is_directed(G):
		G = to_undirected(G)

	return G, np.array(labels_true), n_communities



if __name__=='__main__':


	rng = np.random.default_rng()
	datasets = [ "polbooks", "karate", "football", "polblogs", "strike"]

	for dataset in datasets:
		G, labels_true, _ = load_dataset(dataset)

		num_nodes = len(labels_true)
		print(num_nodes)

		NMIs = []
		ARIs = []
		Accs = []
		for n in range(5, 100, 5):	

			portion_drop =  n/100
			test_size = round(num_nodes * portion_drop)

			print(f"Dropping {n}% of labels. Test set size: {test_size}")

			# Repeat 10 times for each test set size
			nmi = []
			ari = []
			acc = []
			for _ in range(10):
				# Copy the graph bc we will be deleting attributes
				G_ = G.copy()

				# Generate random indices that will define the test set
				test_idx = rng.choice(num_nodes, size=test_size, replace=False)
				test_idx = np.sort(test_idx)

				# Extract test set labels
				y_test = np.array(labels_true)[test_idx]

				# Delete labels of test set in the graph
				for i in test_idx:
					del G_.nodes[i]['value']

				# Get test set predicted labels 
				y_pred = np.array(local_and_global_consistency(G_, label_name='value'), dtype=int)
				y_pred = y_pred[test_idx]

				# Score the predicted labels using various measures
				nmi.append(normalized_mutual_info_score(y_test, y_pred))
				ari.append(adjusted_rand_score(y_test, y_pred))
				acc.append(accuracy_score(y_test, y_pred))

			NMI = np.mean(nmi)
			ARI = np.mean(ari)
			Acc = np.mean(acc)
			print(f"NMI: {NMI}\nARI: {ARI}\nAcc: {Acc}\n")

			NMIs.append(NMI)
			ARIs.append(ARI)
			Accs.append(Acc)


		fig, ax = plt.subplots(figsize=(9,6))
		x_axis = [x for x in range(5, 100, 5)]

		ax.plot(x_axis, Accs, c='pink', label='Accuracy')
		ax.plot(x_axis, ARIs, c='orange', label='ARI')
		ax.plot(x_axis, NMIs, c='purple', label='NMI')

		ax.set_xlabel("Test Set Size (%)")
		ax.set_ylabel("Score")
		ax.legend()

		# Major ticks every 20, minor ticks every 5
		x_major_ticks = np.arange(0, 101, 20)
		x_minor_ticks = np.arange(0, 101, 5)
		y_major_ticks = np.arange(0, 1, .2)
		y_minor_ticks = np.arange(0, 1, .05)

		ax.set_xticks(x_major_ticks)
		ax.set_xticks(x_minor_ticks, minor=True)
		ax.set_yticks(y_major_ticks)
		ax.set_yticks(y_minor_ticks, minor=True)
		
		ax.grid(True, which='both', linestyle='--')
		ax.tick_params(which='both', direction="in", grid_color='grey', grid_alpha=0.2)

		fig.suptitle(f"Local and Global Consistency Node Prediction Performance for \"{dataset}\" Dataset")
		fig.tight_layout()

		fig.savefig(f"./figs/q1_lgc/{dataset}.png", format="png")
		plt.close()


