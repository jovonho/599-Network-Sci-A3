import json
from os import remove, replace
import time
from igraph import Graph
from networkx.readwrite.edgelist import generate_edgelist
import numpy as np
from numpy.lib.shape_base import split
from sklearn import metrics
import gcn
from networkx.algorithms.node_classification import local_and_global_consistency, harmonic_function
from networkx.classes.function import is_directed, to_undirected, to_directed
from sklearn.metrics import roc_auc_score, adjusted_rand_score, accuracy_score
import matplotlib.pyplot as plt
import gcn
import pathlib
import networkx as nx
from networkx.algorithms.link_prediction import *
from networkx.algorithms.components import is_connected
from sklearn.linear_model import LogisticRegression


def load_real_label(dataset, add_val_to_test=False):
	G, labels_true, test_idx, num_unlabelled = gcn.load_data(dataset, add_val_to_test)

	if is_directed(G):
		G = to_undirected(G)

	return G, np.array(labels_true), test_idx, num_unlabelled


def load_real_classic(dataset):
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

	if is_directed(G):
		G = to_undirected(G)

	return G, np.array(labels_true), n_communities



def q1_real_classic():

	rng = np.random.default_rng()
	datasets = ["strike", "karate", "polblogs", "polbooks", "football"]

	algos = [local_and_global_consistency, harmonic_function]

	results_dir, _ = create_output_directories("q1")

	for dataset in datasets:

		G, labels_true, _ = load_real_classic(dataset)
		print(f"\nDataset: {dataset}\n")

		scores = {algo.__name__: [] for algo in algos}

		with open(f"{results_dir}/{dataset}.txt", "w") as outfile:

			print("{:21}\t{:10} {:10}".format("Unlabelled Nodes (%)", "LGC", "Harmonic"))
			outfile.write("{:21}\t{:10} {:10}\n".format("Unlabelled Nodes (%)", "LGC", "Harmonic"))

			num_nodes = len(labels_true)

			for n in range(5, 100, 5):	

				scores_tmp = {algo.__name__: [] for algo in algos}

				portion_drop =  n/100
				test_size = round(num_nodes * portion_drop)
				test_size_pct = round(test_size/len(labels_true) * 100, 2)

				# Repeat 10 times for each test set size
				for _ in range(10):

					for algo in algos:

						# Copy the graph bc we will be deleting attributes
						G_ = G.copy()

						# Generate random indices that will define the test set
						test_idx = rng.choice(num_nodes, size=test_size, replace=False)
						test_idx = np.sort(test_idx)

						# Extract test set labels
						y_test = labels_true[test_idx]

						# Delete labels of test set in the graph
						for i in test_idx:
							del G_.nodes[i]['value']

						# Get test set predicted labels 
						y_pred = np.array(algo(G_, label_name='value'), dtype=int)
						y_pred = y_pred[test_idx]

						acc = accuracy_score(y_test, y_pred)

						scores_tmp[algo.__name__].append(acc)

				scores_print = []

				for algo in algos:
					name = algo.__name__
					mean_score = round(np.mean(scores_tmp[name]), 4)
					scores[name].append(mean_score)
					scores_print.append(mean_score)
				
				print("{:5} ({:3}%)\t{:12}\t{:8}".format(test_size, test_size_pct, *scores_print))
				outfile.write("{:5} ({:3}%)\t{:12}\t{:8}\n".format(test_size, test_size_pct, *scores_print))


def q1_real_classic_stacking():

	rng = np.random.default_rng()
	datasets = ["strike", "karate", "polblogs", "polbooks", "football"]

	algos = [local_and_global_consistency, harmonic_function]

	results_dir, _ = create_output_directories("q1/stacking_l2")

	for dataset in datasets:

		G, labels_true, _ = load_real_classic(dataset)

		print(f"\nDataset: {dataset}\n")

		scores = []

		with open(f"{results_dir}/{dataset}.txt", "w") as outfile:

			print("{:21}\t{:10}".format("Unlabelled Nodes (%)", "Stacked Model"))
			outfile.write("{:21}\t{:10}\n".format("Unlabelled Nodes (%)", "Stacked Model"))

			num_nodes = len(labels_true)

			for n in range(5, 100, 5):	

				scores_tmp = []

				portion_drop =  n/100
				test_size = round(num_nodes * portion_drop)
				test_size_pct = round(test_size/len(labels_true) * 100, 2)

				meta_X_train = np.ndarray(shape=(0, len(algos)))
				meta_y_train = np.array([])

				# Stacking model training 
				# We will generate 10 x <testing set size> samples with <num algorithms> features 
				# The features will be the test set classification given by each stacked algorithm
				# We will train a LogisticRegression model on these. 
				for _ in range(10):

					tmp_X_train = np.ndarray(shape=(test_size, 0))

					# Copy the graph bc we will be deleting attributes
					G_ = G.copy()

					# Generate random indices that will define the test set
					test_idx = rng.choice(num_nodes, size=test_size, replace=False)
					test_idx = np.sort(test_idx)

					# Extract test set labels
					test_truth = labels_true[test_idx]
					# test_truth = np.reshape(test_truth, newshape=(test_size,1))

					# Delete labels of test set in the graph
					for i in test_idx:
						del G_.nodes[i]['value']

					# Run the node classification algorithms 
					# Store their predicted labels in the meta train set
					for algo in algos:

						test_pred = np.array(algo(G_, label_name='value'), dtype=int)
						# Get test set predicted labels 
						test_pred = test_pred[test_idx]
						test_pred = np.reshape(test_pred, newshape=(test_size,1))

						tmp_X_train = np.column_stack((tmp_X_train, test_pred))

					# Stack these predictions as new samples
					meta_X_train = np.row_stack((meta_X_train, tmp_X_train))
					# Append truth values to last column
					meta_y_train = np.append(meta_y_train, test_truth)


				# Train the meta model
				# We use no penalty term here since we only have 2 features
				metamodel = LogisticRegression(max_iter=3000)
				metamodel = metamodel.fit(meta_X_train, meta_y_train)

				# Test the meta model
				# For this we go through the same flow as for training and generate 

				# Repeat 10 times for each number of dropped labels
				for _ in range(10):
					
					meta_X_test = np.ndarray(shape=(test_size, 0))

					# Copy the graph bc we will be deleting attributes
					G_ = G.copy()

					# Generate random indices that will define the test set
					test_idx = rng.choice(num_nodes, size=test_size, replace=False)
					test_idx = np.sort(test_idx)

					# Extract test set labels
					meta_y_test = labels_true[test_idx]

					# Delete labels of test set in the graph
					for i in test_idx:
						del G_.nodes[i]['value']

					# Run the node classification algorithms 
					for algo in algos:

						test_pred = np.array(algo(G_, label_name='value'), dtype=int)
						# Get test set predicted labels 
						test_pred = test_pred[test_idx]
						test_pred = np.reshape(test_pred, newshape=(test_size,1))

						# Create our test set based on predictions from individual algos
						meta_X_test = np.column_stack((meta_X_test, test_pred))

					scores_tmp.append(metamodel.score(meta_X_test, meta_y_test))


				mean_score = round(np.mean(scores_tmp), 4)
				scores.append(mean_score)
				
				print("{:5} ({:3}%)\t{:12}".format(test_size, test_size_pct, mean_score))
				outfile.write("{:5} ({:3}%)\t{:12}\n".format(test_size, test_size_pct, mean_score))



# This verison kept the exact same split as the GCN paper
# Since the split was fixed, we couldn't take the average score over 10 iterations
def q1_real_label_old():
	datasets = ["citeseer", "cora", "pubmed"]
	algos = [local_and_global_consistency, harmonic_function]
	results_dir, _ = create_output_directories("q1")

	for dataset in datasets:

		print(f"\nDataset: {dataset}\n")

		scores = {algo.__name__: [] for algo in algos}

		with open(f"{results_dir}/{dataset}.txt", "w") as outfile:

			print("{:21}\t{:10} {:10}".format("Unlabelled Nodes (%)", "LGC", "Harmonic"))
			outfile.write("{:21}\t{:10} {:10}\n".format("Unlabelled Nodes (%)", "LGC", "Harmonic"))

			# We keep the original train/test split but also
			# try on an extended training set containing the validation set as well
			for add_val in [True, False]:

				G, labels_true, test_idx, num_unlabelled = load_real_label(dataset, add_val)
				num_unlabelled_pct = round(num_unlabelled/len(labels_true) * 100, 2)

				# Can't repreat 10 times here
				# Since we are keeping the existing train/test split 
				# it always gives the same result
				for algo in algos:

					# Copy the graph bc we will be deleting attributes
					G_ = G.copy()

					# Extract test set labels
					y_test = labels_true[test_idx]

					# Get test set predicted labels 
					y_pred = np.array(algo(G_, label_name='value'), dtype=int)
					y_pred = y_pred[test_idx]

					acc = accuracy_score(y_test, y_pred)

					scores[algo.__name__].append(acc)
				
				print("{:5} ({:3}%)\t{:12}\t{:8}".format(num_unlabelled, num_unlabelled_pct, scores["local_and_global_consistency"][-1], scores["harmonic_function"][-1]))
				outfile.write("{:5} ({:3}%)\t{:12}\t{:8}\n".format(num_unlabelled, num_unlabelled_pct, scores["local_and_global_consistency"][-1], scores["harmonic_function"][-1]))



def q1_real_label():
	rng = np.random.default_rng()
	datasets = ["citeseer", "cora", "pubmed"]
	algos = [local_and_global_consistency, harmonic_function]
	results_dir, _ = create_output_directories("q1")

	for dataset in datasets:

		print(f"\nDataset: {dataset}\n")

		scores = {algo.__name__: [] for algo in algos}

		with open(f"{results_dir}/{dataset}.txt", "w") as outfile:

			print("{:21}\t{:10} {:10}".format("Unlabelled Nodes (%)", "LGC", "Harmonic"))
			outfile.write("{:21}\t{:10} {:10}\n".format("Unlabelled Nodes (%)", "LGC", "Harmonic"))

			# We consider the number of unlabelled nodes in the original set to
			# Keep the same split sizes but randomize which nodes are labelled 
			# This way we can average over 10 iterations
			num_labelled_nodes = {
				'citeseer' : [620, 120],
				'cora': [640, 140],
				'pubmed': [560, 60]
			}
			for num_labelled in num_labelled_nodes[dataset]:

				G, labels_true = gcn.load_data_fully_labeled(dataset)

				num_nodes = len(labels_true)
				num_unlabelled = len(labels_true) - num_labelled
				num_unlabelled_pct = round(num_unlabelled/len(labels_true) * 100, 2)

				scores_tmp = {algo.__name__: [] for algo in algos}

				for _ in range(10):

					# Copy the graph bc we will be deleting attributes
					G_ = G.copy()

					# We need the testing set to be 1000 long like in the paper
					# But we need the number of unlabelled nodes to be much bigger
					unlabelled_idx = rng.choice(num_nodes, size=num_unlabelled, replace=False)
					test_idx = unlabelled_idx[:1000]

					# Extract test set labels
					test_idx = np.sort(test_idx)
					test_truth = labels_true[test_idx]

					# Delete labels
					for i in unlabelled_idx:
						del G_.nodes[i]['value']

					# Run the node classification algorithms 
					# Store their predicted labels in the meta train set
					for algo in algos:

						# Get test set predicted labels 
						y_pred = np.array(algo(G_, label_name='value'), dtype=int)
						y_pred = y_pred[test_idx]

						acc = accuracy_score(test_truth, y_pred)
						# print(f"numlabel {num_labelled} iter {_} algo {algo.__name__} acc {acc}")

						scores_tmp[algo.__name__].append(acc)

				scores_print = []

				for algo in algos:
					name = algo.__name__
					mean_score = round(np.mean(scores_tmp[name]), 4)
					scores[name].append(mean_score)
					scores_print.append(mean_score)
				
				print("{:5} ({:3}%)\t{:12}\t{:8}".format(num_unlabelled, num_unlabelled_pct, *scores_print))
				outfile.write("{:5} ({:3}%)\t{:12}\t{:8}\n".format(num_unlabelled, num_unlabelled_pct, *scores_print))


def q1_real_label_stacking():
	rng = np.random.default_rng()
	datasets = ["citeseer", "cora", "pubmed"]
	algos = [local_and_global_consistency, harmonic_function]
	results_dir, _ = create_output_directories("q1/stacking_l2")

	for dataset in datasets:

		print(f"\nDataset: {dataset}\n")

		with open(f"{results_dir}/{dataset}.txt", "w") as outfile:
			
			# For the stacking model we want to generate multiple different samples so we keep the original split
			# in terms of percentage of labelled nodes but randomize the process

			num_labelled_nodes = {
				'citeseer' : [620, 120],
				'cora': [640, 140],
				'pubmed': [560, 60]
			}
			for num_labelled in num_labelled_nodes[dataset]:

				G, labels_true = gcn.load_data_fully_labeled(dataset)

				num_nodes = len(labels_true)
				num_unlabelled = len(labels_true) - num_labelled
				num_unlabelled_pct = round(num_unlabelled/len(labels_true) * 100, 2)

				# The test_size is always 1000 in the original paper
				# Though the number of unlabelled nodes is much bigger
				test_size = 1000

				meta_X_train = np.ndarray(shape=(0, len(algos)))
				meta_y_train = np.array([])

				# Stacking model training 
				# We will generate 10 x <testing set size> samples with <num algorithms> features 
				# The features will be the test set classification given by each stacked algorithm
				# We will train a LogisticRegression model on these. 
				for _ in range(10):

					tmp_X_train = np.ndarray(shape=(test_size, 0))

					# Copy the graph bc we will be deleting attributes
					G_ = G.copy()

					# We need the testing set to be 1000 long like in the paper
					# But we need the numbe rof unlabelled nodes to be much bigger
					unlabelled_idx = rng.choice(num_nodes, size=num_unlabelled, replace=False)
					test_idx = unlabelled_idx[:1000]

					test_idx = np.sort(test_idx)

					# Extract test set labels
					test_truth = labels_true[test_idx]

					# Delete labels
					for i in unlabelled_idx:
						del G_.nodes[i]['value']

					# Run the node classification algorithms 
					# Store their predicted labels in the meta train set
					for algo in algos:
						test_pred = np.array(algo(G_, label_name='value'), dtype=int)
						# Get test set predicted labels 
						test_pred = test_pred[test_idx]
						test_pred = np.reshape(test_pred, newshape=(test_size,1))

						tmp_X_train = np.column_stack((tmp_X_train, test_pred))

					# Stack these predictions as new samples
					meta_X_train = np.row_stack((meta_X_train, tmp_X_train))
					# Append truth values to last column
					meta_y_train = np.append(meta_y_train, test_truth)

				
				metamodel_scores = []

				print("{:21}\t{:10}".format("Unlabelled Nodes (%)", "Stacked Model"))
				outfile.write("{:21}\t{:10}\n".format("Unlabelled Nodes (%)", "Stacked Model"))

				# Train the meta model
				# We use no penalty term here since we only have 2 features
				metamodel = LogisticRegression(max_iter=3000)
				metamodel = metamodel.fit(meta_X_train, meta_y_train)

				# Test the meta model
				# For this we go through the same flow as for training and generate 

				# Repeat 10 times for each number of dropped labels
				for _ in range(10):

					metamodel_scores_tmp = []
					
					meta_X_test = np.ndarray(shape=(test_size, 0))

					# Copy the graph bc we will be deleting attributes
					G_ = G.copy()

					# Generate random indices that will define the test set
					test_idx = rng.choice(num_nodes, size=test_size, replace=False)
					test_idx = np.sort(test_idx)

					# Extract test set labels
					meta_y_test = labels_true[test_idx]

					# Delete labels of test set in the graph
					for i in test_idx:
						del G_.nodes[i]['value']

					# Run the node classification algorithms 
					for algo in algos:

						test_pred = np.array(algo(G_, label_name='value'), dtype=int)
						# Get test set predicted labels 
						test_pred = test_pred[test_idx]
						test_pred = np.reshape(test_pred, newshape=(test_size,1))

						# Create our test set based on predictions from individual algos
						meta_X_test = np.column_stack((meta_X_test, test_pred))

					metamodel_scores_tmp.append(metamodel.score(meta_X_test, meta_y_test))

				mean_score = round(np.mean(metamodel_scores_tmp), 4)
				metamodel_scores.append(mean_score)
				
				print("{:5} ({:3}%)\t{:12}".format(num_unlabelled, num_unlabelled_pct, mean_score))
				outfile.write("{:5} ({:3}%)\t{:12}\n".format(num_unlabelled, num_unlabelled_pct, mean_score))



def create_output_directories(question):
	results_dir = f"./results/{question}/"
	pathlib.Path(results_dir).mkdir(parents=True, exist_ok=True)

	figs_dir = f"./figs/{question}/"
	pathlib.Path(figs_dir).mkdir(parents=True, exist_ok=True)

	return results_dir, figs_dir



def q2_combo():

	rng = np.random.default_rng()
	datasets_label = ["citeseer", "cora", "pubmed"]
	datasets_classic = ["strike", "karate", "polblogs", "polbooks", "football"]
	datasets_all = datasets_classic + datasets_label

	results_dir, figs_dir = create_output_directories("q2/test")

	algos = [resource_allocation_index, jaccard_coefficient, adamic_adar_index, 
		preferential_attachment, cn_soundarajan_hopcroft, ra_index_soundarajan_hopcroft, 
		within_inter_cluster, common_neighbor_centrality]
	
	for dataset in datasets_all:

		with open(f"{results_dir}/{dataset}.txt", "w") as results_file:

			if dataset in datasets_label:
				G, _ = gcn.load_data_fully_labeled(dataset)
			else:
				G, _, _ = load_real_classic(dataset)

			# Convert to a simple, undirected graph
			G = nx.Graph(G)

			print(f"\n{dataset} N={len(G.nodes)}, E={len(G.edges)}\n")
			results_file.write(f"{dataset} N={len(G.nodes)}, E={len(G.edges)}\n\n")

			print("{:25}\t{:15}\t{}".format("Algorithm", "Mean AUC", "Runtime"))
			results_file.write("{:25}\t\t{:15}\t\t{}\n".format("Algorithm", "Mean AUC", "Runtime"))
			
			scores = {algo.__name__: [] for algo in algos}

			# We want to remove 20% of edges
			num_edges = len(G.edges)
			num_edges_to_remove = num_edges * 20 // 100

			for algo in algos:
				runtimes = []
				num_iter = 10

				# Took too long to execute
				if (dataset == "polblogs" or dataset in datasets_label) and algo == common_neighbor_centrality:
					continue

				num_successes = 0
				
				while num_successes < num_iter:
					t1 = time.time()

					G_ = G.copy()

					# Take a sample of negative edges (factor x number of removed edges)
					factor = 2
					negative_edges = []
					while len(negative_edges) < factor * num_edges_to_remove:
						# Replacement is False so no self-edges are generated
						edge = rng.choice(G.nodes, size=(1,2), replace=False)
						edge = tuple(edge[0])
						# We only want negative edges here
						if G.has_edge(edge[0], edge[1]):
							continue
						else:
							negative_edges.append(edge)

					# Randomly choose num_edges_to_remove edges to remove
					remove_edge_idx = rng.choice(num_edges, size=num_edges_to_remove, replace=False)

					removed_edges = []
					# Copy the edgelist to access it by index without it changing after we delete one
					G_edgelist = list(G_.edges(data=False))
					for i in remove_edge_idx:
						edge = G_edgelist[i]
						removed_edges.append(edge)
						G_.remove_edge(edge[0], edge[1])

					ebunch = np.concatenate((negative_edges, removed_edges), axis=0)
					y_true = np.concatenate((np.zeros(len(negative_edges)), np.ones(len(removed_edges))))

					if algo in [cn_soundarajan_hopcroft, ra_index_soundarajan_hopcroft, within_inter_cluster]:
						# These algorithms consider the community of nodes when predicting links
						y_score = list(algo(G_, ebunch, community='value'))
					else:
						try:
							y_score = list(algo(G_, ebunch))
						except KeyError:
							# The common_neighbor_centrality algo fails sometimes while computing 
							# shortest paths if G got disconnected after node removal
							continue
						except ZeroDivisionError:
							# I'm not sure why but degree of w in nx.common_neighbors(u,v) seems to return < 2 for some nodes in 
							# a common neighborhood even though we convert to undirected simple graph, let's ignore and retry
							continue

					y_score = [score[2] for score in y_score]

					auc = roc_auc_score(y_true, y_score)

					scores[algo.__name__].append(auc)

					num_successes += 1

					iteration_time = time.time() - t1
					runtimes.append(iteration_time)

				# Compute the mean score for this algorithm
				mean_auc = round(np.mean(scores[algo.__name__]),4)
				mean_time = round(np.mean(runtimes), 3)
				
				print("{:25}\t{:<15}\t{}s".format(algo.__name__, mean_auc, mean_time))
				results_file.write("{:25}\t{:<15}\t{}s\n".format(algo.__name__, mean_auc, mean_time))
				
				# fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
				display = metrics.RocCurveDisplay.from_predictions(y_true, y_score, pos_label=None)
				display.ax_.set_title(f"ROC of {algo.__name__} for Link Prediction")
				# display.plot()
				display.figure_.savefig(f"{figs_dir}/{dataset}_{algo.__name__}.png", format="png")

				plt.close()


def remove_edges_and_generate_ebunch(G, num_edges_to_remove, factor=2):
	rng = np.random.default_rng()
	
	# Take a sample of negative edges (factor x number of removed edges)
	num_edges = len(G.edges)
	negative_edges = []
	while len(negative_edges) < factor * num_edges_to_remove:
		# Replacement is False so no self-edges are generated
		edge = rng.choice(G.nodes, size=(1,2), replace=False)
		edge = tuple(edge[0])
		# We only want negative edges here
		if G.has_edge(edge[0], edge[1]):
			continue
		else:
			negative_edges.append(edge)

	# Randomly choose num_edges_to_remove edges to remove
	remove_edge_idx = rng.choice(num_edges, size=num_edges_to_remove, replace=False)

	removed_edges = []
	# Copy the edgelist to access it by index without it changing after we delete one
	G_edgelist = list(G.edges(data=False))
	for i in remove_edge_idx:
		edge = G_edgelist[i]
		removed_edges.append(edge)
		G.remove_edge(edge[0], edge[1])

	ebunch = np.concatenate((negative_edges, removed_edges), axis=0)
	y_true = np.concatenate((np.zeros(len(negative_edges)), np.ones(len(removed_edges))))

	return ebunch, y_true


def q2_stacking():
	datasets_label = ["citeseer", "cora", "pubmed"]
	datasets_classic = ["strike", "karate", "polblogs", "polbooks", "football"]
	datasets_all = datasets_classic + datasets_label

	results_dir, _ = create_output_directories("q2/stacking_factor2")

	algos = [resource_allocation_index, jaccard_coefficient, adamic_adar_index, 
		preferential_attachment, cn_soundarajan_hopcroft, ra_index_soundarajan_hopcroft, 
		within_inter_cluster]
	
	for dataset in datasets_all:

		with open(f"{results_dir}/{dataset}.txt", "w") as results_file:

			if dataset in datasets_label:
				G, _ = gcn.load_data_fully_labeled(dataset)
			else:
				G, _, _ = load_real_classic(dataset)

			# Convert to a simple, undirected graph
			G = nx.Graph(G)

			print(f"\n{dataset} N={len(G.nodes)}, E={len(G.edges)}\n")
			results_file.write(f"{dataset} N={len(G.nodes)}, E={len(G.edges)}\n\n")

			print("{:25}\t{:15}".format("Algorithm", "Mean AUC"))
			results_file.write("{:25}\t{:15}\n".format("Algorithm", "Mean AUC"))
			
			# We want to remove 20% of edges
			num_edges = len(G.edges)
			num_edges_to_remove = num_edges * 20 // 100

			algos_used = algos.copy()

			# Took too long to execute
			# if (dataset == "polblogs" or dataset in datasets_label):
			# 	algos_used.remove(common_neighbor_centrality)

			if dataset in ["citeseer", "pubmed"]:
				algos_used.remove(adamic_adar_index)


			X_meta_train = np.ndarray(shape=(0, len(algos_used)))
			y_meta_train = np.array([])

			factor = 2

			for _ in range(10):
				print(f"generating test iter {_}")
				# Generate 10 x (factor + 1) x num_edges_to_remove samples with len(algo_used) features 
				# with the features being the score given by each algo 

				G_ = G.copy()
				# Generate the edges to predict. We need to predict the same edges for each algo 
				# so the scores given by each algo are related to each other
				ebunch, y_true = remove_edges_and_generate_ebunch(G_, num_edges_to_remove, factor)

				# To store this round's samples
				# We initialize with 0 columns and will stack the scores 
				round_X_train = np.ndarray(shape=(len(ebunch), 0))

				for algo in algos_used:

					print(f"algo: {algo.__name__}")

					# Some algorithms sometimes fail. Ensure each of them completed
					algo_successful = False
					while not algo_successful:

						# These algorithms consider the community of nodes when predicting links
						if algo in [cn_soundarajan_hopcroft, ra_index_soundarajan_hopcroft, within_inter_cluster]:
							y_score = list(algo(G_, ebunch, community='value'))
						else:
							try:
								y_score = list(algo(G_, ebunch))
							except KeyError as e:
								# The common_neighbor_centrality algo fails sometimes while computing 
								# shortest paths if G got disconnected after node removal
								continue
							except ZeroDivisionError as e:
								# I'm not sure why but degree of w in nx.common_neighbors(u,v) seems to return < 2 for some nodes in 
								# a common neighborhood even though we convert to undirected simple graph, let's ignore and retry
								continue


						y_score = [score[2] for score in y_score]
						algo_successful = True

						round_X_train = np.column_stack((round_X_train, y_score))



				# Stack these predictions as new samples
				X_meta_train = np.row_stack((X_meta_train, round_X_train))
				# Append truth values to our list of truths
				y_meta_train = np.append(y_meta_train, y_true)

			print(X_meta_train.shape)
			print(y_meta_train.shape)

			# Train the Meta model

			metamodel = LogisticRegression().fit(X_meta_train, y_meta_train)

			metamodel_scores = []

			for _ in range(10):

				G_ = G.copy()
				ebunch, meta_y_test = remove_edges_and_generate_ebunch(G_, num_edges_to_remove, factor)

				meta_X_test = np.ndarray(shape=(len(ebunch), 0))

				for algo in algos_used:
					algo_successful = False
					while not algo_successful:

						# These algorithms consider the community of nodes when predicting links
						if algo in [cn_soundarajan_hopcroft, ra_index_soundarajan_hopcroft, within_inter_cluster]:
							y_score = list(algo(G_, ebunch, community='value'))
						else:
							try:
								y_score = list(algo(G_, ebunch))
							except KeyError as e:
								# The common_neighbor_centrality algo fails sometimes while computing 
								# shortest paths if G got disconnected after node removal
								continue
							except ZeroDivisionError as e:
								# I'm not sure why but degree of w in nx.common_neighbors(u,v) seems to return < 2 for some nodes in 
								# a common neighborhood even though we convert to undirected simple graph, let's ignore and retry
								continue


						y_score = [score[2] for score in y_score]
						algo_successful = True

						meta_X_test = np.column_stack((meta_X_test, y_score))	


				meta_y_pred = metamodel.predict(meta_X_test)

				auc = roc_auc_score(meta_y_test, meta_y_pred)

				metamodel_scores.append(auc)


			# Compute the mean score for this algorithm
			mean_auc = round(np.mean(metamodel_scores),4)
			
			print("{:25}\t{:<15}".format("Stacking model", mean_auc))
			results_file.write("{:25}\t{:<15}\n".format("Stacking model", mean_auc))


def q2_stacking_v2():
	# This stacking model is trained over all datasets
	datasets_label = ["citeseer", "cora", "pubmed"]
	datasets_classic = ["strike", "karate", "polblogs", "polbooks", "football"]
	datasets_all = datasets_classic + datasets_label

	results_dir, _ = create_output_directories("q2/stackingv2")

	algos = [resource_allocation_index, jaccard_coefficient, 
		preferential_attachment, cn_soundarajan_hopcroft, ra_index_soundarajan_hopcroft, 
		within_inter_cluster]

	X_meta_train = np.ndarray(shape=(0, len(algos)))
	y_meta_train = np.array([])	

	for dataset in datasets_all:
		if dataset in datasets_label:
			G, _ = gcn.load_data_fully_labeled(dataset)
		else:
			G, _, _ = load_real_classic(dataset)

		# Convert to a simple, undirected graph
		G = nx.Graph(G)

		print(f"\n{dataset} N={len(G.nodes)}, E={len(G.edges)}\n")

		print("{:25}\t{:15}".format("Algorithm", "Mean AUC"))
		
		# We want to remove 20% of edges
		num_edges = len(G.edges)
		num_edges_to_remove = num_edges * 20 // 100

		factor = 2

		dataset_X_meta_train = np.ndarray(shape=(0, len(algos)))
		dataset_y_meta_train = np.array([])

		for _ in range(2):

			G_ = G.copy()
			# Generate the edges to predict. We need to predict the same edges for each algo 
			# so the scores given by each algo are related to each other
			ebunch, y_true = remove_edges_and_generate_ebunch(G_, num_edges_to_remove, factor)

			# To store this round's samples
			# We initialize with 0 columns and will stack the scores 
			round_X_train = np.ndarray(shape=(len(ebunch), 0))

			for algo in algos:

				# Some algorithms sometimes fail. Ensure each of them completed
				algo_successful = False
				while not algo_successful:

					# These algorithms consider the community of nodes when predicting links
					if algo in [cn_soundarajan_hopcroft, ra_index_soundarajan_hopcroft, within_inter_cluster]:
						y_score = list(algo(G_, ebunch, community='value'))
					else:
						try:
							y_score = list(algo(G_, ebunch))
						except KeyError as e:
							# The common_neighbor_centrality algo fails sometimes while computing 
							# shortest paths if G got disconnected after node removal
							continue
						except ZeroDivisionError as e:
							# I'm not sure why but degree of w in nx.common_neighbors(u,v) seems to return < 2 for some nodes in 
							# a common neighborhood even though we convert to undirected simple graph, let's ignore and retry
							continue


					y_score = [score[2] for score in y_score]
					algo_successful = True

					round_X_train = np.column_stack((round_X_train, y_score))


			# Stack these predictions as new samples
			dataset_X_meta_train = np.row_stack((dataset_X_meta_train, round_X_train))
			# Append truth values to our list of truths
			dataset_y_meta_train = np.append(dataset_y_meta_train, y_true)

			print(dataset_X_meta_train.shape)
			print(dataset_y_meta_train.shape)

		# Stack these predictions as new samples
		X_meta_train = np.row_stack((X_meta_train, dataset_X_meta_train))
		# Append truth values to our list of truths
		y_meta_train = np.append(y_meta_train, dataset_y_meta_train)

		print(X_meta_train.shape)
		print(y_meta_train.shape)


	# Train the Meta model
	metamodel = LogisticRegression().fit(X_meta_train, y_meta_train)

	metamodel_scores = []

	for dataset in datasets_all:
		if dataset in datasets_label:
			G, _ = gcn.load_data_fully_labeled(dataset)
		else:
			G, _, _ = load_real_classic(dataset)

		# Convert to a simple, undirected graph
		G = nx.Graph(G)

		with open(f"{results_dir}/{dataset}.txt", "w") as results_file:
			print(f"\n{dataset} N={len(G.nodes)}, E={len(G.edges)}\n")

			print("{:25}\t{:15}".format("Algorithm", "Mean AUC"))
			results_file.write("{:25}\t{:15}\n".format("Algorithm", "Mean AUC"))

			# We want to remove 20% of edges
			num_edges = len(G.edges)
			num_edges_to_remove = num_edges * 20 // 100

			factor = 2

			dataset_X_meta_train = np.ndarray(shape=(0, len(algos)))
			dataset_y_meta_train = np.array([])

			for _ in range(10):

				G_ = G.copy()
				ebunch, meta_y_test = remove_edges_and_generate_ebunch(G_, num_edges_to_remove, factor)

				meta_X_test = np.ndarray(shape=(len(ebunch), 0))

				for algo in algos:
					algo_successful = False
					while not algo_successful:

						# These algorithms consider the community of nodes when predicting links
						if algo in [cn_soundarajan_hopcroft, ra_index_soundarajan_hopcroft, within_inter_cluster]:
							y_score = list(algo(G_, ebunch, community='value'))
						else:
							try:
								y_score = list(algo(G_, ebunch))
							except KeyError as e:
								# The common_neighbor_centrality algo fails sometimes while computing 
								# shortest paths if G got disconnected after node removal
								continue
							except ZeroDivisionError as e:
								# I'm not sure why but degree of w in nx.common_neighbors(u,v) seems to return < 2 for some nodes in 
								# a common neighborhood even though we convert to undirected simple graph, let's ignore and retry
								continue


						y_score = [score[2] for score in y_score]
						algo_successful = True

						meta_X_test = np.column_stack((meta_X_test, y_score))	


				meta_y_pred = metamodel.predict(meta_X_test)

				auc = roc_auc_score(meta_y_test, meta_y_pred)

				metamodel_scores.append(auc)

			# Compute the mean score for this algorithm
			mean_auc = round(np.mean(metamodel_scores),4)
			
			print("{:25}\t{:<15}".format("Stacking model v2", mean_auc))
			results_file.write("{:25}\t{:<15}\n".format("Stacking model v2", mean_auc))




if __name__=='__main__':
	# q1_real_classic()
	q1_real_classic_stacking()

	# q1_real_label_old()
	q1_real_label_stacking()

	# q2_combo()
	# q2_stacking()
	# q2_stacking_v2()
	

