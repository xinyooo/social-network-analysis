import pandas as pd
import numpy as np
import networkx as nx
import itertools
import random
from node2vec import Node2Vec
from node2vec.edges import HadamardEmbedder, WeightedL1Embedder, WeightedL2Embedder
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def graph_embedding(train_data):
    # GE using n2v
    train_G = nx.from_pandas_edgelist(train_data, 'node1', 'node2')
    n2v = Node2Vec(train_G, dimensions=32, walk_length=150, num_walks=250, workers=4)
    model = n2v.fit(window=10, min_count=1, batch_words=4)
    edges_embs = HadamardEmbedder(keyed_vectors=model.wv)
    return edges_embs


def get_all_proximity_score(G, edges):
    proximity_score_list = [[] for i in itertools.repeat(None, len(edges))]
    cc = [nx.square_clustering(G, edge[0]) + nx.square_clustering(G, edge[1]) for edge in edges]
    cn = [len(list(nx.common_neighbors(G, edge[0], edge[1]))) for edge in edges]
    jc = nx.jaccard_coefficient(G, edges)
    pa = nx.preferential_attachment(G, edges)
    rai = nx.resource_allocation_index(G, edges)
    for i, data in enumerate(cc):
        proximity_score_list[i].append(data)
    for i, data in enumerate(cn):
        proximity_score_list[i].append(data)
    for i, data in enumerate(jc):
        proximity_score_list[i].append(data[2])
    for i, data in enumerate(pa):
        proximity_score_list[i].append(data[2])
    for i, data in enumerate(rai):
        proximity_score_list[i].append(data[2])
    return proximity_score_list


def data_preprocessing(train_data, sample_time, seed, edges_embs):
    # generate data
    train_G_pos = nx.from_pandas_edgelist(train_data, 'node1', 'node2')
    train_G_node = list(train_G_pos.nodes())
    train_G_neg = nx.Graph()
    train_G_neg.add_nodes_from(train_G_node)
    train_G_neg.add_edges_from(itertools.combinations(train_G_node, 2))
    train_G_neg.remove_edges_from(train_G_pos.edges())
    random.seed(seed)

    pos_edges = [(data.node1, data.node2) for data in train_data.itertuples()]
    neg_edges = random.sample(train_G_neg.edges, int(train_data.shape[0] * sample_time))

    pos_score = get_all_proximity_score(train_G_pos, pos_edges)
    neg_score = get_all_proximity_score(train_G_pos, neg_edges)

    X = []
    Y = []
    for i, data in enumerate(pos_edges):
        x_list = pos_score[i]
        x_list += edges_embs[(str(data[0]), str(data[1]))].tolist()
        X.append(x_list)
        Y.append(1)
    for i, data in enumerate(neg_edges):
        x_list = neg_score[i]
        x_list += edges_embs[(str(data[0]), str(data[1]))].tolist()
        X.append(x_list)
        Y.append(0)

    X = np.array(X)
    Y = np.array(Y)
    return train_test_split(X, Y, test_size=0.2, random_state=seed)


def get_accuracy(prediction, label):
    return sum(1 for x, y in zip(prediction, label) if x == y) / len(label)


if __name__ == '__main__':
    seed = 1234
    sample_time = 0.9
    # load dataset
    train_data = pd.read_csv('./facebook_friendship.txt', names=['node1', 'node2'], header=0, sep=' ')
    # node2vec
    edges_embs = graph_embedding(train_data)
    # prepare data to learning
    print('Generating training data...')
    train_X, test_X, train_Y, test_Y = data_preprocessing(train_data, sample_time, seed, edges_embs)
    # ML part
    print('Learning...')
    # clf = LogisticRegression(random_state=seed, solver='lbfgs')
    # clf = svm.SVC(kernel='rbf')
    clf = RandomForestClassifier(n_estimators=400)
    clf.fit(train_X, train_Y)
    # testing part
    print('Predicting...')
    predict_Y = clf.predict(test_X)
    print(get_accuracy(predict_Y, test_Y))
