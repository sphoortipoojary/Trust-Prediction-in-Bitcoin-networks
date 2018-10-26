from __future__ import division
import networkx as nx
import math
import sys
import sklearn
from sklearn.metrics import mean_squared_error
import operator
import matplotlib as plt
from sklearn import linear_model
import numpy as np


"Generating Directed Graph using NetworkX"
G = nx.DiGraph()

file_name = open("soc-sign-bitcoinalpha.csv", "r")

for row in file_name:
    n = row.strip().split(",")
    G.add_edge(n[0], n[1], weight=float(n[2]) /10.0)    #dividing by 10 to scale data from -1 to 1
file_name.close()

"Fairness Goodness Algorithm"


def fairness_goodness(G):
    nodes = G.nodes()
    fairness = {}
    goodness = {}

    for n in nodes:
        fairness[n] = 1
        try:
            indeg = float(G.in_degree(n))
            indeg_weight = float(G.in_degree(n, weight='weight'))
            goodness[n] = indeg_weight / indeg

        except:
            goodness[n] = 0

    error = 0.0001
    converge = False
    nodes = G.nodes()
    epoch = 0

    while converge == False:

        delta_f = 0
        delta_g = 0

        print('initialization')

        for n in nodes:
            inedges = G.in_edges(n, data='weight')
            g = 0.0
            for edge in inedges:
                g = g + (fairness[edge[0]] * edge[2])

            try:
                m = len(inedges)
                delta_g = delta_g + abs(g / (m - goodness[n]))
                goodness[n] = g / m

            except:
                pass

        for n in nodes:
            outedges = G.out_edges(n, data='weight')
            f = 0.0
            for edge in outedges:
                f = f + (1.0 - abs(edge[2] - goodness[edge[1]]) / 2.0)
            try:
                m = len(outedges)
                delta_f = delta_f + abs((f / m) - fairness[n])
                fairness[n] = f / m

            except:
                pass

        print("change in fairness = ", delta_f, " and change in goodness = ",  delta_g)
        if delta_f < error:
            converge = True
        else:
            converge = False

        epoch = epoch + 1

    print ("epochs = ", epoch)

    return fairness, goodness


fairness, goodness = fairness_goodness(G)

print("fairness =", fairness,  " goodness =", goodness)

"Calculating edge-weights  by mutiplying fairness of source and goodness of target node"

edges_graph = list(G.edges(data='weight'))
# print("type= ", type(edges_graph))
pred_edge_weight = []
y_true = []
for e in edges_graph:
    pred_edge_weight.append(fairness[e[0]] * goodness[e[1]])
    y_true.append(e[2])
print("computed_edge_weight =", pred_edge_weight)

fairness_list = []
goodness_list = []

f_train = fairness_list
# print ("len f", len(f_train))
# print ("total edges ", len(edges_graph))
g_train = goodness_list
X_list = []
for e in edges_graph:
    X_list.append([fairness[e[0]], goodness[e[1]]])
# print ("new len y_true", len(y_true))

"Function to Predict Edge Weight using Linear Regression using Fairness and Goodness as Feature Vectors"


def prediction_FGA ():

    y_pred_list = []

    for index in range(len(y_true)):
        # print ("index", index)

        X_train_2 = np.array(X_list).reshape(len(X_list), 2)
        X_test = np.array(X_list[index]).reshape(1,2)
        X_train = X_train_2.copy()

        y_train_2 = np.array(y_true)
        y_train = y_train_2.copy()

        X_train = np.delete(X_train, index, axis=0)
        y_train = np.delete(y_train, index, axis=0)

        regr = linear_model.LinearRegression()

        regr.fit(X_train, y_train)
        y_pred = regr.predict(X_test)
        y_pred_list.append(y_pred) #appending each predicted value in a list

    return y_pred_list


y_pred_FGA = prediction_FGA()
print ()
y_pred_FGA = np.array(y_pred_FGA).reshape(len(y_true), 1)

y_true_numpy = np.array(y_true).reshape(len(y_true),1)

"Calculating RMSE of predicted value and true edge weights"
rmse_FGA_prediction = mean_squared_error(y_true_numpy, y_pred_FGA)

print ("RMSE FGA prediction", rmse_FGA_prediction)


with open('soc-sign-bitcoinalpha.csv', 'r') as file_name:
    lines = [[i.strip() for i in line.strip().split(',')]
             for line in file_name.readlines()]


new_lines = [line + [str(pred_edge_weight[i])] for i, line in enumerate(lines)]

with open('goodness_fairness_weights.csv', 'w') as fo:
    for line in new_lines:
        fo.write(','.join(line) + '\n')

# rmse = mean_squared_error(y_true, pred_edge_weight)
# print("rmse = ", rmse)
G_weighted = G
weighted_pr = nx.pagerank(G, alpha=0.9, weight= True)
# print ("weighted_pr =", weighted_pr)

"Calculating RMSE of predicted weights using PageRank"
pr_pred_edge_weight = []
for e in edges_graph:
    pr_pred_edge_weight.append(weighted_pr[e[0]] - weighted_pr[e[1]])
# print("edge weight with page rank =", pr_pred_edge_weight)

rmse_pr = mean_squared_error(y_true, pr_pred_edge_weight)
print ("RMSE for edge weights using PageRank for prediction =", rmse_pr)

G_undirected = nx.Graph()

file_name = open("soc-sign-bitcoinalpha.csv", "r")
for nodes in file_name:
    n = nodes.strip().split(",")
    G_undirected.add_edge(n[0], n[1], weight= float(n[2])/10.0)
file_name.close()

"Generating Network Characteristics"
numbr_cc = nx.number_connected_components(G_undirected)
print ("number of connected components =", numbr_cc)

weakly_connected =nx.number_weakly_connected_components(G)
strongly_connected =nx.number_strongly_connected_components(G)
print ("weakly connected components = ", weakly_connected)
print("strongly connected components =", strongly_connected)


clust = nx.clustering(G_undirected)
print ("clust", clust)
sorted_clust = (sorted(clust.items(), key=operator.itemgetter(1), reverse=True))
sorted_clust_list = list(sorted_clust)

k = 5
sorted_nodes = []
for key, val in enumerate(sorted_clust):
    sorted_nodes.append(val[0])
    print(" node = ", val[0], "clustering coeff =", val[1])
    k = k - 1
    if k <= 0:
        break

print ("top 5 clustering nodes =", sorted_nodes)

betweenness_centrality = nx.betweenness_centrality(G)

sorted_betweenness = (sorted(betweenness_centrality.items(), key=operator.itemgetter(1), reverse=True))

k = 5
print ("betweenness centrality of top 5 nodes =")
sorted_nodes_bw = []
for key, val in enumerate(sorted_betweenness):
    sorted_nodes_bw.append(val[0])
    print (" node = ", val[0]," betweenness centrality =", val[1] )
    k = k - 1
    if k <=0 :
        break

print ("top 5 nodes sorted as per betweenness centrality =", sorted_nodes_bw)

