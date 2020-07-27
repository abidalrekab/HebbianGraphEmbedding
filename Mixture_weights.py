
def Mixture_weights(d_graph, node):
    #print(d_graph)
    retrun_prob = []
    n_neighbors = d_graph[node]['neighbors']
    prob = d_graph[node]['probabilities']
    for index, neighbor in enumerate(n_neighbors):
        #print(neighbor)
        retrun_prob.append(prob[neighbor][index])
    return retrun_prob, n_neighbors
