from collections import defaultdict, Counter, OrderedDict
import matplotlib.pyplot as plt
from networkx.utils import cuthill_mckee_ordering
import numpy as np
import networkx as nx
from joblib import Parallel, delayed
from tqdm import tqdm
from parallel import parallel_generate_walks

"""
        Initiates the generates the walks.

        :param graph: Input graph
        :param dimensions: Embedding dimensions (default: 128)
        :param walk_length: Number of nodes in each walk (default: 80)
        :param num_walks: Number of walks per node (default: 10)
        :param p: Return hyper parameter (default: 1)
        :param q: Inout parameter (default: 1)
        :param weight_key: On weighted graphs, this is the key for the weight attribute (default: 'weight')
        :param workers: Number of workers for parallel execution (default: 1)
        :param sampling_strategy: Node specific sampling strategies, supports setting node specific 'q', 'p', 'num_walks' and 'walk_length'.
        Use these keys exactly. If not set, will use the global ones which were passed on the object initialization
        :param temp_folder: Path to folder with enough space to hold the memory map of self.d_graph (for big graphs); to be passed joblib.Parallel.temp_folder
"""


def _precompute_probabilities(graph):
    """
    Precomputes transition probabilities for each node.
    """
    d_graph = defaultdict(dict)
    quiet = 'False'
    PROBABILITIES_KEY = 'probabilities'
    sampling_strategy = {}
    P_KEY = 'p'
    Q_KEY = 'q'
    p = 1
    q = 0.9
    weight_key = 'weight'
    NEIGHBORS_KEY = 'neighbors'
    FIRST_TRAVEL_KEY = 'first_travel_key'

    nodes_generator = graph.nodes() if quiet \
        else tqdm(graph.nodes(), desc='Computing transition probabilities')

    for source in nodes_generator:

        # Init probabilities dict for first travel
        if PROBABILITIES_KEY not in d_graph[source]:
            d_graph[source][PROBABILITIES_KEY] = dict()

        for current_node in graph.neighbors(source):

            # Init probabilities dict
            if PROBABILITIES_KEY not in d_graph[current_node]:
                d_graph[current_node][PROBABILITIES_KEY] = dict()

            unnormalized_weights = list()
            d_neighbors = list()

            # Calculate unnormalized weights
            for destination in graph.neighbors(current_node):

                p = sampling_strategy[current_node].get(P_KEY,
                                                             p) if current_node in sampling_strategy else p
                q = sampling_strategy[current_node].get(Q_KEY,
                                                             q) if current_node in sampling_strategy else q

                if destination == source:  # Backwards probability
                    ss_weight = graph[current_node][destination].get(weight_key, 1) * 1 / p
                elif destination in graph[source]:  # If the neighbor is connected to the source
                    ss_weight = graph[current_node][destination].get(weight_key, 1)
                else:
                    ss_weight = graph[current_node][destination].get(weight_key, 1) * 1 / q

                # Assign the unnormalized sampling strategy weight, normalize during random walk
                unnormalized_weights.append(ss_weight)
                d_neighbors.append(destination)

            # Normalize
            unnormalized_weights = np.array(unnormalized_weights)
            d_graph[current_node][PROBABILITIES_KEY][
                source] = unnormalized_weights / unnormalized_weights.sum()

            # Save neighbors
            d_graph[current_node][NEIGHBORS_KEY] = d_neighbors

        # Calculate first_travel weights for source
        first_travel_weights = []

        for destination in graph.neighbors(source):
            first_travel_weights.append(graph[source][destination].get(weight_key, 1))

        first_travel_weights = np.array(first_travel_weights)
        d_graph[source][FIRST_TRAVEL_KEY] = first_travel_weights / first_travel_weights.sum()
    return d_graph

def _generate_walks(d_graph, walk_length=30, num_walks=200, workers=4) -> list:
    """
    Generates the random walks which will be used as the skip-gram input.
    :return: List of walks. Each walk is a list of nodes.
    """
    quiet = 'False'
    PROBABILITIES_KEY = 'probabilities'
    sampling_strategy = {}
    NEIGHBORS_KEY = 'neighbors'
    FIRST_TRAVEL_KEY = 'first_travel_key'
    # ---------------------------------------#
    NUM_WALKS_KEY = 'num_walks'
    WALK_LENGTH_KEY = 'walk_length'


    flatten = lambda l: [item for sublist in l for item in sublist]

    # Split num_walks for each worker
    num_walks_lists = np.array_split(range(num_walks), workers)
    temp_folder, require = None, None
    walk_results = Parallel(n_jobs=workers, temp_folder = temp_folder, require= require)(
        delayed(parallel_generate_walks)(d_graph,
                                         walk_length,
                                         len(num_walks),
                                         idx,
                                         sampling_strategy,
                                         NUM_WALKS_KEY,
                                         WALK_LENGTH_KEY,
                                         NEIGHBORS_KEY,
                                         PROBABILITIES_KEY,
                                         FIRST_TRAVEL_KEY,
                                         quiet) for
        idx, num_walks
        in enumerate(num_walks_lists, 1))

    walks = flatten(walk_results)

    return walks

def CountFrequency(my_list, freq):
    # Creating an empty dictionary
    for item in my_list:
        if (item in freq):
            freq[item] += 1
        else:
            freq[item] = 1
    return freq

def GraphDegree(G):
    degree_sequence = np.asarray([d for n, d in G.degree()])  # degree sequence
    node_sequence =   np.asarray([n for n, d in G.degree()])  # degree sequence
    #print("Degree sequence", degree_sequence)
    #print("node sequence", node_sequence)
    fig, ax = plt.subplots()
    plt.bar(node_sequence, degree_sequence, width=0.80, color='b')
    plt.title("Degrees Histogram")
    plt.ylabel("degrees")
    plt.xlabel("nodes")
    plt.show()

def RandomWalkHist(freq):
    od = OrderedDict(sorted(freq.items()))
    list_key_value = [[k, v] for k, v in od.items()]

    frequency_count = np.asarray([d for n, d in list_key_value])
    node_sequence = np.asarray([n for n, d in list_key_value])
    #print("Degree sequence", frequency_count)
    #print("node sequence", node_sequence)
    fig, ax = plt.subplots()
    plt.bar(node_sequence, frequency_count, width=0.80, color='b')
    plt.title("Frequency Histogram")
    plt.ylabel("Frequnecy")
    plt.xlabel("nodes")
    plt.show()


