from flask import Flask, request, jsonify

app = Flask(__name__)

class PriorityQueue:
    def __init__(self):
        self.pq = []    # array of node names
        self.qp = {}    # inverse of pq, qp[pq[i]] = pq[qp[i]] = i
        self.keys = {}  # array of element keys/priorities
        self.size = 0   # current size of the priority queue

    def is_empty(self):
        return self.size == 0

    def extract_min(self):
        if self.size == 0:
            raise Exception('Heap is empty!')
        min_name = self.pq[0]
        self.swap(0, self.size - 1)
        self.pq.pop()
        del self.qp[min_name]
        del self.keys[min_name]
        self.size -= 1
        self.sink(0)
        return min_name

    def insert(self, name, key):
        self.size += 1
        self.qp[name] = self.size - 1
        self.keys[name] = key
        self.pq.append(name)
        self.swim(self.size - 1)

    def decrease_key(self, name, new_key):
        if name in self.qp:
            self.keys[name] = new_key
            self.sink(self.qp[name])
            self.swim(self.qp[name])
        else:
            raise Exception('No such element exists!')

    def swap(self, i, j):
        self.pq[i], self.pq[j] = self.pq[j], self.pq[i]
        self.qp[self.pq[i]] = i
        self.qp[self.pq[j]] = j

    def swim(self, k):
        while k > 0 and self.keys[self.pq[k]] < self.keys[self.pq[(k - 1) // 2]]:
            self.swap(k, (k - 1) // 2)
            k = (k - 1) // 2

    def sink(self, k):
        while 2 * k + 1 < self.size:
            j = 2 * k + 1
            if j < self.size - 1 and self.keys[self.pq[j]] > self.keys[self.pq[j + 1]]:
                j += 1
            if self.keys[self.pq[k]] <= self.keys[self.pq[j]]:
                break
            self.swap(k, j)
            k = j
    def contains(self, name):
            return name in self.qp
        
import scipy.io #import mathlab tabs
graph6 = scipy.io.loadmat('./graph6.mat')

import numpy as np


# Extract the 'graph6' variable from the loaded data
graph_data6 = graph6['graph6']

# Convert the extracted data into a NumPy array
array6 = np.array(graph_data6, dtype=int)

# Get the number of nodes in the graph
num_nodes6 = len(array6)

# Initialize an empty adjacency matrix
adjacency_matrix6 = np.zeros((num_nodes6, num_nodes6), dtype=int)


# Fill in the adjacency matrix based on the array6
for i in range(num_nodes6):
    for j in range(num_nodes6):
        if array6[i, j] != 0:
            adjacency_matrix6[i, j] = array6[i, j]

def dijkstra(adj_matrix6, origin):
    num_nodes1 = len(adj_matrix6)
    dist = [float('inf')] * num_nodes6
    prev = [None] * num_nodes1
    dist[origin] = 0

    pq = PriorityQueue()
    pq.insert(origin, 0)

    while not pq.is_empty():
        u = pq.extract_min()
        for v in range(num_nodes6):
            if adj_matrix6[u][v] > 0:
                alt = dist[u] + adj_matrix6[u][v]
                if alt < dist[v]:
                    dist[v] = alt
                    prev[v] = u
                    if pq.contains(v):
                        pq.decrease_key(v, alt)
                    else:
                        pq.insert(v, alt)

    return dist, prev


def main():
    # Given adjacency matrix
    adj_matrix6 = np.array(graph_data6, dtype=int)

    # Define the origin node (e.g., 0 for the first node)
    origin = 0

    # Run Dijkstra's algorithm
    dist, prev = dijkstra(adj_matrix6, origin)

    # Print the results
    print("Shortest distances:")
    for node, distance in enumerate(dist):
        print(f"Node {origin} to Node {node}: {distance}")

    # Print the array containing the previous node on the shortest path for each node
    print("Previous nodes on shortest path:")
    for node, prev_node in enumerate(prev):
        if prev_node is not None:
            print(f"Node {node}: {prev_node + 1}")
        if prev_node is  None:
            print(f"Node {node}: {0}")# Adjust node numbering here

if __name__ == "__main__":
      main()

@app.route('/shortest_path', methods=['GET'])
def shortest_path():
    origin = int(request.args.get('origin'))
    destination = int(request.args.get('destination'))
    
    # Run Dijkstra's algorithm
    dist, prev = dijkstra(adj_matrix6, origin)
    
    # Retrieve the shortest path from the 'prev' array
    path = []
    while destination is not None:
        path.insert(0, destination)
        destination = prev[destination]
    
    return jsonify({"shortest_path": path, "shortest_distance": dist[destination]})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8501)  
