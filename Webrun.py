from flask import Flask, request, jsonify
from HW4_mydijkastra2 import dijkstra_algorithm_function

app = Flask(__name__)

@app.route('/shortest-path', methods=['GET'])
def shortest_path():
    origin = int(request.args.get('origin'))
    destination = int(request.args.get('destination'))


    dist, prev = dijkstra_algorithm_function(adj_matrix6, origin)

    shortest_path = []
    while destination is not None:
        shortest_path.append(destination)
        destination = prev[destination]

    shortest_path = shortest_path[::-1]  

    return jsonify({"shortest_path": shortest_path})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8501)
