from graphs.graph_erdos_renyi import ErdosRenyiGraph

graph = ErdosRenyiGraph(num_nodes=15, nonlinear=True)
graph.set_target("8")

print(graph.misspecify_graph())
