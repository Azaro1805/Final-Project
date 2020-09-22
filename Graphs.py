import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import random

gr1 = nx.Graph()

gr1.add_node(0, pos=(1, 1))
gr1.add_node(1, pos=(1, 2))
gr1.add_node(2, pos=(2, 4))
gr1.add_node(3, pos=(3, 4))
gr1.add_node(4, pos=(4, 5))

'''add edge : gr1.add_edge(0, 1) '''

'''add many edges at once :  gr1.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 1)])'''

'''gr1.add_edge(0, 1, weight=10)
gr1.add_edge(1, 2, weight=10)
gr1.add_edge(2, 3, weight=10)
gr1.add_edge(3, 1, weight=10)

weight = nx.get_edge_attributes(gr1, 'weight')
pos = nx.get_node_attributes(gr1, 'pos')

plt.figure()
nx.draw_networkx(gr1, pos)
plt.show()
'''

'''nx.draw_networkx_edge_labels(gr1, pos, edge_labels=weight)'''
'''add weight to the graph'''

gr2 = nx.gnp_random_graph(10, 0.4, seed=354, directed=False)
nx.draw(gr2, with_labels=True)
pos=nx.spring_layout(gr2)

'''sp = dict(nx.all_pairs_shortest_path(gr2))
sp[0]
print(sp)'''

labels = {}
for x in range(len(gr2)):
    labels[x] = random.choice(['A', 'B', 'C'])
    '''gr2.add_node(random.choice(['A', 'B', 'C']))'''

print("Labels :", labels)

'''o2 = list(gr2.nodes)
print(o2)'''

'''o=list(gr2.neighbors('B'))
print(o)'''

neighbors = gr2.edges()
print ( "the neighbors : ", neighbors)

nx.draw_networkx_labels(gr2,pos,labels,font_size=5)
plt.show()


