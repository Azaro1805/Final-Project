import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import random

gr2 = nx.gnp_random_graph(10, 0.4, seed=354, directed=False)
nx.draw(gr2, with_labels=True)
pos=nx.spring_layout(gr2)


labels = {}
for x in range(len(gr2)):
    labels[x] = random.choice(['A', 'B', 'C'])

print("Labels :", labels)

neighbors = gr2.edges()
print ( "the neighbors egdes: ", neighbors)

nx.draw_networkx_labels(gr2,pos,labels,font_size=5)
plt.show()