import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import random
import numpy as np
from collections import defaultdict

numberOfCom = 2

MinPeople = 4
MaxPeople = 5
sizes = [ 0 for i in range(numberOfCom) ]

probs = [ [ 0 for i in range(numberOfCom) ] for j in range(numberOfCom) ]
MinFriendsIn = 0.6
MaxFriendsIn = 0.9
MinFriendsOut = 0.01
MaxFriendsOut = 0.1

for i in range (numberOfCom):
    sizes[i] = random.randint(MinPeople,MaxPeople)




for a in range (numberOfCom):
    for b in range(numberOfCom):
        if ( probs[a][b] == 0):
            if(a == b):
                probs[a][b]= round(random.uniform(MinFriendsIn,MaxFriendsIn),2)
            else:
                probs[a][b] = round(random.uniform(MinFriendsOut, MaxFriendsOut),2)
                probs[b][a] = probs[a][b]


print("size of Coum :")
print(sizes)

print("probs :")
print(np.matrix(probs))

BlockGraph = nx.stochastic_block_model(sizes, probs, seed=364)
edges = nx.edges(BlockGraph)

Opinions={}
for x in range(len(BlockGraph)):
    Opinions[x] = random.choice(['A', 'B', 'C'])
print(" opinions:")
print (Opinions )

print("edges :")
print(edges)
pair=list(edges.keys())[0]
print(pair[0])

friends= [  {int} for j in range(len(BlockGraph) )]
friends[0].add(1)
print("friends :")
print (friends)
'''for d in range(len(edges)):
    pair=list(edges.keys())[d]
    friends[pair[0]].add(pair[1])
    friends[pair[1]].add(pair[0])

print("friends :")
print(friends)'''


'''plt.show(nx.draw(BlockGraph , pos = nx.spring_layout(BlockGraph)))'''

