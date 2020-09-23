import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import random
import numpy as np

n=2
m=10
numberOfneb =1000
result = [ [ 0 for i in range(m) ] for j in range(n) ]
numOfIterations=0

for iz in range (10):
    print ("iteraion i = ", iz ," % " , (iz+1)/10 )
    result[0][iz]= (iz+1)/10
    gr2 = nx.gnp_random_graph(numberOfneb, (iz+1)/10 , seed=354, directed=False)
    '''nx.draw(gr2, with_labels=True)'''
    pos = nx.spring_layout(gr2)

    labels = {}
    for x in range(len(gr2)):
        labels[x] = random.choice(['A', 'B', 'C'])
    '''print("Labels :", labels)'''

    neighbors = gr2.edges
    '''print("the neighbors: ", neighbors)'''

    m2=3
    votes = [ [ 0 for i in range(m2) ] for j in range(numberOfneb) ]
    votes2 = [ [ 0 for i in range(m2) ] for j in range(numberOfneb) ]
    opinion = ['A', 'B', 'C']
    threshold = 0.55
    numOfIterations=0
    change = True

    while (change):
        change = False
        votes = [ [ 0 for i in range(m) ] for j in range(numberOfneb) ]
        numOfIterations = numOfIterations+1
        for i in range (len(labels)):
            counter = 0
            for j  in range(len(neighbors)):
                res = list(neighbors.keys())[j]
                if(res[0] == i ):
                    for k in range (len(opinion)):
                        if (labels[res[1]] == opinion[k]):
                            votes[i][k] = votes[i][k]+1
                            counter = counter+1
                if(res[1] == i ):
                    for k in range(len(opinion)):
                        if (labels[res[1]] == opinion[k]):
                            votes[i][k] = votes[i][k] + 1
                            counter = counter+1
            for l in range (len(opinion)):
                if(counter > 0):
                    votes2[i][l] = votes[i][l]/counter
                    if(votes2[i][l] >= threshold and labels[i] != opinion[l] ):
                        change = True
                        labels[i] = opinion[l]
                        '''print(votes2)
                        print(" we want to change i : ", i, " labels[i] = ", labels[i], "l", l, "opinion[l] = ", opinion[l]) 
                        labels[i] = opinion[l] 
                        print("labels after change :", labels)
                        print(" ") '''


    print("votes : ", votes)
    print("votes2 : ", votes2)
    print("numOfIterations : ", numOfIterations)

    result[1][iz] = numOfIterations

    '''nx.draw_networkx(gr2)
    plt.show()'''

print ()
print("result = ", result)

# Plot the data
plt.plot(result[0], result[1], label='linear')

# Add a legend
plt.legend()

# Show the plot
plt.show()

