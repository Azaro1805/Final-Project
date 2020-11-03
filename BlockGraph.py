import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import random
import numpy as np
import copy
from collections import defaultdict


def getValue(a):
    for b in a:
        x=b
        a.remove(b)
        return  x

'''set1={"1","2"}
print(getValue(set1))
print(set1)
print(getValue(set1))
print(set1)'''

'''for i,val in enumerate(set1):
    print(i,val)'''


numberOfCom = 10

MinPeople = 20
MaxPeople = 30
sizes = [ 0 for i in range(numberOfCom) ]

probs = [ [ 0 for i in range(numberOfCom) ] for j in range(numberOfCom) ]
MinFriendsIn = 0.4
MaxFriendsIn = 0.5
MinFriendsOut = 0.01
MaxFriendsOut = 0.1
threshold = 0
typeOfVotes=["A","B"]
winnerStart=''
winnerFinal=''
xLengthGraph=10
Xlegend = "threshold"
TotalIter = [0 for a3 in range (xLengthGraph)]
changeVar = [0 for a4 in range (xLengthGraph)]


WinnerGraph = [ 0 for i in range(xLengthGraph) ]


for i in range(numberOfCom):
    sizes[i] = random.randint(MinPeople, MaxPeople)

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

Opinions2={}
winnerVotes= [0 for i2 in range(len(typeOfVotes))]
for x in range(len(BlockGraph)):
    Opinions2[x] = random.choice(typeOfVotes)
    for i3 in range(len(typeOfVotes)):
        if (Opinions2[x] == typeOfVotes[i3]):
            winnerVotes[i3] = (winnerVotes[i3] + 1)
max=0

for i4 in range (len(typeOfVotes)):
    if(winnerVotes[i4]>max):
        max=winnerVotes[i4]
        winnerStart=typeOfVotes[i4]
print("Start votes")
print(np.matrix(typeOfVotes))
print(np.matrix(winnerVotes))
print("Start winner is:",winnerStart)
print(" opinions:")
print (Opinions2 )


print("edges :")
print(edges)

friends= [  {-1} for j in range(len(BlockGraph) )]


for d in range(len(edges)):
    pair=list(edges.keys())[d]
    friends[pair[0]].add(pair[1])
    friends[pair[1]].add(pair[0])

for e in range(len(friends)):
    friends[e].remove(-1)

print("friends :")
print(friends)

for a1 in range (xLengthGraph):
    threshold = 0.55 + a1/20
    changeVar[a1] = threshold
    Opinions = copy.deepcopy(Opinions2)
    print()
    print("The Round : " , a1 , "the" , Xlegend , "is : " , threshold )
    change = True
    numOfIteration=0
    while(change ):
        friends2 = copy.deepcopy(friends)
        votes = [[0 for i in range(len(typeOfVotes))] for j in range(len(friends))]
        votes2 = [[0 for i in range(len(typeOfVotes))] for j in range(len(friends))]

        print("iteration number:", numOfIteration)
        numOfIteration=numOfIteration+1
        change = False
        for f in range(len(friends)):
            counter=0
            if(len(friends2[f])==0 and (len(friends2)-1)!=f):
                f=f+1
            for h1 in range(len(typeOfVotes)):
                if (Opinions[f]== typeOfVotes[h1]):
                    votes[f][h1] = (votes[f][h1] + 2)
                    counter= counter+2
            set=friends2[f]
            for g in range(len(set)):
                friendOpi = Opinions[getValue(set)]
                for h in range(len(typeOfVotes)):
                   if(friendOpi==typeOfVotes[h]):
                       votes[f][h] = (votes[f][h] + 1)
                       counter = counter + 1
            if (counter>0):
                for h2 in range(len(typeOfVotes)):
                    allVote=votes[f][h2]
                    votes2[f][h2]=round(allVote/counter,2)
                    if (votes2[f][h2]>= threshold and typeOfVotes[h2]!=Opinions[f] ):
                       change = True
                       print("new opinion:", typeOfVotes[h2],"   old opinion:", Opinions[f], "  friend:",f, "   precent:",votes2[f][h2] )
                       Opinions[f]= typeOfVotes[h2]
        '''print(np.matrix(votes2))'''

    winnerVotes= [0 for i7 in range(len(typeOfVotes))]
    '''x1=max(winnerVotes)'''

    for y2 in range(len(BlockGraph)):
        for i6 in range(len(typeOfVotes)):
            if (Opinions[y2] == typeOfVotes[i6]):
                winnerVotes[i6] = (winnerVotes[i6] + 1)
    draw = False
    max2=0
    index = [0 for a56 in range (len(typeOfVotes))]
    counter = 0
    for i5 in range (len(winnerVotes)):
        if(winnerVotes[i5]>max2):
            max2=winnerVotes[i5]
            winnerFinal = typeOfVotes[i5]
    for i55 in range(len(winnerVotes)):
        if(winnerVotes[i55]==max2):
            index[counter] = typeOfVotes[i55]
            counter = counter+1
    if (counter > 1):
        winnerFinal = ""
        for i9 in range(len(winnerVotes)):
                winnerFinal = winnerFinal+", "+typeOfVotes[i9]
    WinnerGraph[a1] = winnerFinal
    print("final votes")
    print(np.matrix(typeOfVotes))
    print(np.matrix(winnerVotes))
    print("Final winner is:",winnerFinal)
    '''print(np.matrix(votes))
    print()
    print(np.matrix(votes2))
    print("num of iteration")
    print(numOfIteration)'''
    TotalIter[a1] = numOfIteration

    '''plt.show(nx.draw(BlockGraph , pos = nx.spring_layout(BlockGraph)))'''

print()
print("End  : ")
print(np.matrix(changeVar))
print(np.matrix(TotalIter))

# Threshold plot
dataPlot = plt.plot(changeVar, TotalIter, label='linear')
plt.xlabel(Xlegend)
plt.ylabel("number of iterations")
plt.legend()
plt.show()

# Winner Plot
print(np.matrix(WinnerGraph))
plt.scatter(changeVar,WinnerGraph,label = ("The Start Winner is: "+ winnerStart) )
plt.xlabel("threshold : " )
plt.ylabel(" Final Winner is :")
plt.legend()
plt.show()