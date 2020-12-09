import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import random
import numpy as np
import copy
from collections import defaultdict

random.seed(364)
'''get two arrays and return the winner name'''
def getWinner (winnerVotes,typeOfVotes):
    winners2 =""
    maxVotes =0
    for i in range (len(winnerVotes)):
        if(maxVotes<winnerVotes[i]):
            maxVotes = winnerVotes[i]
    for i2 in range (len(winnerVotes)):
        if( winnerVotes[i2] == maxVotes):
            winners2=winners2 + typeOfVotes[i2]
    return  winners2

'''get a friend and remove him from set'''
def getValue(a):
    for b in a:
        x=b
        a.remove(b)
        return x

def probMatrix(numberOfCom,probs,MinFriendsIn,MaxFriendsIn):
    for a in range (numberOfCom):
        for b in range(numberOfCom):
            if ( probs[a][b] == 0):
                if(a == b):
                    probs[a][b]= round(random.uniform(MinFriendsIn,MaxFriendsIn),2)
                else:
                    probs[a][b] = round(random.uniform(MinFriendsOut, MaxFriendsOut),2)
                    probs[b][a] = probs[a][b]

def creatOpinions(BlockGraph, typeOfVotes, Opinions2, winnerVotes):
    for x in range(len(BlockGraph)):
        Opinions2[x] = random.choice(typeOfVotes)
        Countvotes(typeOfVotes, winnerVotes, Opinions2,x)
    Opinions = copy.deepcopy(Opinions2)
    return Opinions

def createCom (numberOfCom,MinPeople, MaxPeople):
    for i in range(numberOfCom):
        sizes[i] = random.randint(MinPeople, MaxPeople)

def Countvotes(typeOfVotes, winnerVotes, Opinions2,x):

    for i3 in range(len(typeOfVotes)):
        if (Opinions2[x] == typeOfVotes[i3]):
            winnerVotes[i3] = (winnerVotes[i3] + 1)

def setFridends(edges, friends):
    '''pair=(1,2),friend[1]={2}'''
    for d in range(len(edges)):
        pair = list(edges.keys())[d]
        friends[pair[0]].add(pair[1])
        friends[pair[1]].add(pair[0])
    for e in range(len(friends)):
        friends[e].remove(-1)

def runInit(friends, votes, votes2):
    friends2 = copy.deepcopy(friends)
    CleanDouble(votes)
    CleanDouble(votes2)
    return friends2

def countMyOpinion(typeOfVotes, Opinions, votes, counter,f):

    for h1 in range(len(typeOfVotes)):
        if (Opinions[f] == typeOfVotes[h1]):
            votes[f][h1] = (votes[f][h1] + 2)
            counter = counter + 2
            return counter

def countFriendsOpinion(typeOfVotes, Opinions, votes, counter,f):
    if (len(friends2[f]) == 0 ):
        return counter
    set = friends2[f]
    for g in range(len(set)):
        friendOpi = Opinions[getValue(set)]
        for h in range(len(typeOfVotes)):
            if (friendOpi == typeOfVotes[h]):
                votes[f][h] = (votes[f][h] + 1)
                counter = counter + 1
    return counter

def percentOfVotes(counter,typeOfVotes,votes2,threshold,Opinions,f,change):
    if (counter>0):
        for h2 in range(len(typeOfVotes)):
            allVote=votes[f][h2]
            votes2[f][h2]=round(allVote/counter,2)
            '''need to change opinion if A != B , percent > threshold'''
            if (votes2[f][h2] >= threshold and typeOfVotes[h2] != Opinions[f]):
                change = changeOpinion(votes2, threshold, typeOfVotes, Opinions, f, h2, change)
    return change

def changeOpinion(votes2,threshold,typeOfVotes,Opinions,f,h2,change):
       change = True
       print("new opinion:", typeOfVotes[h2],"   old opinion:", Opinions[f], "  friend:",f, "   precent:",votes2[f][h2] )
       Opinions[f]= typeOfVotes[h2]
       return change

def CleanDouble(array):
    for i in range (len(array)):
        for j in range(len(array[0])):
            array[i][j]=0

def Clean(array):
    for i in range (len(array)):
        array[i] = 0

def CreatePlotGraph (changeVar, TotalIter , Xlegend , Ylabel):
    dataPlot = plt.plot(changeVar, TotalIter, label='linear')
    plt.xlabel(Xlegend)
    plt.ylabel(Ylabel)
    plt.legend()
    plt.show()

def CreatescatterGraph (changeVar, WinnerGraph , Xlegend , Ylabel):
    plt.scatter(changeVar,WinnerGraph,label = ("The Start Winner is: "+ winnerStart) )
    plt.xlabel(Xlegend)
    plt.ylabel(Ylabel)
    plt.legend()
    plt.show()

def prints (typeOfVotes, winnerVotes, winnerStart, Opinions2, edges):
    print("Start votes")
    print(np.matrix(typeOfVotes))
    print(np.matrix(winnerVotes))
    print("Start winner is:",winnerStart)
    print(" opinions:")
    print (Opinions2 )
    print("edges :")
    print(edges)

def getValue(a):
    for b in a:
        x=b
        a.remove(b)
        return  x
max_of_iter= 6
numberOfCom = 10
MinPeople = 20
MaxPeople = 30
sizes = [ 0 for i in range(numberOfCom) ]
probs = [ [ 0 for i in range(numberOfCom) ] for j in range(numberOfCom) ]
MinFriendsIn = 0.4
MaxFriendsIn = 0.6
MinFriendsOut = 0.01
MaxFriendsOut = 0.1
threshold = 0
typeOfVotes=["A","B","C"]
winnerStart=''
winnerFinal=''
xLengthGraph=10
Xlegend = "Threshold"
TotalIter = [0 for a3 in range (xLengthGraph)]
changeVar = [0 for a4 in range (xLengthGraph)]
WinnerGraph = [ 0 for i in range(xLengthGraph) ]
winnerVotes= [0 for i2 in range(len(typeOfVotes))]
Opinions2={}

createCom(numberOfCom,MinPeople, MaxPeople)
probMatrix(numberOfCom,probs,MinFriendsIn,MaxFriendsIn)

print("size of Coum :")
print(sizes)

print("probs :")
print(np.matrix(probs))

BlockGraph = nx.stochastic_block_model(sizes, probs, seed=364)
edges = nx.edges(BlockGraph)
friends= [  {-1} for j in range(len(BlockGraph) )]

Opinions=creatOpinions(BlockGraph, typeOfVotes, Opinions2, winnerVotes)
winnerStart = getWinner (winnerVotes,typeOfVotes)
votes = [[0 for i in range(len(typeOfVotes))] for j in range(len(friends))]
votes2 = [[0 for i in range(len(typeOfVotes))] for j in range(len(friends))]

prints (typeOfVotes, winnerVotes, winnerStart, Opinions2, edges)

setFridends(edges, friends)

print("friends :")
print(friends)

for a1 in range (xLengthGraph):
    threshold = 0.35 + a1/20
    changeVar[a1] = threshold
    Opinions = copy.deepcopy(Opinions2)
    print()
    print("The Round : " , a1+1 , "the" , Xlegend , "is : " , threshold )
    change = True
    numOfIteration=0
    while(change):
        if(max_of_iter==numOfIteration):
            print("break while")
            break
        friends2 = runInit(friends, votes, votes2)
        numOfIteration = numOfIteration + 1
        print("numOfIteration is : ", numOfIteration)
        change = False

        for f in range(len(friends)):
            counter=0
            counter = countMyOpinion(typeOfVotes, Opinions, votes, counter, f)
            counter = countFriendsOpinion(typeOfVotes, Opinions, votes, counter, f)
            change = percentOfVotes(counter, typeOfVotes, votes2, threshold, Opinions, f, change)

    Clean(winnerVotes)
    for x1 in range(len(BlockGraph)):
        Countvotes(typeOfVotes, winnerVotes, Opinions, x1)

    WinnerGraph[a1] = getWinner(winnerVotes, typeOfVotes)

    print("final votes")
    print(np.matrix(typeOfVotes))
    print(np.matrix(winnerVotes))
    print("Final Winner is:",WinnerGraph[a1])

    TotalIter[a1] = numOfIteration

print()
print("End  : ")
print(np.matrix(changeVar))
print(np.matrix(TotalIter))
# Threshold plot
Ylabel = "Number of Iterations"
CreatePlotGraph (changeVar, TotalIter , Xlegend , Ylabel)

# Winner Plot
#Ylabel = " Final Winner is :"
#CreatescatterGraph (changeVar, WinnerGraph , Xlegend , Ylabel)