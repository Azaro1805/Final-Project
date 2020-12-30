import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import random
import numpy as np
import copy
from collections import defaultdict

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

def creatOpinions(BlockGraph, typeOfVotes, Opinions2, winnerVotes, sizes):
    k=0
    for i in range(len(sizes)):
        MostOpinions = random.choice(typeOfVotes)
        for j in range (sizes[i]):
            randomNum = random.random()
            if(randomNum<=0.5):
                Opinions2[k] = MostOpinions
            else:
                typeOfVotes2 = copy.deepcopy(typeOfVotes)
                typeOfVotes2.remove(MostOpinions)
                Opinions2[k] = random.choice(typeOfVotes2)
            Countvotes(typeOfVotes, winnerVotes, Opinions2, k)
            k = k + 1
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
    return change

def changeOpinion(votes2,threshold,typeOfVotes,Opinions,f,h2,change):
       change = True
       #print("new opinion:", typeOfVotes[h2],"   old opinion:", Opinions[f], "  friend:",f, "   precent:",votes2[f][h2] )
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
    #print(" opinions:")
    #print (Opinions2 )
    #print("edges :")
    #print(edges)

def getValue(a):
    for b in a:
        x=b
        a.remove(b)
        return  x


def numberOfVotes (numberOfVotes):
    if(numberOfVotes == 2):
        typeOfVotes = ["A", "B"]
        return typeOfVotes
    if (numberOfVotes == 3):
        typeOfVotes = ["A", "B" , "C"]
        return typeOfVotes
    if (numberOfVotes == 4):
        typeOfVotes = ["A", "B" , "C", "D"]
        return typeOfVotes
    if (numberOfVotes == 5):
        typeOfVotes = ["A", "B" , "C", "D" , "E"]
        return typeOfVotes
    if (numberOfVotes == 6):
        typeOfVotes = ["A", "B" , "C", "D" , "E" ,"F"]
        return typeOfVotes
def calculate_winner_to_graph(start_winner_graph, end_winner_graph, winner_per_graph ,num_vote ,seedi , a1):
    winnerEnd = getWinner(end_winner_graph, typeOfVotes)
    precente = 0
    if (winnerEnd == "A"):
        precente=end_winner_graph[0]-start_winner_graph[0]
    if (winnerEnd == "B"):
        precente = end_winner_graph[1] - start_winner_graph[1]
    if (winnerEnd == "C"):
        precente = end_winner_graph[2] - start_winner_graph[2]
    if (winnerEnd == "D"):
        precente = end_winner_graph[2] - start_winner_graph[2]
    if (winnerEnd == "E"):
        precente = end_winner_graph[2] - start_winner_graph[2]
    if (winnerEnd == "F"):
        precente = end_winner_graph[2] - start_winner_graph[2]
    precente = abs(precente/num_vote)
    #print(precente)
    winner_per_graph[seedi][a1] = precente

def calculate_winner_change_to_graph(start_change_winner_graph, end_winner_graph, winner_change_graph ,seedi , a1):
    winnerEnd = getWinner(end_winner_graph, typeOfVotes)
    if (winnerEnd == start_change_winner_graph):
        winner_change_graph[seedi][a1] = 0
    else:
        winner_change_graph[seedi][a1] = 1

number_of_seeds = 50
max_of_iter= 10
MinFriendsIn = 0.4
MaxFriendsIn = 0.7
MinFriendsOut = 0.1
MaxFriendsOut = 0.3
threshold = 0.5
winnerStart=''
winnerFinal=''
xLengthGraph= 5  # between 2 - 6 types of votes ! dont change !
Xlegend = "Number of Votes"
TotalIter = [0 for a3 in range (xLengthGraph)]
changeVar = [0 for a4 in range (xLengthGraph)]
numberOfCom = 0
WinnerGraph = [ "" for j in range(xLengthGraph)]
Opinions2 = {}
change = True
numberOfCom = 5
MinPeople = 20
MaxPeople = 25

sizes = [0 for i in range(numberOfCom)]
probs = [[0 for i in range(numberOfCom)] for j in range(numberOfCom)]

GarphIter = [[0 for j in range(xLengthGraph)] for i in range(number_of_seeds)]
GarphIter_avg = [0 for j in range(xLengthGraph)]

winner_per_graph = [[0 for j in range(xLengthGraph)] for i in range(number_of_seeds)]
winner_per_graph_avg = [0 for j in range(xLengthGraph)]

start_change_winner_graph = "T"
winner_change_graph = [[0 for j in range(xLengthGraph)] for i in range(number_of_seeds)]
winner_change_graph_avg = [0 for j in range(xLengthGraph)]

###############################################################################
for seedi in range(number_of_seeds):
    seede= 363+ seedi*140
    random.seed(seede)
    print( "######################### next seed , the seed is ", seede , "#################")
    '''checking on different size of type Of Votes'''

    probMatrix(numberOfCom, probs, MinFriendsIn, MaxFriendsIn)
    for i in range(numberOfCom):
        sizes[i] = random.randint(MinPeople, MaxPeople)
    print("size of Coum :")
    print(sizes)
    #print("probs :")
   # print(np.matrix(probs))

    BlockGraph = nx.stochastic_block_model(sizes, probs, seed=seede)
    edges = nx.edges(BlockGraph)
    friends = [set() for j in range(len(BlockGraph))]

    for a1 in range (xLengthGraph):
        change = True
        numOfIteration=0
        # change votes length

        typeOfVotes = numberOfVotes(a1+2)
        start_winner_graph = [0 for j in range(len(typeOfVotes))]
        print("The Round : ", a1+1, "the", Xlegend, " is ", len(typeOfVotes))
        winnerVotes = [0 for i2 in range(len(typeOfVotes))]
        changeVar[a1] = len(typeOfVotes)

        Opinions = creatOpinions(BlockGraph, typeOfVotes, Opinions2, winnerVotes, sizes)
        votes = [[0 for i in range(len(typeOfVotes))] for j in range(len(friends))]
        votes2 = [[0 for i in range(len(typeOfVotes))] for j in range(len(friends))]
        winnerStart = getWinner(winnerVotes, typeOfVotes)
        start_change_winner_graph = copy.deepcopy(winnerStart)
        start_winner_graph = copy.deepcopy(winnerVotes)
        prints(typeOfVotes, winnerVotes, winnerStart, Opinions2, edges)
        Clean(winnerVotes)

        for x in range(len(BlockGraph)):
            Countvotes(typeOfVotes, winnerVotes, Opinions2, x)

        setFridends(edges, friends)

        #print("friends :")
        #print(friends)

        while(change):
            if (max_of_iter == numOfIteration):
                print("break while")
                break
            friends2 = runInit(friends, votes, votes2)
            #winnerStart = getWinner(winnerVotes, typeOfVotes)
            numOfIteration = numOfIteration + 1
            print( "numOfIteration is : " , numOfIteration)
            change = False

            for f in range(len(friends)):
                counter=0
                counter = countMyOpinion(typeOfVotes, Opinions, votes, counter, f)
                counter = countFriendsOpinion(typeOfVotes, Opinions, votes, counter, f)
                change = percentOfVotes(counter, typeOfVotes, votes2, threshold, Opinions, f, change)

        #print(np.matrix(Opinions))
        Clean(winnerVotes)
        for x1 in range(len(BlockGraph)):
            Countvotes(typeOfVotes, winnerVotes, Opinions, x1)

        print("final votes")
        print(np.matrix(typeOfVotes))
        print(np.matrix(winnerVotes))
        WinnerGraph[a1] = getWinner(winnerVotes,typeOfVotes)
        print("Final winner is:",WinnerGraph[a1])

        '''print(np.matrix(votes))
        print()
        print(np.matrix(votes2))
        print("num of iteration")
        print(numOfIteration)'''
        TotalIter[a1] = numOfIteration

        '''plt.show(nx.draw(BlockGraph , pos = nx.spring_layout(BlockGraph)))'''
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX Next Round XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print()
        calculate_winner_to_graph(start_winner_graph, winnerVotes, winner_per_graph, len(BlockGraph), seedi, a1)
        calculate_winner_change_to_graph(start_change_winner_graph, winnerVotes, winner_change_graph, seedi, a1)
    for i in range(len(TotalIter)):
        GarphIter[seedi][i] = TotalIter[i]
    Clean(winnerVotes)
    print()
    print("End  : ")
    print(np.matrix(changeVar))
    print(np.matrix(TotalIter))

# Number of Iterations VS number of com plot
print("Number of Iterations VS number of opinions plot")
print(np.matrix(GarphIter))
for i in range (number_of_seeds):
    for j in range(len(TotalIter)):
        GarphIter_avg[j] = GarphIter_avg[j] + GarphIter[i][j]

print(np.matrix(GarphIter_avg))

for i in range(len(TotalIter)):
    GarphIter_avg[i] = round(GarphIter_avg[i]/number_of_seeds,3)

print(np.matrix(GarphIter_avg))

Ylabel= "Number of Iterations"
CreatePlotGraph (changeVar, GarphIter_avg , "Number of opinions" , Ylabel)


# Winner Present Votes VS number of com plot
print("Winner Present Votes VS number of com plot")
print(np.matrix(winner_per_graph))

for i in range (number_of_seeds):
    for j in range(len(TotalIter)):
        winner_per_graph_avg[j] = winner_per_graph_avg[j] + winner_per_graph[i][j]
print()
print(np.matrix(winner_per_graph_avg))

for i in range(len(TotalIter)):
    winner_per_graph_avg[i] = round(winner_per_graph_avg[i]/number_of_seeds,3)
print()
print(np.matrix(winner_per_graph_avg))


Ylabel= "Diff Between Start to End Present Winner Votes"
CreatePlotGraph (changeVar, winner_per_graph_avg , "Number of opinions" , Ylabel)

# Winner change VS number of com plot
print("Winner change VS number of com plot")
print(np.matrix(winner_change_graph))

for i in range (number_of_seeds):
    for j in range(len(TotalIter)):
        winner_change_graph_avg[j] = winner_change_graph_avg[j] + winner_change_graph[i][j]

print(np.matrix(GarphIter_avg))

for i in range(len(TotalIter)):
    winner_change_graph_avg[i] = round(winner_change_graph_avg[i]/number_of_seeds,3)

print(np.matrix(winner_change_graph_avg))

Ylabel= "Winner change in %"
CreatePlotGraph (changeVar, winner_change_graph_avg , "Number of opinions" , Ylabel)
