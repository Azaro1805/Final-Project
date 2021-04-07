import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import random
import numpy as np
import copy
from collections import defaultdict


#################
def choose_friends(friends, Opinions ,barabasiGraph , blocked, percent_of_change):
    rand_num = random.randrange(len(barabasiGraph))
    if(rand_num in blocked):
        while(True):
            rand_num = random.randrange(len(barabasiGraph))
            if (rand_num not in blocked):
                break
    #print("friend num = " , rand_num)
    blocked.add(rand_num)
    set_of_friends = friends[rand_num]
    #print("his friends = " , set_of_friends)
    new_set_of_friends = take_p_precent(0.2, set_of_friends, blocked, Opinions, rand_num)
    if(len(new_set_of_friends)==0):
        return
    #print("the friends ", new_set_of_friends)
    add_to_set(new_set_of_friends, blocked)
    #print("blocked after add = " , blocked)
    change_opinions(new_set_of_friends, rand_num,Opinions, percent_of_change)
def add_to_set (set_from, blocked):
    for i in set_from:
        blocked.add(i)
# return friends possible to change
def take_p_precent(precent,set_of_friends, blocked, Opinions, The_friends):
    new_set_of_friends = set()
    counter = 0
    for i in set_of_friends:
        if(i not in blocked):
            if(Opinions[i] != Opinions[The_friends]):
                #print("Opinions["+str(i)+"] =", Opinions[i], " | Friends_opinion = " , Opinions[The_friends] )
                counter += 1
                new_set_of_friends.add(i)
                if(counter/len(set_of_friends) >= precent):
                    break
    return new_set_of_friends
def change_opinions(F_to_change, F_change_from,Opinions, precente_of_change):
    for f in F_to_change:
        rand_num = random.random()
        if(rand_num <= precente_of_change):
            Opinions[f]=Opinions[F_change_from]
            #print(round(rand_num,3),"Opinions[",f,"] = = ", Opinions[f] , "Opinions[F_change_from] = " ,Opinions[F_change_from])
####


def getWinner(winnerVotes, typeOfVotes):
    winners2 = ""
    maxVotes = 0
    for i in range(len(winnerVotes)):
        if (maxVotes < winnerVotes[i]):
            maxVotes = winnerVotes[i]
    for i2 in range(len(winnerVotes)):
        if (winnerVotes[i2] == maxVotes):
            winners2 = winners2 + typeOfVotes[i2]
    return winners2


'''get a friend and remove him from set'''


def getValue(a):
    for b in a:
        x = b
        a.remove(b)
        return x


def probMatrix(numberOfCom, probs, MinFriendsIn, MaxFriendsIn):
    for a in range(numberOfCom):
        for b in range(numberOfCom):
            if (probs[a][b] == 0):
                if (a == b):
                    probs[a][b] = round(random.uniform(MinFriendsIn, MaxFriendsIn), 2)
                else:
                    probs[a][b] = round(random.uniform(MinFriendsOut, MaxFriendsOut), 2)
                    probs[b][a] = probs[a][b]


def creatOpinions(BlockGraph, typeOfVotes, Opinions2, winnerVotes):
    for x in range(len(BlockGraph)):
        Opinions2[x] = random.choice(typeOfVotes)
        Countvotes(typeOfVotes, winnerVotes, Opinions2, x)
    Opinions = copy.deepcopy(Opinions2)
    return Opinions


def createCom(numberOfCom, MinPeople, MaxPeople):
    for i in range(numberOfCom):
        sizes[i] = random.randint(MinPeople, MaxPeople)


def Countvotes(typeOfVotes, winnerVotes, Opinions2, x):
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


def countMyOpinion(typeOfVotes, Opinions, votes, counter, f):
    for h1 in range(len(typeOfVotes)):
        if (Opinions[f] == typeOfVotes[h1]):
            votes[f][h1] = (votes[f][h1] + 2)
            counter = counter + 2
            return counter



def percentOfVotes(counter, typeOfVotes, votes2, threshold, Opinions, f, change):
    if (counter > 0):
        for h2 in range(len(typeOfVotes)):
            allVote = votes[f][h2]
            votes2[f][h2] = round(allVote / counter, 2)
            '''need to change opinion if A != B , percent > threshold'''
            if (votes2[f][h2] >= threshold and typeOfVotes[h2] != Opinions[f]):
                change = changeOpinion(votes2, threshold, typeOfVotes, Opinions, f, h2, change)
                return change
    return change


def changeOpinion(votes2, threshold, typeOfVotes, Opinions, f, h2, change):
    change = True
    # print("new opinion:", typeOfVotes[h2],"   old opinion:", Opinions[f], "  friend:",f, "   precent:",votes2[f][h2] )
    Opinions[f] = typeOfVotes[h2]
    return change


def CleanDouble(array):
    for i in range(len(array)):
        for j in range(len(array[0])):
            array[i][j] = 0


def Clean(array):
    for i in range(len(array)):
        array[i] = 0


def CreatePlotGraph(changeVar, TotalIter, Xlegend, Ylabel):
    dataPlot = plt.plot(changeVar, TotalIter, label='linear')
    plt.xlabel(Xlegend)
    plt.ylabel(Ylabel)
    plt.legend()
    plt.show()


def CreatescatterGraph(changeVar, WinnerGraph, Xlegend, Ylabel):
    plt.scatter(changeVar, WinnerGraph, label=("The Start Winner is: " + winnerStart))
    plt.xlabel(Xlegend)
    plt.ylabel(Ylabel)
    plt.legend()
    plt.show()


def prints(typeOfVotes, winnerVotes, winnerStart, Opinions2, edges):
    print("Start votes")
    print(np.matrix(typeOfVotes))
    print(np.matrix(winnerVotes))
    print("Start winner is:", winnerStart)
    # print(" opinions:")
    # print (Opinions2 )
    # print("edges :")
    # print(edges)

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


def getValue(a):
    for b in a:
        x = b
        a.remove(b)
        return x

def cor_check (cor):
    if(cor == 1 ):
        return "perfect positive linear relationship"
    if (cor == -1):
        return "perfect negative linear relationship"
    if (cor == 0):
        return "no linear relationship"
    if (cor > 0):
        return "positive correlation"
    if (cor < 0):
        return "negative correlation"    
    
percent_of_change = 0.6
number_of_seeds = 20
max_of_iter = 10
MinFriendsIn = 0.4
MaxFriendsIn = 0.7
MinFriendsOut = 0.1
MaxFriendsOut = 0.3
threshold = 0.5
typeOfVotes = ["A", "B", "C"]
winnerStart = ''
winnerFinal = ''
xLengthGraph = 10
Xlegend = "Number of friends"
TotalIter = [0 for a3 in range(xLengthGraph)]
changeVar = [0 for a4 in range(xLengthGraph)]
numberOfCom = 0
WinnerGraph = ["" for j in range(xLengthGraph)]
Opinions2 = {}
winnerVotes = [0 for i2 in range(len(typeOfVotes))]
change = True
blocked = set()

GarphIter = [[0 for j in range(xLengthGraph)] for i in range(number_of_seeds)]
GarphIter_avg = [0 for j in range(xLengthGraph)]

start_winner_graph = [0 for j in range(len(typeOfVotes))]
winner_per_graph = [[0 for j in range(xLengthGraph)] for i in range(number_of_seeds)]
winner_per_graph_avg = [0 for j in range(xLengthGraph)]

start_change_winner_graph = "T"
winner_change_graph = [[0 for j in range(xLengthGraph)] for i in range(number_of_seeds)]
winner_change_graph_avg = [0 for j in range(xLengthGraph)]

'''checking on different size of communities'''
for seedi in range(number_of_seeds):
    seede = 363 + seedi * 140
    random.seed(seede)
    print("######################### next seed , the seed is ", seede, "#################")
    Clean(winnerVotes)

    for a1 in range(xLengthGraph):
        Clean(winnerVotes)
        numberOfCom = a1 + 2
        change = True
        numOfIteration = 0
        MinPeople = round(100 / numberOfCom) - 5
        MaxPeople = round(100 / numberOfCom) + 5
        print("The Round : ", a1 + 1, "the", Xlegend, " min  : ", MinPeople, " max  : ", MaxPeople)

        sizes = [0 for i in range(numberOfCom)]
        probs = [[0 for i in range(numberOfCom)] for j in range(numberOfCom)]

        '''creats random communities '''
        for i in range(numberOfCom):
            sizes[i] = random.randint(MinPeople, MaxPeople)
        changeVar[a1] = numberOfCom

        probMatrix(numberOfCom, probs, MinFriendsIn, MaxFriendsIn)

        print("size of Coum :")
        print(sizes)

        #  print("probs :")
        #   print(np.matrix(probs))

        BlockGraph = nx.stochastic_block_model(sizes, probs, seed=seede)
        edges = nx.edges(BlockGraph)
        friends = [set() for j in range(len(BlockGraph))]
        Opinions = creatOpinions(BlockGraph, typeOfVotes, Opinions2, winnerVotes)
        votes = [[0 for i in range(len(typeOfVotes))] for j in range(len(friends))]
        votes2 = [[0 for i in range(len(typeOfVotes))] for j in range(len(friends))]
        #פה
        winnerStart = getWinner(winnerVotes, typeOfVotes)
        start_change_winner_graph = copy.deepcopy(winnerStart)
        prints(typeOfVotes, winnerVotes, winnerStart, Opinions2, edges)
        start_winner_graph = copy.deepcopy(winnerVotes)
        for x in range(len(BlockGraph)):
            Countvotes(typeOfVotes, winnerVotes, Opinions2, x)
            Clean(winnerVotes)

        setFridends(edges, friends)

        #   print("friends :")
        #  print(friends)

        for number_of_iter in range(max_of_iter):
            numOfIteration = 0
            blocked.clear()
            while (True):
                numOfIteration += 1
                choose_friends(friends, Opinions, BlockGraph, blocked, percent_of_change)
                if (len(blocked) == len(BlockGraph)):
                    # print("Enter Break")
                    break
        #  print(np.matrix(Opinions))
        Clean(winnerVotes)
        for x1 in range(len(BlockGraph)):
            Countvotes(typeOfVotes, winnerVotes, Opinions, x1)

        print("final votes")
        print(np.matrix(typeOfVotes))
        print(np.matrix(winnerVotes))
        WinnerGraph[a1] = getWinner(winnerVotes, typeOfVotes)
        print("Final winner is:", WinnerGraph[a1])

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
    #print(np.matrix(probs))
    print()
    print("End  : ")
    # print(np.matrix(changeVar))
    print(np.matrix(TotalIter))

############################################## Graphs ###################################

# Number of Changes Opinions graph

# box plots graph
print()
box_lists = list()
box_lists_sparate = list()
for j in range(len(TotalIter)):
    for i in range (number_of_seeds):
        box_lists_sparate.append(GarphIter[i][j]/10)
    box1 = box_lists_sparate[:]
    box_lists.append(box1)
    box_lists_sparate.clear()

print(box_lists)
print()
ax = sns.boxplot(data=box_lists)
plt.xticks(range(0,xLengthGraph),changeVar)
plt.xlabel("Size of Com")
plt.ylabel("Number of Changes Opinions")
plt.show()

# graph
print("Number of Iterations VS number of com plot")
print(np.matrix(GarphIter))
for i in range (number_of_seeds):
    for j in range(len(TotalIter)):
        GarphIter_avg[j] = GarphIter_avg[j] + GarphIter[i][j]

print(np.matrix(GarphIter_avg))

for i in range(len(TotalIter)):
    GarphIter_avg[i] = round(GarphIter_avg[i]/number_of_seeds,3)

print(np.matrix(GarphIter_avg))

Ylabel= "Number of Iterations"
CreatePlotGraph (changeVar, GarphIter_avg , "Number of Communities" , Ylabel)

# Winner Present Votes

#boxplot
box_lists_sparate.clear()
box_lists.clear()
print()
for j in range(len(TotalIter)):
    for i in range (number_of_seeds):
        box_lists_sparate.append(winner_per_graph[i][j])
    box1 = box_lists_sparate[:]
    box_lists.append(box1)
    box_lists_sparate.clear()

print(box_lists)
print()
ax = sns.boxplot(data=box_lists)
plt.xticks(range(0,xLengthGraph),changeVar)
plt.xlabel("Size Of Com")
plt.ylabel("Diff Between Start to End Present Winner Votes")
plt.show()

# graph
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
CreatePlotGraph (changeVar, winner_per_graph_avg , "Number of Communities" , Ylabel)

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
CreatePlotGraph (changeVar, winner_change_graph_avg , "Number of Communities" , Ylabel)

## Correlation Between x and y
cor1 = scipy.stats.pearsonr(changeVar, GarphIter_avg)[0]
cor2 = scipy.stats.pearsonr(changeVar, winner_per_graph_avg)[0]
cor3 = scipy.stats.pearsonr(changeVar, winner_change_graph_avg)[0]

graph1_cor = cor_check(cor1)
graph2_cor = cor_check(cor2)
graph3_cor = cor_check(cor3)
print()
print("graph 1 Correlation Between x and y :", round(cor1,3), graph1_cor)
print("graph 2 Correlation Between x and y :", round(cor2,3), graph2_cor)
print("graph 3 Correlation Between x and y :", round(cor3,3), graph3_cor)
