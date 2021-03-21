import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import random
import numpy as np
import copy
from collections import defaultdict

def CreatePlotGraph(changeVar, TotalIter, Xlegend, Ylabel,ymin,ymax ,xmin,xmax):
    dataPlot = plt.plot(changeVar, TotalIter, label='linear')
    plt.xlabel(Xlegend)
    plt.ylabel(Ylabel)
    plt.legend()
    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)
    plt.show()

######################################## PROB  - RAND ################################################

winner_per_graph_avg = [0.232,0.208,0.247,0.21,0.201,0.243,0.224,0.224,0.232,0.238]
changeVar = [0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65]

Ylabel= "Diff Between Start to End Present Winner Votes"
#CreatePlotGraph (changeVar, winner_per_graph_avg , "Prob of friends" , Ylabel,0,0.5,0.15,0.7)

winner_change_graph_avg = [0.3, 0.16, 0.24, 0.28, 0.18, 0.18, 0.22, 0.24, 0.14, 0.1]
changeVar = [0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65]

Ylabel= "Winner change in %"
#CreatePlotGraph (changeVar, winner_change_graph_avg , "Prob of friends" , Ylabel, 0,0.5,0.15,0.7)


GarphIter_avg = [4.5, 4.46, 4.76, 4.5, 4.46, 4.86, 4.58, 4.62, 4.58, 4.56]
Ylabel= "Number of Iterations"
#CreatePlotGraph (changeVar, GarphIter_avg , "Prob of friends" , Ylabel, 2,7,0.15,0.7)
#CreatePlotGraph (changeVar, GarphIter_avg , "Prob of friends" , Ylabel, 3,6,0.15,0.7)


######################################## PROB - DOM ################################################


winner_per_graph_avg = [0.12,0.1,0.12,0.16,0.16,0.14,0.14,0.06,0.16,0.16]
changeVar = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]

Ylabel= "Winner change in %"
#CreatePlotGraph (changeVar, winner_per_graph_avg , "Prob of friends" , Ylabel,0,0.3,0.15,0.7)
#CreatePlotGraph (changeVar, winner_per_graph_avg , "Prob of friends" , Ylabel,0,0.4,0.15,0.7)

winner_change_graph_avg = [0.414,0.431, 0.406, 0.328, 0.363, 0.348, 0.373, 0.344, 0.364, 0.335]
changeVar = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]

Ylabel= "Diff Between Start to End Present Winner Votes"
#CreatePlotGraph (changeVar, winner_change_graph_avg , "Prob of friends" , Ylabel, 0.1,0.6,0.15,0.7)


GarphIter_avg = [5.52, 5.1, 5.32, 5.24, 5.14, 4.98, 5.14, 5.02, 5.02, 5.42]
Ylabel= "Number of Iterations"
#CreatePlotGraph (changeVar, GarphIter_avg , "Prob of friends" , Ylabel, 3,7,0.15,0.7)


################################# NUM COM - RAND #############################################


winner_per_graph_avg = [0.027, 0.085, 0.098, 0.076, 0.102, 0.064, 0.077, 0.09,  0.127, 0.058]
changeVar = [2,3,4,5,6,7,8,9,10,11]

Ylabel= "Diff Between Start to End Present Winner Votes"
#CreatePlotGraph (changeVar, winner_per_graph_avg , "Number of Communities" , Ylabel,0,0.3,1,12)

winner_change_graph_avg = [0.04,0.02 ,0.02 ,0.04 ,0.02 ,0.1 , 0.1 , 0.1 , 0.08, 0.2]

Ylabel= "Winner change in %"
#CreatePlotGraph (changeVar, winner_change_graph_avg , "Number of Communities" , Ylabel, 0,0.4,1,12)


GarphIter_avg = [1.7,1.76 ,2.02 ,2.04 ,2.3  ,2.04 ,2.26 ,2.34, 2.6 , 2.38]
Ylabel= "Number of Iterations"
#CreatePlotGraph (changeVar, GarphIter_avg , "Number of Communities" , Ylabel, 0,4,1,12)


################################# NUM COM - DOM #############################################

winner_per_graph_avg = [0.396, 0.435, 0.434, 0.33,  0.295, 0.251, 0.35,  0.241, 0.32,  0.168]
changeVar = [2,3,4,5,6,7,8,9,10,11]

Ylabel= "Diff Between Start to End Present Winner Votes"
#CreatePlotGraph (changeVar, winner_per_graph_avg , "Number of Communities" , Ylabel,0.1,0.5,1,12)

winner_change_graph_avg = [0.14, 0.06, 0.04, 0.02, 0.1,  0.06, 0.02, 0.12, 0.02, 0.12]

Ylabel= "Winner change in %"
#CreatePlotGraph (changeVar, winner_change_graph_avg , "Number of Communities" , Ylabel, 0,0.5,1,12)


GarphIter_avg = [3.58, 3.6,  3.88, 3.36, 3.2,  2.98, 3.26, 3.22, 3.08, 2.74]
Ylabel= "Number of Iterations"
#CreatePlotGraph (changeVar, GarphIter_avg , "Number of Communities" , Ylabel, 1,5,1,12)

#################################################################################################
#########################################barbasi#################################################

GarphIter_avg = [0.2 , 0.2 , 0.18, 0.1 , 0.12]
changeVar = [2,3,4,5,6]
Ylabel= "Winner change in %"
#CreatePlotGraph (changeVar, GarphIter_avg , "Number of Opinions" , Ylabel, 0,0.4,2,6)

GarphIter_avg = [3.16 ,3.42 ,2.24 ,1.96 ,1.6 ]
changeVar = [2,3,4,5,6]
Ylabel= "Number of Iterations"
#CreatePlotGraph (changeVar, GarphIter_avg , "Number of Opinions" , Ylabel, 1,4,2,6)

#################################################################################################


GarphIter_avg = [3.32 ,3.2 , 3.32 ,3.4 , 3.72 ,3.66 ,3.62 ,3.46, 3.72, 3.42]
changeVar = [50 , 75 ,100 ,125, 150, 175, 200 ,225 ,250, 275]
Ylabel= "Number of Iterations"
#CreatePlotGraph (changeVar, GarphIter_avg , "Number of Friends" , Ylabel, 2,5,50,275)


GarphIter_avg = [0.16, 0.36 ,0.24, 0.08, 0.22 ,0.28 ,0.24, 0.34, 0.24, 0.4]
changeVar = [50 , 75 ,100 ,125, 150, 175, 200 ,225 ,250, 275]
Ylabel= "Winner change in %"
#CreatePlotGraph (changeVar, GarphIter_avg , "Number of Friends" , Ylabel, 0,0.6,50,275)

GarphIter_avg = [0.343 ,0.272 ,0.251 ,0.216 ,0.27  ,0.215 ,0.219 ,0.186 ,0.143 ,0.154]
changeVar = [50 , 75 ,100 ,125, 150, 175, 200 ,225 ,250, 275]
Ylabel= "Diff Between Start to End Present Winner Votes"
#CreatePlotGraph (changeVar, GarphIter_avg , "Number of Friends" , Ylabel, 0.1,0.5,50,275)

############################################################################

GarphIter_avg = [2.02 ,2.26, 2.72 ,3.24 ,3.54, 3.56 ,3.84 ,3.76, 3.66, 3.54]
changeVar = [ 1 , 2  ,3  ,4  ,5  ,6  ,7 , 8 , 9 , 10]
Ylabel= "Number of Iterations"
#CreatePlotGraph (changeVar, GarphIter_avg , "Number of Arcs" , Ylabel, 1,5,1,10)

GarphIter_avg = [0.26 ,0.34 ,0.28 ,0.3  ,0.38, 0.34, 0.28, 0.2 , 0.42 ,0.32]
changeVar = [ 1 , 2  ,3  ,4  ,5  ,6  ,7 , 8 , 9 , 10]
Ylabel= "Winner change in %"
#CreatePlotGraph (changeVar, GarphIter_avg , "Number of Arcs" , Ylabel, 0.1,0.5,1,10)

GarphIter_avg = [0.012, 0.046, 0.037, 0.078, 0.097, 0.118, 0.13,  0.188, 0.122, 0.14 ]
changeVar = [ 1 , 2  ,3  ,4  ,5  ,6  ,7 , 8 , 9 , 10]
Ylabel= "Diff Between Start to End Present Winner Votes"
CreatePlotGraph (changeVar, GarphIter_avg , "Number of Arcs" , Ylabel, 0,0.25,1,10)

############################################################################
