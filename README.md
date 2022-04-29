# A Graph Learning library for Humans

These novel algorithms include but are not limited to:
* A graph construction and graph searching class can be found [here](https://github.com/richardtjornhammar/graphtastic/blob/master/src/graphtastic/graphs.py) (NodeGraph). It was developed and invented as a faster alternative for hierarchical DAG construction and searching.
* A fast DBSCAN method utilizing [my](https://richardtjornhammar.github.io/) connectivity code as invented during my PhD.
* A NLP pattern matching algorithm useful for sequence alignment clustering.
* High dimensional alignment code for aligning models to data.
* An SVD based variant of the Distance Geometry algorithm. For going from relative to absolute coordinates.

[![License](https://img.shields.io/github/license/Qiskit/qiskit.svg?)](https://opensource.org/licenses/Apache-2.0)
[![DOI](https://zenodo.org/badge/456418653.svg)](https://zenodo.org/badge/latestdoi/456418653)
[![Downloads](https://pepy.tech/badge/graphtastic)](https://pepy.tech/project/graphtastic)

Visit the active code via :
https://github.com/richardtjornhammar/graphtastic

# Pip installation with :
```
pip install graphtastic
```

# Version controlled installation of the Graphtastic library

The Graphtastic library

In order to run these code snippets we recommend that you download the nix package manager. Nix package manager links from Februari 2022:

https://nixos.org/download.html

```
$ curl -L https://nixos.org/nix/install | sh
```

If you cannot install it using your Wintendo then please consider installing Windows Subsystem for Linux first:

```
https://docs.microsoft.com/en-us/windows/wsl/install-win10
```

In order to run the code in this notebook you must enter a sensible working environment. Don't worry! We have created one for you. It's version controlled against python3.9 (and experimental python3.10 support) and you can get the file here:

https://github.com/richardtjornhammar/graphtastic/blob/master/env/env39.nix

Since you have installed Nix as well as WSL, or use a Linux (NixOS) or bsd like system, you should be able to execute the following command in a termnial:

```
$ nix-shell env39.nix
```

Now you should be able to start your jupyter notebook locally:

```
$ jupyter-notebook graphhaxxor.ipynb
```

and that's it.

# EXAMPLE 0

Running

```
import graphtastic.graphs as gg
import graphtastic.clustering as gl
import graphtastic.fit as gf
import graphtastic.convert as gc
import graphtastic.utility as gu
import graphtastic.stats as gs
```
Should work if the install was succesful


# Example 1 : Absolute and relative coordinates

In this example, we will use the SVD based distance geometry method to go between absolute coordinates, relative coordinate distances and back to ordered absolute coordinates. Absolute coordinates are float values describing the position of something in space. If you have several of these then the same information can be conveyed via the pairwise distance graph. Going from absolute coordinates to pairwise distances is simple and only requires you to calculate all the pairwise distances between your absolute coordinates. Going back to mutually orthogonal ordered coordinates from the pariwise distances is trickier, but a solved problem. The [distance geometry](https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.37.8051) can be obtained with SVD and it is implemented in the `graphtastic.fit` module under the name `distance_matrix_to_absolute_coordinates`. We start by defining coordinates afterwhich we can calculate the pair distance matrix and transforming it back by using the code below

```
import numpy as np

coordinates = np.array([[-23.7100 ,  24.1000 ,  85.4400],
  [-22.5600 ,  23.7600 ,  85.6500],
  [-21.5500 ,  24.6200 ,  85.3800],
  [-22.2600 ,  22.4200 ,  86.1900],
  [-23.2900 ,  21.5300 ,  86.4800],
  [-20.9300 ,  22.0300 ,  86.4300],
  [-20.7100 ,  20.7600 ,  86.9400],
  [-21.7900 ,  19.9300 ,  87.1900],
  [-23.0300 ,  20.3300 ,  86.9600],
  [-24.1300 ,  19.4200 ,  87.2500],
  [-23.7400 ,  18.0500 ,  87.0000],
  [-24.4900 ,  19.4600 ,  88.7500],
  [-23.3700 ,  19.8900 ,  89.5200],
  [-24.8500 ,  18.0000 ,  89.0900],
  [-23.9600 ,  17.4800 ,  90.0800],
  [-24.6600 ,  17.2400 ,  87.7500],
  [-24.0800 ,  15.8500 ,  88.0100],
  [-23.9600 ,  15.1600 ,  86.7600],
  [-23.3400 ,  13.7100 ,  87.1000],
  [-21.9600 ,  13.8700 ,  87.6300],
  [-24.1800 ,  13.0300 ,  88.1100],
  [-23.2900 ,  12.8200 ,  85.7600],
  [-23.1900 ,  11.2800 ,  86.2200],
  [-21.8100 ,  11.0000 ,  86.7000],
  [-24.1500 ,  11.0300 ,  87.3200],
  [-23.5300 ,  10.3200 ,  84.9800],
  [-23.5400 ,   8.9800 ,  85.4800],
  [-23.8600 ,   8.0100 ,  84.3400],
  [-23.9800 ,   6.5760 ,  84.8900],
  [-23.2800 ,   6.4460 ,  86.1300],
  [-23.3000 ,   5.7330 ,  83.7800],
  [-22.7300 ,   4.5360 ,  84.3100],
  [-22.2000 ,   6.7130 ,  83.3000],
  [-22.7900 ,   8.0170 ,  83.3800],
  [-21.8100 ,   6.4120 ,  81.9200],
  [-20.8500 ,   5.5220 ,  81.5200],
  [-20.8300 ,   5.5670 ,  80.1200],
  [-21.7700 ,   6.4720 ,  79.7400],
  [-22.3400 ,   6.9680 ,  80.8000],
  [-20.0100 ,   4.6970 ,  82.1500],
  [-19.1800 ,   3.9390 ,  81.4700] ]);

if __name__=='__main__':

    import graphtastic.fit as gf

    distance_matrix = gf.absolute_coordinates_to_distance_matrix( coordinates )
    ordered_coordinates = gf.distance_matrix_to_absolute_coordinates( distance_matrix , n_dimensions=3 )

    print ( ordered_coordinates )

```

You will notice that the largest variation is now aligned with the `X axis`, the second most variation aligned with the `Y axis` and the third most, aligned with the `Z axis` while the graph topology remained unchanged.


# Example 2 : Deterministic DBSCAN

[DBSCAN](https://en.wikipedia.org/wiki/DBSCAN) is a clustering algorithm that can be seen as a way of rejecting points, from any cluster, that are positioned in low dense regions of a point cloud. This introduces holes and may result in a larger segment, that would otherwise be connected via a non dense link to become disconnected and form two segments, or clusters. The rejection criterion is simple. The central concern is to evaluate a distance matrix <img src="https://render.githubusercontent.com/render/math?math=A_{ij}">  with an applied cutoff <img src="https://render.githubusercontent.com/render/math?math=\epsilon"> this turns the distances into true or false values depending on if a pair distance between point i and j is within the distance cutoff. This new binary Neighbour matrix <img src="https://render.githubusercontent.com/render/math?math=N_{ij}=A_{ij}<\epsilon"> tells you wether or not two points are neighbours (including itself). The DBSCAN criterion states that a point is not part of any cluster if it has fewer than `minPts` neighbors. Once you've calculated the distance matrix you can immediately evaluate the number of neighbors each point has and the rejection criterion, via <img src="https://render.githubusercontent.com/render/math?math=R_i=(\sum_{j} A_{ij}<\epsilon)-1 < minPts">. If the rejection vector R value of a point is True then all the pairwise distances in the distance matrix of that point is set to a value larger than epsilon. This ensures that a distance matrix search will reject those points as neighbours of any other for the choosen epsilon. By tracing out all points that are neighbors and assessing the [connectivity](https://github.com/richardtjornhammar/graphtastic/blob/master/src/graphtastic/clustering.py) (search for connectivity) you can find all the clusters.

```
import numpy as np
from graphtastic.clustering import dbscan, reformat_dbscan_results
from graphtastic.fit import absolute_coordinates_to_distance_matrix

N   = 100
N05 = int ( np.floor(0.5*N) )
P   = 0.25*np.random.randn(N).reshape(N05,2) + 1.5
Q   = 0.50*np.random.randn(N).reshape(N05,2)

coordinates = np.array([*P,*Q])

results = dbscan ( distance_matrix = absolute_coordinates_to_distance_matrix(coordinates) , eps=0.45 , minPts=4 )
clusters = reformat_dbscan_results(results)
print ( clusters )

```

# Example 3 : NodeGraph, distance matrix to DAG

Here we demonstrate how to convert the graph coordinates into a hierarchy. The leaf nodes will correspond to the coordinate positions.

```
import numpy as np

coordinates = np.array([[-23.7100 ,  24.1000 ,  85.4400],
  [-22.5600 ,  23.7600 ,  85.6500],
  [-21.5500 ,  24.6200 ,  85.3800],
  [-22.2600 ,  22.4200 ,  86.1900],
  [-23.2900 ,  21.5300 ,  86.4800],
  [-20.9300 ,  22.0300 ,  86.4300],
  [-20.7100 ,  20.7600 ,  86.9400],
  [-21.7900 ,  19.9300 ,  87.1900],
  [-23.0300 ,  20.3300 ,  86.9600],
  [-24.1300 ,  19.4200 ,  87.2500],
  [-23.7400 ,  18.0500 ,  87.0000],
  [-24.4900 ,  19.4600 ,  88.7500],
  [-23.3700 ,  19.8900 ,  89.5200],
  [-24.8500 ,  18.0000 ,  89.0900],
  [-23.9600 ,  17.4800 ,  90.0800],
  [-24.6600 ,  17.2400 ,  87.7500],
  [-24.0800 ,  15.8500 ,  88.0100],
  [-23.9600 ,  15.1600 ,  86.7600],
  [-23.3400 ,  13.7100 ,  87.1000],
  [-21.9600 ,  13.8700 ,  87.6300],
  [-24.1800 ,  13.0300 ,  88.1100],
  [-23.2900 ,  12.8200 ,  85.7600],
  [-23.1900 ,  11.2800 ,  86.2200],
  [-21.8100 ,  11.0000 ,  86.7000],
  [-24.1500 ,  11.0300 ,  87.3200],
  [-23.5300 ,  10.3200 ,  84.9800],
  [-23.5400 ,   8.9800 ,  85.4800],
  [-23.8600 ,   8.0100 ,  84.3400],
  [-23.9800 ,   6.5760 ,  84.8900],
  [-23.2800 ,   6.4460 ,  86.1300],
  [-23.3000 ,   5.7330 ,  83.7800],
  [-22.7300 ,   4.5360 ,  84.3100],
  [-22.2000 ,   6.7130 ,  83.3000],
  [-22.7900 ,   8.0170 ,  83.3800],
  [-21.8100 ,   6.4120 ,  81.9200],
  [-20.8500 ,   5.5220 ,  81.5200],
  [-20.8300 ,   5.5670 ,  80.1200],
  [-21.7700 ,   6.4720 ,  79.7400],
  [-22.3400 ,   6.9680 ,  80.8000],
  [-20.0100 ,   4.6970 ,  82.1500],
  [-19.1800 ,   3.9390 ,  81.4700] ]);


if __name__=='__main__':

    import graphtastic.graphs as gg
    import graphtastic.fit as gf
    GN = gg.NodeGraph()
    #
    distm = gf.absolute_coordinates_to_distance_matrix( coordinates )
    #
    # Now a Graph DAG is constructed from the pairwise distances
    GN.distance_matrix_to_graph_dag( distm )
    #
    # And write it to a json file so that we may employ JS visualisations
    # such as D3 or other nice packages to view our hierarchy
    GN.write_json( jsonfile='./graph_hierarchy.json' )
```

# Example 4 : NodeGraph, linkages to DAG

An alternative way of constructing a DAG hierarchy is by using distance matrix linkages. For a given distance matrix the smallest non-diagonal distance is located and the two indices that share this distance are merged into a pair cluster. The new cluster distances to all other points are then determined by either assigning the smallest distance to any of the two points or the maximum. The `max` method is sometimes refered to as complete linkage while the `min` is refered to as single linkage. The distance matrix is then updated by adding the newly calculated cluster distances while the old index distances, that are no longer of any use, are discarded. If this approximate approach is adopted instead of evaluating the `connectivity` at every unique distance in the distance matrix then you have opted for `agglomerative hierarchical clustering`. This is indeed a lot faster, but ambigious in how linkages should be determined.

You can calculate linkages as well as construct a DAG using the `NodeGraph` functionality that employs native agglomerative hierarchical clustering by executing the below code :

```
import numpy as np
import typing

if __name__=='__main__' :

    import time
    from graphtastic.hierarchical import linkages

    D = [[0,9,3,6,11],[9,0,7,5,10],[3,7,0,9,2],[6,5,9,0,8],[11,10,2,8,0] ]
    print ( np.array(D) )
    t0 = time.time()
    links = linkages( D, command='min')
    dt = time.time()-t0
    print ('min>', linkages( D, command='min') , dt) # SINGLE LINKAGE (MORE ACCURATE)
    print ('max>', linkages( D, command='max') )     # COMPLETE LINKAGE

    import graphtastic.fit as gf
    import graphtastic.graphs as gg

    GN = gg.NodeGraph()
    GN .linkages_to_graph_dag( links )
    GN .write_json( jsonfile='./graph_hierarchy.json' )

```
NOTE : This new implementation of agglomerative hierarchical clustering differs from the one present in `scipy`


# Example 5 : Connectivity, hierarchies and linkages

In the `impetuous.clustering` module you will find several codes for assessing if distance matrices are connected at some distance or not. `connectivity` and `connectedness` are two methods for establishing the number of clusters in the binary Neighbour matrix. The Neighbour matrix is just the pairwise distance between the parts `i` and `j` of your system (<img src="https://render.githubusercontent.com/render/math?math=D_{ij}">) with an applied cutoff (<img src="https://render.githubusercontent.com/render/math?math=N_{ij}=D_{ij}\le\epsilon">) and is related to the adjacency matrix from graph theory by adding an identity matrix to the adjacency matrix (<img src="https://render.githubusercontent.com/render/math?math=A_{ij}=N_{ij} - I_{ij}">). The three boolean matrices that describe a system at some distance cutoff (<img src="https://render.githubusercontent.com/render/math?math=\epsilon">) are: the Identity matrix (<img src="https://render.githubusercontent.com/render/math?math=I_{ij} = D_{ij}\equiv0 ">), the Adjacency matrix (<img src="https://render.githubusercontent.com/render/math?math=A_{ij}= D_{ij}\le\epsilon - I_{ij}">) and the Community matrix (<img src="https://render.githubusercontent.com/render/math?math=C_{ij}=D_{ij}>\epsilon">). We note that summing the three matrices will return `1` for any `i,j` pair. 

"Connection" algorithms, such as the two mentioned, evaluate every distance and add them to the same cluster if there is any true overlap for a specific distance cutoff. ["Link" algorithms](https://online.stat.psu.edu/stat555/node/85/) try to determine the number of clusters for all unique distances by reducing and ignoring some connections to already linked constituents of the system in accord with a chosen heuristic. 

The "Link" codes are more efficient at creating a link hierarchy of the data but can be thought of as throwing away information at every linking step. The lost information is deemed unuseful by the heuristic. The full link algorithm determines the new cluster distance to the rest of the points in a self consistent fashion by employing the same heuristic. Using simple linkage, or `min` value distance assignment, will produce an equivalent [hierarchy](https://online.stat.psu.edu/stat555/node/86/) as compared to the one deduced by a connection algorithm. Except for some of the cases when there are distance ties in the link evaluation. This is a computational quirk that does not affect "connection" based hierarchy construction.

The "Link" method is thereby not useful for the deterministic treatment of a particle system where all the true connections in it are important, such as in a water bulk system when you want all your quantum-mechanical waters to be treated at the same level of theory based on their connectivity at a specific level or distance. This is indeed why my connectivity algorithm was invented by me in 2009. If you are only doing black box statistics on a complete hierarchy then this distinction is not important and computational efficiency is probably what you care about. You can construct hierarchies from both algorithm types but the connection algorithm will always produce a unique and well-determined structure while the link algorithms will be unique but structurally dependent on how ties are resolved and which heuristic is employed for construction. The connection hierarchy is exact and deterministic, but slow to construct, while the link hierarchies are heuristic dependent, but fast to construct. We will study this more in the following code example as well as the case when they are equivalent.

## 5.1 Link hierarchy construction
The following code produces two distance matrices. One has distance ties and the other one does not. The second matrix is well known and the correct minimal linkage hierarchy is well known. Lets see compare the results between scipy and our method.
```
import numpy as np
from graphtastic.fit import absolute_coordinates_to_distance_matrix
from graphtastic.hierarchical import linkages, scipylinkages
from graphtastic.utility import lint2lstr

if __name__ == '__main__' :
    
    xds = np.array([ [5,2],
                   [8,4],
                   [4,6],
                   [3,7],
                   [8,7],
                   [5,10]
                  ])

    tied_D = np.array([ np.sum((p-q)**2) for p in xds for q in xds ]).reshape(len(xds),len(xds))

    print ( tied_D )
    lnx1 = linkages ( tied_D.copy() , command='min' )
    lnx2 = scipylinkages(tied_D,'min')

    print ( '\n',lnx1 ,'\n', lnx2 )
    
    D = np.array([[0,9,3,6,11],[9,0,7,5,10],[3,7,0,9,2],[6,5,9,0,8],[11,10,2,8,0] ])

    print ('\n', np.array(D) )

    lnx1 = linkages ( D , command='min' )
    lnx2 = scipylinkages( D,'min')

    print ( '\n',lnx1 ,'\n', lnx2 )
```

We study the results below
```
[[ 0 13 17 29 34 64]
 [13  0 20 34  9 45]
 [17 20  0  2 17 17]
 [29 34  2  0 25 13]
 [34  9 17 25  0 18]
 [64 45 17 13 18  0]]

 {'2.3': 2, '1.4': 9.0, '1.4.0': 13.0, '2.3.5': 13.0, '2.3.5.1.4.0': 17.0, '0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0} 
 {'1': 2.0, '4': 2.0, '0': 2.0, '2.3': 2.0, '5': 2.0, '1.4': 9.0, '0.1.4': 13.0, '2.3.5': 13.0, '0.1.2.3.4.5': 17.0}

 [[ 0  9  3  6 11]
 [ 9  0  7  5 10]
 [ 3  7  0  9  2]
 [ 6  5  9  0  8]
 [11 10  2  8  0]]

 {'2.4': 2, '2.4.0': 3.0, '1.3': 5.0, '1.3.2.4.0': 6.0, '0': 0, '1': 0, '2': 0, '3': 0, '4': 0}
 {'2.4': 2.0, '0': 2.0, '1': 2.0, '3': 2.0, '0.2.4': 3.0, '1.3': 5.0, '0.1.2.3.4': 6.0}
```
We see that the only difference for these two examples are how the unclustered indices are treated. In our method they are set to the identity distance value of zero while scipy attributes them the lowest non diagonal value in the distance matrix.

## 5.2 Connectivity hierarchy construction
Now we employ the `connectivity` algorithm for construction of the hierarchy. In the below code segment the first loop calls the function directly and the second calls the `impetuous.hierarchy_matrix` function
```
    import graphtastic.hierarchical as imph
    from graphtastic.clustering import connectivity

    unique_distances = sorted(list(set(D.reshape(-1))))
    for u in unique_distances :
        results = connectivity(D,u)
        print ( u , results )
        if len(results[0]) == 1 :
            break

    res = imph.hierarchy_matrix ( D )
    print ( res )
```
with the results
```
0 ([1, 1, 1, 1, 1], array([[0, 0],
       [1, 1],
       [2, 2],
       [3, 3],
       [4, 4]]))
2 ([1, 1, 1, 2], array([[0, 0],
       [1, 1],
       [3, 2],
       [2, 3],
       [3, 4]]))
3 ([1, 1, 3], array([[2, 0],
       [0, 1],
       [2, 2],
       [1, 3],
       [2, 4]]))
5 ([2, 3], array([[1, 0],
       [0, 1],
       [1, 2],
       [0, 3],
       [1, 4]]))
6 ([5], array([[0, 0],
       [0, 1],
       [0, 2],
       [0, 3],
       [0, 4]]))
{'hierarchy matrix':(array([[0, 1, 2, 3, 4],
       [0, 1, 3, 2, 3],
       [2, 0, 2, 1, 2],
       [1, 0, 1, 0, 1],
       [0, 0, 0, 0, 0]]),'lookup':{0: [0, 0, 1.0], 1: [1, 2, 1.25], 2: [2, 3, 1.6666666666666667], 3: [3, 5, 2.5], 4: [4, 6, 5.0]}}
```
and we see that the system has 5 unique levels. The hierarchy matrix increase in distance as you traverse down. The first row is level `0` with distance `0` and all items are assigned to each own cluster. The third row, level `2`, contains three clusters at distance `3` and the three clusters are `0.2.4` as well as `1` and `3`. We see that they become joined at level `3` corresponding to distance `5`.

The final complete clustering results can be obtained in this alternative way for the `connectivity` hierarchy
```
    print ( imph.reformat_hierarchy_matrix_results ( res['hierarchy matrix'],res['lookup'] ) )
```
with the result
```
{(0,): 0, (1,): 0, (2,): 0, (3,): 0, (4,): 0, (2, 4): 2, (0, 2, 4): 3, (1, 3): 5, (0, 1, 2, 3, 4): 6}
```
which is well aligned with the previous results, but the `connectivity` approach is slower to employ for constructing a hierarchy.

## Comparing hierarchies of an equidistant plaque

We know by heart that a triagonal mesh with a link length of one is fully connected at only that distance. So lets study what the hierarchical clustering results will yield.
```
    def generate_plaque(N) :
        L,l = 1,1
        a  = np.array( [l*0.5, np.sqrt(3)*l*0.5] )
        b  = np.array( [l*0.5,-np.sqrt(3)*l*0.5] )
        x_ = np.linspace( 1,N,N )
        y_ = np.linspace( 1,N,N )
        Nx , My = np.meshgrid ( x_,y_ )
        Rs = np.array( [ a*n+b*m for n,m in zip(Nx.reshape(-1),My.reshape(-1)) ] )
        return ( Rs )

    from graphtastic.fit import absolute_coordinates_to_distance_matrix as c2D
    D = c2D( generate_plaque(N=3))
    #
    # CONNECTIVITY CONSTRUCTION
    print ( imph.reformat_hierarchy_matrix_results ( *imph.hierarchy_matrix( D ).values() ) )
    #
    # SCIPY LINKAGE CONSTRUCTION
    print ( scipylinkages(D,'min',bStrKeys=False) )
```
which readily tells us that
```
{(0,): 0.0, (1,): 0.0, (2,): 0.0, (3,): 0.0, (4,): 0.0, (5,): 0.0, (6,): 0.0, (7,): 0.0, (8,): 0.0, (0, 1, 3, 4): 0.9999999999999999, (2, 5): 0.9999999999999999, (6, 7): 0.9999999999999999, (0, 1, 2, 3, 4, 5, 6, 7, 8): 1.0}

{(6, 7): 0.9999999999999999, (0, 1, 3, 4): 0.9999999999999999, (2, 5): 0.9999999999999999, (8,): 0.9999999999999999, (0, 1, 2, 3, 4, 5, 6, 7, 8): 1.0}
```
and we see that everything is connected at the distance `1` and that the numerical treatment seems to have confused both algorithms in a similar fashion, but that `scipy` is assigning single index clusters the distance `1`

So it is clear that a linkage method is more efficient for constructing complete hierarchies while a single `connectivity` calculation will be faster if you only want the clusters at a predetermined distance. Because in that case you don't need to calculate the entire hierarchy.


# Manually updated code backups for this library :

[GitLab](https://gitlab.com/richardtjornhammar/graphtastic) | https://gitlab.com/richardtjornhammar/graphtastic

[CSDN](https://codechina.csdn.net/m0_52121311/graphtastic) | https://codechina.csdn.net/m0_52121311/graphtastic
