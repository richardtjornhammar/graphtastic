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
R   = 0.25*np.random.randn(N).reshape(N05,2) + 1.5
P   = 0.50*np.random.randn(N).reshape(N05,2)

coordinates = np.array([*P,*R])

results = dbscan ( distance_matrix = absolute_coordinates_to_distance_matrix(coordinates,bInvPow=True) , eps=0.45 , minPts=4 )
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
    # bInvPow refers to the distance type. If True then R distances are returned
    # instead of R2 (R**2) distances. That is also computing the square root if True
    #
    distm = gf.absolute_coordinates_to_distance_matrix( coordinates , bInvPow=True )
    #
    # Now a Graph DAG is constructed from the pairwise distances
    GN.distance_matrix_to_graph_dag( distm )
    #
    # And write it to a json file so that we may employ JS visualisations
    # such as D3 or other nice packages to view our hierarchy
    GN.write_json( jsonfile='./graph_hierarchy.json' )
```

# Manually updated code backups for this library :

[GitLab](https://gitlab.com/richardtjornhammar/graphtastic) | https://gitlab.com/richardtjornhammar/graphtastic

[CSDN](https://codechina.csdn.net/m0_52121311/graphtastic) | https://codechina.csdn.net/m0_52121311/graphtastic
