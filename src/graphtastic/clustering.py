"""
Copyright 2022 RICHARD TJÖRNHAMMAR

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy as np
import typing
import sys

try :
        from numba import jit
        bUseNumba = True
except ImportError :
        print ( "ImportError:"," NUMBA. WILL NOT USE IT")
        bUseNumba = False
except OSError:
        print ( "OSError:"," NUMBA. WILL NOT USE IT")
        bUseNumba = False

# THE FOLLOWING KMEANS ALGORITHM IS THE AUTHOR OWN LOCAL VERSION
if bUseNumba :
        @jit(nopython=True)
        def seeded_kmeans( dat:np.array, cent:np.array ) -> dict :
                #
                # PYTHON ADAPTATION OF MY C++ CODE THAT CAN BE FOUND IN
                # https://github.com/richardtjornhammar/RichTools/blob/master/src/cluster.cc
                # AROUND LINE 2345
                # AGAIN CONSIDER USING THE C++ VERSION SINCE IT IS ALOT FASTER
                # HERE WE SPEED IT UP USING NUMBA IF THE USER HAS IT INSTALLED AS A MODULE
                #
                NN , MM = np.shape ( dat  )
                KK , LL = np.shape ( cent )
                if not LL == MM :
                        print ( 'WARNING DATA FORMAT ERROR. NON COALESCING COORDINATE AXIS' )

                labels = [ int(z) for z in np.zeros(NN) ]
                w = labels
                counts = np.zeros(KK)
                tmp_ce = np.zeros(KK*MM).reshape(KK,MM)
                old_error , error , TOL = 0. , 1. , 1.0E-10
                while abs ( error - old_error ) > TOL :
                        old_error = error
                        error = 0.
                        counts = counts * 0.
                        tmp_ce = tmp_ce * 0.
                        # START BC
                        for h in range ( NN ) :
                                min_distance = 1.0E30
                                for i in range ( KK ) :
                                        distance = np.sum( ( dat[h]-cent[i] )**2 )
                                        if distance < min_distance :
                                                labels[h] = i
                                                min_distance = distance
                                tmp_ce[labels[h]] += dat[ h ]
                                counts[labels[h]] += 1.0
                                error += min_distance
                        # END BC
                        for i in range ( KK ) :
                                if counts[i]>0:
                                        cent[i] = tmp_ce[i]/counts[i]
                centroids = cent
                return ( { 'labels': labels , 'centroids': centroids } )
else :
        def seeded_kmeans( dat:np.array, cent:np.array ) -> dict :
                #
                # SLOW SLUGGISH KMEANS WITH A DUBBLE FOR LOOP
                # IN PYTHON! WOW! SUCH SPEED!
                #
                NN , MM = np.shape ( dat  )
                KK , LL = np.shape ( cent )
                if not LL == MM :
                        print ( 'WARNING DATA FORMAT ERROR. NON COALESCING COORDINATE AXIS' )

                labels = [ int(z) for z in np.zeros(NN) ]
                w = labels
                counts = np.zeros(KK)
                tmp_ce = np.zeros(KK*MM).reshape(KK,MM)
                old_error , error , TOL = 0. , 1. , 1.0E-10
                while abs ( error - old_error ) > TOL :
                        old_error = error
                        error = 0.
                        counts = counts * 0.
                        tmp_ce = tmp_ce * 0.
                        # START BC
                        for h in range ( NN ) :
                                min_distance = 1.0E30
                                for i in range ( KK ) :
                                        distance = np.sum( ( dat[h]-cent[i] )**2 )
                                        if distance < min_distance :
                                                labels[h] = i
                                                min_distance = distance
                                tmp_ce[labels[h]] += dat[ h ]
                                counts[labels[h]] += 1.0
                                error += min_distance
                        # END BC
                        for i in range ( KK ) :
                                if counts[i]>0:
                                        cent[i] = tmp_ce[i]/counts[i]
                centroids = cent
                return ( { 'labels':labels , 'centroids':centroids } )


if bUseNumba :
        @jit(nopython=True)
        def connectivity ( B:np.array , val:float , bVerbose:bool = False ) -> dict :
                description = """ This is a cutoff based clustering algorithm. The intended use is to supply a distance matrix and a cutoff value (then becomes symmetric positive).  For a small distance cutoff, you should see all the parts of the system and for a large distance cutoff, you should see the entire system. It has been employed for statistical analysis work as well as the original application where it was employed to segment molecular systems."""
                if bVerbose :
                        print ( "CONNECTIVITY CLUSTERING OF ", np.shape(B), " MATRIX" )
                # PYTHON ADAPTATION OF MY C++ CODE THAT CAN BE FOUND IN
                # https://github.com/richardtjornhammar/RichTools/blob/master/src/cluster.cc
                # AROUND LINE 2277
                # CONSIDER COMPILING AND USING THAT AS A MODULE INSTEAD OF THIS SINCE IT IS
                # A LOT FASTER
                # FOR A DESCRIPTION READ PAGE 30 (16 INTERNAL NUMBERING) of:
                # https://kth.diva-portal.org/smash/get/diva2:748464/FULLTEXT01.pdf
                #
                # https://github.com/richardtjornhammar/RichTools/blob/master/src/cluster.cc
                # ADDED TO RICHTOOLS HERE: https://github.com/richardtjornhammar/RichTools/commit/74b35df9c623bf03570707a24eafe828f461ed90#diff-25a6634263c1b1f6fc4697a04e2b9904ea4b042a89af59dc93ec1f5d44848a26
                # CONNECTIVITY SEARCH FOR (connectivity) CONNECTIVITY
                #
                nr_sq,mr_sq = np.shape(B)
                if nr_sq != mr_sq :
                        print ( 'ERROR: FAILED' )
                N = mr_sq
                res , nvisi, s, NN, ndx, C = [0], [0], [0], [0], [0], 0
                res .append(0)
                for i in range(N) :
                        nvisi.append(i+1)
                        res.append(0); res.append(0)
                        ndx.append(i)

                res   = res[1:]
                nvisi = nvisi[1:]
                ndx   = ndx[1:]
                while ( len(ndx)>0 ) :
                        i = ndx[-1] ; ndx = ndx[:-1]
                        NN = []
                        if ( nvisi[i]>0 ) :
                                C-=1
                                for j in range(N) :
                                        if ( B[i,j]<=val ) :
                                                NN.append(j)
                                while ( len(NN)>0 ) :
                                        # back pop_back
                                        k = NN[-1]; NN = NN[:-1]
                                        nvisi[k] = C
                                        for j in range(N):
                                                if ( B[j,k]<=val ) :
                                                        for q in range(N) :
                                                                if ( nvisi[q] == j+1 ) :
                                                                        NN.append(q)
                if bVerbose : # VERBOSE
                        print ( "INFO "+str(-1*C) +" clusters" )
                Nc = [ 0 for i in range(-1*C) ]
                for q in range(N) :
                        res[  q*2+1 ] = q;
                        res[  q*2   ] = nvisi[q]-C;
                        Nc [res[q*2]]+= 1;
                        if bVerbose :
                                print ( " "+str(res[q*2])+" "+str(res[2*q+1]) )
                if bVerbose :
                        for i in range(-1*C) :
                                print( "CLUSTER "  +str(i)+ " HAS " + str(Nc[i]) + " ELEMENTS")
                return ( { 'clustercontent' : Nc , 'clustercontacts' : np.array(res[:-1]).reshape(-1,2) } )
else :
        def connectivity ( B:np.array , val:float , bVerbose:bool = False ) -> dict :
                description="""
This is a cutoff based clustering algorithm. The intended use is to supply a distance matrix and a cutoff value (then becomes symmetric positive).  For a small distanc>
        """
                if bVerbose :
                        print ( "CONNECTIVITY CLUSTERING OF ", np.shape(B), " MATRIX" )
                # PYTHON ADAPTATION OF MY C++ CODE THAT CAN BE FOUND IN
                # https://github.com/richardtjornhammar/RichTools/blob/master/src/cluster.cc
                # AROUND LINE 2277
                # CONSIDER COMPILING AND USING THAT AS A MODULE INSTEAD OF THIS SINCE IT IS
                # A LOT FASTER
                # FOR A DESCRIPTION READ PAGE 30 (16 INTERNAL NUMBERING) of:
                # https://kth.diva-portal.org/smash/get/diva2:748464/FULLTEXT01.pdf
                #
                nr_sq,mr_sq = np.shape(B)
                if nr_sq != mr_sq :
                        print ( 'ERROR' )
                        return ( -1 )
                N = mr_sq
                res , nvisi, s, NN, ndx, C = [], [], [], [], [], 0
                res .append(0)
                for i in range(N) :
                        nvisi.append(i+1)
                        res.append(0); res.append(0)
                        ndx.append(i)
                while ( len(ndx)>0 ) :
                        i = ndx[-1] ; ndx = ndx[:-1]
                        NN = []
                        if ( nvisi[i]>0 ) :
                                C-=1
                                for j in range(N) :
                                        if ( B[i,j]<=val ) :
                                                NN.append(j)
                                while ( len(NN)>0 ) :
                                        # back pop_back
                                        k = NN[-1]; NN = NN[:-1]
                                        nvisi[k] = C
                                        for j in range(N):
                                                if ( B[j,k]<=val ) :
                                                        for q in range(N) :
                                                                if ( nvisi[q] == j+1 ) :
                                                                        NN.append(q)
                if bVerbose : # VERBOSE
                        print ( "INFO "+str(-1*C) +" clusters" )
                Nc = [ 0 for i in range(-1*C) ]
                for q in range(N) :
                        res[  q*2+1 ] = q;
                        res[  q*2   ] = nvisi[q]-C;
                        Nc [res[q*2]]+= 1;
                        if bVerbose :
                                print ( " "+str(res[q*2])+" "+str(res[2*q+1]) )
                if bVerbose:
                        for i in range(-1*C) :
                                print( "CLUSTER "  +str(i)+ " HAS " + str(Nc[i]) + " ELEMENTS")
                return ( { 'clustercontent' : Nc , 'clustercontacts' : np.array(res[:-1]).reshape(-1,2) } )

if bUseNumba :
        @jit(nopython=True)
        def connectedness ( distm:np.array , alpha:float , n_connections:int=1 ) -> list :
            #
            # AN ALTERNATIVE METHOD
            # DOES THE SAME THING AS THE CONNECTIVITY CODE IN MY
            # CLUSTERING MODULE (in src/impetuous/clustering.py )
            # OR IN https://github.com/richardtjornhammar/RichTools/blob/master/src/cluster.cc
            # https://github.com/richardtjornhammar/RichTools/commit/74b35df9c623bf03570707a24eafe828f461ed90#diff-25a6634263c1b1f6fc4697a04e2b9904ea4b042a89af59dc93ec1f5d44848a26
            # CONNECTIVITY SEARCH FOR (connectivity) CONNECTIVITY
            #
            # THIS ROUTINE RETURNS A LIST BELONGING TO THE CLUSTERS
            # WITH THE SET OF INDICES THAT MAPS TO THE CLUSTER
            #
            if len ( distm.shape ) < 2 :
                print ( 'PLEASE SUBMIT A SQUARE DISTANCE MATRIX' )

            def b2i ( a:list ) -> list :
                return ( [ i for b,i in zip(a,range(len(a))) if b ] )
            def f2i ( a:list,alf:float ) -> list :
                return ( b2i( a<=alf ) )

            L = []
            for a in distm :
                bAdd = True
                ids = set( f2i(a,alpha) )
                for i in range(len(L)) :
                    if len( L[i]&ids ) >=  n_connections :
                        L[i] = L[i] | ids
                        bAdd = False
                        break
                if bAdd and len(ids) >= n_connections :
                    L .append( ids )
            return ( L )
else :
        def connectedness ( distm:np.array , alpha:float , n_connections:int=1 ) -> list :
            #
            # AN ALTERNATIVE METHOD
            # DOES THE SAME THING AS THE CONNECTIVITY CODE IN MY
            # CLUSTERING MODULE (in src/impetuous/clustering.py )
            # OR IN https://github.com/richardtjornhammar/RichTools/blob/master/src/cluster.cc
            # as of commit https://github.com/richardtjornhammar/RichTools/commit/76201bb07687017ae16a4e57cb1ed9fd8c394f18 2016
            # CONNECTIVITY SEARCH FOR (connectivity) CONNECTIVITY
            #
            # THIS ROUTINE RETURNS A LIST BELONGING TO THE CLUSTERS
            # WITH THE SET OF INDICES THAT MAPS TO THE CLUSTER
            #
            if len ( distm.shape ) < 2 :
                print ( 'PLEASE SUBMIT A SQUARE DISTANCE MATRIX' )

            def b2i ( a:list ) -> list :
                return ( [ i for b,i in zip(a,range(len(a))) if b ] )
            def f2i ( a:list,alf:float ) -> list :
                return ( b2i( a<=alf ) )

            L = []
            for a in distm :
                bAdd = True
                ids = set( f2i(a,alpha) )
                for i in range(len(L)) :
                    if len( L[i]&ids ) >=  n_connections :
                        L[i] = L[i] | ids
                        bAdd = False
                        break
                if bAdd and len(ids) >= n_connections :
                    L .append( ids )
            return ( L )

        S = 1. if S==0 else S
        crd = np.dot(use.values,coords_s.loc[use.index.values].values)/S
        centroid_coordinates.append(crd)


def dbscan ( coordinates:np.array = None , distance_matrix:np.array = None ,
        eps:float = None, minPts:int = None , bVerbose:bool = False ) -> dict :

    def absolute_coordinates_to_distance_matrix ( Q:np.array , power:int=2 , bInvPow:bool=False ) -> np.array :
        DP = np.array( [ np.sum((np.array(p)-np.array(q))**power) for p in Q for q in Q] ).reshape(np.shape(Q)[0],np.shape(Q)[0])
        if bInvPow :
            DP = DP**(1.0/power)
        return ( DP )

    if bVerbose :
        print ( "THIS IMPLEMENTATION FOR DBSCAN" )
        print ( "ASSESSMENT OF NOISE DIFFERS FROM" )
        print ( "THE IMPLEMENTATION FOUND IN SKLEARN" )
        print ( "ASSUMES LINEAR DISTANCES, NOT SQUARED" )
    #
    # FOR A DESCRIPTION OF THE CONNECTIVITY READ PAGE 30 (16 INTERNAL NUMBERING) of:
    # https://kth.diva-portal.org/smash/get/diva2:748464/FULLTEXT01.pdf
    #from impetuous.clustering import absolute_coordinates_to_distance_matrix
    #from impetuous.clustering import connectivity

    import operator
    if not operator.xor( data_frame is None , distance_matrix is None ) :
        print ( "ONLY SUPPLY A SINGE DATA FRAME OR A DISTANCE MATRIX" )
        print ( "dbscan FAILED" )
        print ( "DATA MATRICES NEEDS TO BE SPECIFIED WITH \" distance_matrix = ... \" " )
        exit(1)

    if distance_matrix is None :
        distance_matrix_ = absolute_coordinates_to_distance_matrix ( coordinates )
        eps = eps**2.0
    else :
        distance_matrix_ = distance_matrix

    isNoise = np.sum(distance_matrix_<eps,0)-1 < minPts
    i_ = 0
    for ib in isNoise :
        if ib :
            distance_matrix_ [ i_] = ( 1+eps )*10.0
            distance_matrix_.T[i_] = ( 1+eps )*10.0
            distance_matrix_[i_][i_] = 0.
        i_ = i_+1
    rd = connectivity ( distance_matrix_ , eps )
    clustercontent , clustercontacts  =  rd['clustercontent'] , rd['clustercontacts']
    return ( {'cluster content': clustercontent, 'clusterid-particleid' : clustercontacts, 'is noise':isNoise} )

def reformat_dbscan_results ( results:dict ) -> dict :
    if True :
        clusters = {}
        for icontent in range(len(results['cluster content'])) :
            content = results[ 'cluster content' ][ icontent ]
            for c in results [ 'clusterid-particleid' ] :
                if c[0] == icontent :
                    if results[ 'is noise' ][c[1]] :
                        icontent=-1
                    if icontent in clusters:
                        clusters[ icontent ] .append( c[1] )
                    else :
                        clusters[ icontent ] = [ c[1] ]
        return ( clusters )


"""
def calculate_rdf ( particles_i = None , particles_o = None , nbins=100 ,
                    distance_matrix = None , bInGroup = None , bNotInGroup = None ,
                    n_dimensions = 3 , xformat="%.3f" ,
                    constant=4.0/3.0 , rho=1.0 , rmax=None ,
                    bRemoveZeros = False ) :

    import operator
    crit0 = particles_i is None
    crit1 = particles_i is None and particles_o is None
    crit2 = bInGroup is None and distance_matrix is None and bNotInGroup is None

    if not crit2 :
        particles_i = distance_matrix_to_absolute_coordinates ( \
                         select_from_distance_matrix ( bInGroup    , distance_matrix ) ,
                         n_dimensions = n_dimensions ).T
        particles_o = distance_matrix_to_absolute_coordinates ( \
                         select_from_distance_matrix ( bNotInGroup , distance_matrix ) ,
                         n_dimensions = n_dimensions ).T

    if operator.xor( (not crit1) or (not crit0)  , not crit2 ) :
        if not crit0 and particles_o is None :
            particles_o = particles_i
            bRemoveZeros = True
        rdf_p = pd.DataFrame ( exclusive_pdist ( particles_i , particles_o ) ).apply( np.sqrt ).values.reshape(-1)
        if bRemoveZeros :
            rdf_p = [ r for r in rdf_p if not r==0. ]
        if rmax is None :
            rmax  = np.max ( rdf_p ) / diar( n_dimensions+1 )

        rdf_p  = np.array ( [ r for r in rdf_p if r < rmax ] )
        Y_ , X = np.histogram ( rdf_p , bins=nbins )
        X_     = 0.5 * ( X[1:]+X[:-1] )

        norm   = constant * np.pi * ( ( X_ + np.diff(X) )**(n_dimensions) - X_**(n_dimensions) ) * rho
        dd     = Y_ / norm
        rd     = X_

        rdf_source = {'density_values': dd, 'density_ids':[ xformat % (d) for d in rd ] }
        return ( rdf_source , rdf_p )
"""
