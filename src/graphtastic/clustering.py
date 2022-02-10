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
        def seeded_kmeans( dat:np.array, cent:np.array ) :
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
                return ( labels , centroids  )
else :
        def seeded_kmeans( dat:np.array, cent:np.array ) :
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
                return ( labels , centroids )


if bUseNumba :
        @jit(nopython=True)
        def connectivity ( B:np.array , val:float , bVerbose:bool = False ) :
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
                return ( Nc , np.array(res[:-1]).reshape(-1,2) )
else :
        def connectivity ( B:np.array , val:float , bVerbose:bool = False ) :
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
                return ( Nc , np.array(res[:-1]).reshape(-1,2) )

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
    if not operator.xor( coordinates is None , distance_matrix is None ) :
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
    clustercontent , clustercontacts  = connectivity ( distance_matrix_ , eps )
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

def unpack ( seq ) : # seq:Union -> Union
    if isinstance ( seq,(list,tuple,set)) :
        yield from ( x for y in seq for x in unpack(y) )
    elif isinstance ( seq , dict ):
        yield from ( x for item in seq.items() for y in item for x in unpack(y) )
    else :
        yield seq

def rem ( a:list , H:list ) -> list :
    h0 = []
    for h in H:
        hp = h - np.sum(h>np.array(h0))
        h0 .append(h)
        a .pop(hp)
    return(a)

def linkage ( distm:np.array , command:str = 'max' ) -> dict :
    #
    # CALUCULATES WHEN SAIGAS ARE LINKED
    # MAX CORRESPONDS TO COMPLETE LINKAGE
    # MIN CORRESPONDS TO SINGLE LINKAGE
    #
    D = distm
    N = len(D)

    sidx = [ str(i)+'-'+str(j) for i in range(N) for j in range(N) if i<j ]
    pidx = [ [i,j] for i in range(N) for j in range(N) if i<j ]
    R = [ D[idx[0]][idx[1]] for idx in pidx ]

    label_order = lambda i,k: i+'-'+k if '.' in i else k+'-'+i if '.' in k else  i+'-'+k if int(i)<int(k) else k+'-'+i

    if command == 'max':
        func  = lambda R,i,j : [(R[i],i),(R[j],j)] if R[i]>R[j] else [(R[j],j),(R[i],i)]
    if command == 'min':
        func  = lambda R,i,j : [(R[i],i),(R[j],j)] if R[i]<R[j] else [(R[j],j),(R[i],i)]

    LINKS = {}
    cleared = set()

    for I in range( N**2 ) : # WORST CASE SCENARIO

        clear = []
        if len(R) == 0 :
            break

        nar   = np.argmin( R )
        clu_idx = sidx[nar].replace('-','.')

        LINKS = { **{ clu_idx : R[nar] } , **LINKS }
        lp    = {}

        for l in range(len(sidx)) :
            lidx = sidx[l]
            lp [ lidx ] = l

        pij   = sidx[nar].split('-')
        cidx  = set( unpack( [ s.split('-') for s in [ s for s in sidx if len(set(s.split('-'))-set(pij)) == 1 ] ] ) )-set(pij)
        ccidx = set( [ s for s in [ s for s in sidx if len(set(s.split('-'))-set(pij)) == 1 ] ] )
        found = {}

        i   = pij[0]
        j   = pij[1]

        for k in cidx :
            h0 , h1 , h2 , q0, J = None , None , None , None, 0
            if k == j or k == i :
                continue
            la1 = label_order(i,k)
            if la1 in lp :
                J+=1
                h1 = lp[ label_order(i,k) ]
                h0 = h1 ; q0 = i
            la2 = label_order(j,k)
            if la2 in lp :
                J+=1
                h2 = lp[ label_order(j,k) ]
                h0 = h2 ; q0 = j
            if J == 2 :
                Dijk = func ( R , h1 , h2 )
            elif J == 1 :
                Dijk = [[ R[h0],h0 ]]
            else :
                continue
            nidx = [ s for s in sidx[Dijk[0][1]].split('-') ]
            nidx = list( set(sidx[Dijk[0][1]].split('-'))-set(pij) )[0]
            nclu_idx = clu_idx + '-' + nidx
            found[ nclu_idx ] = Dijk[0][0]

        clear = [*[ lp[c] for c in ccidx ],*[nar]]
        cleared = cleared | set( unpack( [ sidx[c].split('-') for c in clear ]) )

        R = rem(R,clear)
        sidx = rem(sidx,clear)

        for label,d in found.items() :
            R.append(d)
            sidx.append(label)

    return ( LINKS )


if __name__=='__main__' :

    D = [[0,9,3,6,11],[9,0,7,5,10],[3,7,0,9,2],[6,5,9,0,8],[11,10,2,8,0] ]
    print ( np.array(D) )
    print ( 'min>' , linkage( D, command='min') )
    print ( 'max>' , linkage( D, command='max') )
