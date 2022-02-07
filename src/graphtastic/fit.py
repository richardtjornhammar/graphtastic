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
import operator
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


if bUseNumba :
    @jit(nopython=True)
    def exclusive_pdist ( P:np.array , Q:np.array , power:int=2, bInvPow:bool=False ) -> np.array :
        Np , Nq = len(P), len(Q)
        R2 = np.zeros(Np*Nq).reshape(Np,Nq)
        for i in range(len(P)):
            for j in range(len(Q)):
                R2[i][j] = np.sum((P[i]-Q[j])**power)
        if bLengthScale :
            return ( R2**(1.0/power) )
        else :
            return ( R2 )
else :
    def exclusive_pdist ( P:np.array , Q:np.array , power:int=2,  bInvPow:bool=False ) -> np.array :
        Np , Nq = len(P), len(Q)
        R2 = np.zeros(Np*Nq).reshape(Np,Nq)
        for i in range(len(P)):
            for j in range(len(Q)):
                R2[i][j] = np.sum((P[i]-Q[j])**power)
        if bLengthScale :
            return ( R2**(1.0/power) )
        else :
            return ( R2 )

def absolute_coordinates_to_distance_matrix ( Q:np.array , power:int=2 , bInvPow:bool=False ) -> np.array :
    DP = np.array( [ np.sum((np.array(p)-np.array(q))**power) for p in Q for q in Q] ).reshape(np.shape(Q)[0],np.shape(Q)[0])
    if bLengthScale :
        DP = DP**(1.0/power)
    return ( DP )

distance_matrix_to_geometry_conversion_notes__ = """
*) TAKE NOTE THAT THE OLD ALGORITHM CALLED DISTANCE GEOMETRY EXISTS. IT CAN BE EMPLOYED TO ANY DIMENSIONAL DATA. HERE YOU FIND A SVD BASED VERSION
*) THE DISTANCE MATRIX CONVERSION ROUTINE BACK TO ABSOLUTE COORDINATES USES R2 DISTANCES.
"""

if bUseNumba :
        @jit(nopython=True)
        def distance_matrix_to_absolute_coordinates ( D:np.array , bSquared:bool = True , n_dimensions:int = 2 , power:int=2 )->np.array :
                # C++ https://github.com/richardtjornhammar/RichTools/commit/be0c4dfa8f61915b0701561e39ca906a9a2e0bae
                if not bSquared :
                        D = D**power
                DIM = n_dimensions
                DIJ = D*0.
                M = len(D)
                for i in range(M) :
                        for j in range(M) :
                                DIJ[i,j] = 0.5* (D[i,-1]+D[j,-1]-D[i,j])
                D = DIJ
                U,S,Vt = np.linalg.svd ( D , full_matrices = True )
                S[DIM:] *= 0.
                Z = np.diag(S**0.5)[:,:DIM]
                xr = np.dot( Z.T,Vt )
                return ( xr.T )
else :
        def distance_matrix_to_absolute_coordinates ( D:np.array , bSquared:bool = True , n_dimensions = 2 , power:int = 2 ) -> np.array :
                # C++ https://github.com/richardtjornhammar/RichTools/commit/be0c4dfa8f61915b0701561e39ca906a9a2e0bae
                if not bSquared :
                        D = D**power
                DIM = n_dimensions
                DIJ = D*0.
                M = len(D)
                for i in range(M) :
                        for j in range(M) :
                                DIJ[i,j] = 0.5* (D[i,-1]+D[j,-1]-D[i,j])
                D = DIJ
                U,S,Vt = np.linalg.svd ( D , full_matrices = True )
                S[DIM:] *= 0.
                Z = np.diag(S**0.5)[:,:DIM]
                xr = np.dot( Z.T,Vt )
                return ( xr.T )

def select_from_distance_matrix ( boolean_list:bool , distance_matrix:np.array ) -> np.array :
    return ( np.array( [ d[boolean_list] for d in distance_matrix[boolean_list]] ) )

def scoring_function ( l1:str,l2:str ) -> float :
    s_ = np.log2(  2*( l1==l2 ) + 1 )
    return ( s_ )

def check_input ( strp:list[str] ) :
    err_msg = "must be called with two strings placed in a list"
    bad = False
    if not 'list' in str( type(strp) ) :
        bad = True
    else:
        for str_ in strp :
            if not 'str' in str(type(str_)):
                bad=True
    if bad :
        print ( err_msg )
        exit ( 1 )

def sdist ( strp:list[str] , scoring_function = scoring_function ) :
    check_input( strp )
    s1 , s2 = strp[0] , strp[1]
    N  , M  = len(s1) , len(s2)
    mg = np.meshgrid( range(N),range(M) )
    W  = np.zeros(N*M).reshape(N,M)
    for pos in zip( mg[0].reshape(-1),mg[1].reshape(-1) ):
        pos_ = np.array( [(pos[0]+0.5)/N , (pos[1]+0.5)/M] )
        dij = np.log2( np.sum( np.diff(pos_)**2 ) + 1 ) + 1
        sij = scoring_function( s1[pos[0]],s2[pos[1]] )
        W [ pos[0],pos[1] ] = sij/dij
    return ( W )

def score_alignment ( string_list:list[str] ,
                      scoring_function = scoring_function ,
                      shift_allowance:int = 1 , off_diagonal_power:int = None ,
                      main_diagonal_power:int = 2 ) -> float :
    check_input(string_list)
    strp  = string_list.copy()
    n,m   = len(strp[0]) , len(strp[1])
    shnm  = [n,m]
    nm,mn = np.max( shnm ) , np.min( shnm )
    axis  = int( n>m )
    paddington = np.repeat([s for s in strp[axis]],shnm[axis]).reshape(shnm[axis],shnm[axis]).T.reshape(-1)[:nm]
    strp[axis] = ''.join(paddington)
    W          = sdist( strp , scoring_function=scoring_function)
    if axis==1 :
        W = W.T
    Smax , SL = 0,[0]

    mdp = main_diagonal_power
    sha = shift_allowance
    for i in range(nm) :
        Sma_ = np.sum( np.diag( W,i ))**mdp
        for d in range( sha ) :
            p_ = 1.
            d_ = d + 1
            if 'list' in str(type(off_diagonal_power)):
                if len ( off_diagonal_power ) == sha :
                    p_ = off_diagonal_power[d]
            if i+d_ < nm :
                Sma_ += np.sum( np.diag( W , i+d_ ))**p_
            if i-d_ >= 0 :
                Sma_ += np.sum( np.diag( W , i-d_ ))**p_
        if Sma_ > Smax:
            Smax = Sma_
            SL.append(Sma_)
    return ( Smax/(2*sha+1)/(n+m)*mn )

def kabsch_alignment( P:np.array,Q:np.array )->np.array :
    #
    # https://en.wikipedia.org/wiki/Kabsch_algorithm
    # C++ VERSION: https://github.com/richardtjornhammar/RichTools/blob/master/src/richfit.cc
    # as of commit https://github.com/richardtjornhammar/RichTools/commit/99c79d94c2338252b1ef1067c0c061179b6edbd9 (YEAR:2016)
    # IN VINCINITY OF LINE 524
    #
    N,DIM  = np.shape( P )
    M,DIM  = np.shape( Q )
    if DIM>N or not N==M :
        print( 'MALFORMED COORDINATE PROBLEM' )
        exit( 1 )

    q0 , p0 = np.mean(Q,0) , np.mean(P,0)
    cQ , cP = Q - q0 , P - p0

    H = np.dot(cP.T,cQ)
    I  = np.eye( DIM )

    U, S, VT = np.linalg.svd( H, full_matrices=False )
    Ut = np.dot( VT.T,U.T )
    I[DIM-1,DIM-1] = 2*(np.linalg.det(Ut) > 0)-1
    ROT = np.dot( VT.T,np.dot(I,U.T) )
    B = np.dot(ROT,P.T).T + q0 - np.dot(ROT,p0)

    return ( B )


def shape_alignment( P:np.array, Q:np.array ,
                bReturnTransform :bool = False ,
                bShiftModel:bool = True ,
                bUnrestricted:bool = False ) -> np.array :
    #
    # [*] C++ VERSION: https://github.com/richardtjornhammar/RichTools/blob/master/src/richfit.cc
    # as of commit https://github.com/richardtjornhammar/RichTools/commit/99c79d94c2338252b1ef1067c0c061179b6edbd9 (YEAR:2016)
    # FIND SHAPE FIT FOR A SIMILIAR CODE IN THE RICHFIT REPO
    #
    description = """
     A NAIVE SHAPE FIT PROCEDURE TO WHICH MORE SOPHISTICATED
     VERSIONS WRITTEN IN C++ CAN BE FOUND IN MY C++[*] REPO

     HERE WE WORK UNDER THE ASSUMPTION THAT Q IS THE MODEL
     SO THAT WE SHOULD HAVE SIZE Q < SIZE P WITH UNKNOWN
     ORDERING AND THAT THEY SHARE A COMMON SECOND DIMENSION

     IN THIS ROUTINE THE COARSE GRAINED DATA ( THE MODEL ) IS
     MOVED TO FIT THE FINE GRAINED DATA ( THE DATA )
    """

    N,DIM  = np.shape( P )
    M,DIM  = np.shape( Q )
    W = (N<M)*N+(N>=M)*M

    if (DIM>W or N<M) and not bUnrestricted :
        print ( 'MALFORMED PROBLEM' )
        print ( description )
        exit ( 1 )

    q0 , p0 = np.mean(Q,0) , np.mean(P,0)
    cQ , cP = Q - q0 , P - p0
    sQ = np.dot( cQ.T,cQ )
    sP = np.dot( cP.T,cP )

    H = np.dot(sP.T,sQ)
    I = np.eye( DIM )

    U, S, VT = np.linalg.svd( H, full_matrices=False )
    Ut = np.dot( VT.T,U.T )
    I[DIM-1,DIM-1] = 2*(np.linalg.det(Ut) > 0)-1
    ROT = np.dot( VT.T,np.dot(I,U.T) )
    if bReturnTransform :
        return ( ROT,q0,p0 )

    if bShiftModel :# SHIFT THE COARSE GRAINED DATA
        B = np.dot(ROT,Q.T).T +p0 - np.dot(ROT,q0)
    else : # SHIFT THE FINE GRAINED DATA
        B = np.dot(ROT,P.T).T +q0 - np.dot(ROT,p0)

    return ( B )


def high_dimensional_alignment ( P:np.array , Q:np.array ) -> np.array :
    # HIGHER DIMENSIONAL VERSION OF
    # def KabschAlignment ( P , Q )
    #
    # https://en.wikipedia.org/wiki/Kabsch_algorithm
    # C++ VERSION: https://github.com/richardtjornhammar/RichTools/blob/master/src/richfit.cc
    #   IN VINCINITY OF LINE 524
    #
    # https://github.com/richardtjornhammar/RichTools/blob/master/src/richfit.cc
    # as of commit https://github.com/richardtjornhammar/RichTools/commit/99c79d94c2338252b1ef1067c0c061179b6edbd9 2016
    # SHAPE ALIGNMENT SEARCH FOR (shape_fit) SHAPE FIT
    #
    # THE DISTANCE GEMOETRY TO ABSOLUTE COORDINATES CAN BE FOUND HERE (2015)
    # https://github.com/richardtjornhammar/RichTools/commit/a6eef7c0712d1f87a20f319f951e09379a4171f0#diff-25a6634263c1b1f6fc4697a04e2b9904ea4b042a89af59dc93ec1f5d44848a26
    #
    # ALSO AN ALIGNMENT METHOD BUT NOT REDUCED TO ELLIPSOIDS WHERE THERE ARE SIGN AMBIGUITIES
    #
    # HERE P IS THE MODEL AND Q IS THE DATA
    # WE MOVE THE MODEL
    #
    N , DIM  = np.shape( P )
    M , DIM  = np.shape( Q )
    P0 = P.copy()
    Q0 = Q.copy()
    #
    if DIM > N :
        print ( 'MALFORMED COORDINATE PROBLEM' )
        exit ( 1 )
    #
    DP = np.array( [ np.sqrt(np.sum((p-q)**2)) for p in P for q in P ] ) .reshape( N,N )
    DQ = np.array( [ np.sqrt(np.sum((p-q)**2)) for p in Q for q in Q ] ) .reshape( M,M )
    #
    PX = distance_matrix_to_absolute_coordinates ( DP , n_dimensions = DIM )
    QX = distance_matrix_to_absolute_coordinates ( DQ , n_dimensions = DIM )
    #
    P = QX
    Q = Q
    #
    q0 , p0 , p0x = np.mean(Q,0) , np.mean(P,0), np.mean(PX,0)
    cQ , cP = Q - q0 , P - p0
    #
    H = np.dot(cP.T,cQ)
    I  = np.eye( DIM )
    #
    U, S, VT = np.linalg.svd( H, full_matrices = False )
    Ut = np.dot( VT.T,U.T )
    I[DIM-1,DIM-1] = 2*(np.linalg.det(Ut) > 0)-1
    ROT = np.dot( VT.T,np.dot(I,U.T) )
    #
    B = np.dot(ROT,PX.T).T + q0 - np.dot(ROT,p0x)
    #
    return ( B )
