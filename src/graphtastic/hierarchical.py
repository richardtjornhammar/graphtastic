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
contact__ = "richard.tjornhammar@gmail.com"

import numpy as np
from graphtastic.utility import nppop, unpack, rem, lint2lstr


def hierarchy_matrix ( distance_matrix:np.array   = None ,
                       coordinates:np.array       = None ,
                       linkage_distances:np.array = None ) -> dict :
    from graphtastic.clustering import connectivity , absolute_coordinates_to_distance_matrix
    import operator
    if not operator.xor( coordinates is None , distance_matrix is None ) :
        print ( "ONLY COORDINATES OR A DISTANCE MATRIX" )
        print ( "calculate_hierarchy_matrix FAILED" )
        print ( "DATA MATRICES NEEDS TO BE SPECIFIED WITH \" distance_matrix = ... \" " )
        exit(1)
    if not coordinates is None :
        distance_matrix = absolute_coordinates_to_distance_matrix(coordinates)

    nmt_ = np.shape(distance_matrix)
    if linkage_distances is None :
        uco_v = sorted(list(set(distance_matrix.reshape(-1))))
    else :
        uco_v = sorted(list(set(linkage_distances.reshape(-1))))

    level_distance_lookup = {}
    hsers = []
    for icut in range(len(uco_v)) :
        cutoff = uco_v[icut]
        # clustercontacts : clusterid , particleid relation
        # clustercontent : clusterid to number of particles in range
        #from clustering import connectivity # LOCAL TESTING
        clustercontent , clustercontacts = connectivity ( distance_matrix , cutoff )
        #
        # internal ordering is a range so this does not need to be a dict
        level_distance_lookup[icut] = [ icut , cutoff , np.mean(clustercontent) ]
        hsers.append(clustercontacts[:,0])
        if len( set(clustercontacts[:,0]) ) == 1 : # DON'T DO HIGHER COMPLETE SYSTEM VALUES
            break
    return ( { 'hierarchy matrix':np.array(hsers) , 'lookup':level_distance_lookup} )


def reformat_hierarchy_matrix_results ( hierarchy_matrix:np.array , lookup:dict=None ) -> dict :
    CL = {}
    for i in range(len(hierarchy_matrix)):
        row = hierarchy_matrix[i]
        d = i
        if not lookup is None :
            d   = lookup[i][1]
        sv_ = sorted(list(set(row)))
        cl  = {s:[] for s in sv_}
        for i in range( len( row ) ) :
            cl[row[i]].append(i)
        for v_ in list( cl.values() ) :
            if tuple(v_) not in CL :
                CL[ tuple(v_) ] = d
    return ( CL )


def build_pclist_word_hierarchy ( filename:str         = None   , ledger:dict          = None ,
                                  delete:list[str]     = ['\n'] , group_id_prefix:str  = None ,
                                  analyte_prefix:str   = 'ENSG' , root_name:str        = 'COMP0000000000',
                                  bUseGroupPrefix:bool = False  , bSingleChild:bool    = False ,
                                  bSingleDescent:bool  = True ) -> list :

    bSingleChild = bSingleChild or bSingleDescent # USER SHOULD SET THE bSingleDescent OPTION

    bUseFile = not filename is None
    import operator
    error ( not operator.xor ( filename is None , ledger is None ), "YOU MUST SUPPLY A GMT FILE XOR A DICTIONARY" )
    if bUseFile :
        error ( not '.gmt' in filename , 'MUST HAVE A VALID GMT FILE' )
    #
    # RETURNS THE PC LIST THAT CREATES THE WORD HIERARCHY
    # LATANTLY PRESENT IN THE GMT ANALYTE (GROUPING) DEFINITIONS
    #
    S_M = set()
    D_i = dict()

    bUseGroupPrefix = not group_id_prefix is None
    if bUseGroupPrefix :
        bUseGroupPrefix = 'str' in str(type(group_id_prefix)).lower()
    check_prefix = analyte_prefix
    if bUseGroupPrefix :
        check_prefix = group_id_prefix

    if bUseFile :
        with open ( filename,'r' ) as input :
            for line in input :
                lsp = ordered_remove(line,delete).split('\t')
                if not check_prefix in line :
                    continue
                S_i = set(lsp[2:])
                D_i [ lsp[0] ] = tuple( (lsp[1] , S_i , len(S_i)) )
                S_M = S_M | S_i
    else :
        for item in ledger.items() :
            print(item)
            if bUseGroupPrefix :
                if not check_prefix in item[0]:
                    continue
            else :
                if not check_prefix in ''.join(item[1][1]):
                    continue
            S_i = set( item[1][1] )
            D_i [ item[0] ] = tuple( (item[1][0] , S_i , len(S_i)) )
            S_M = S_M | S_i

    isDecendant  = lambda sj,sk : len(sj-sk)==0
    relative_idx = lambda sj,sk : len(sk-sj)

    parent_id = root_name
    parent_words = S_M

    all_potential_parents = [ [root_name,S_M] , *[ [ d[0],d[1][1]] for d in D_i.items() ] ]

    PClist = []
    CPlist = {}
    for parent_id,parent_words in all_potential_parents:
        lookup    = {}
        for d in D_i .items() :
            if isDecendant ( d[1][1] , parent_words ) :
                Nij = relative_idx ( d[1][1] , parent_words  )
                if Nij in lookup :
                    lookup[Nij] .append(d[0])
                else :
                    lookup[Nij] = [d[0]]
        ledger = sorted ( lookup.items() )

        for ie_ in range( len( ledger ) ) :
            l1 = ledger[ ie_ ][0]
            for potential_child in ledger[ie_][1]:
                pchild_words  = D_i[ potential_child ][1]
                bIsChild      = True
                if potential_child == parent_id :
                    bIsChild  = False
                    break
                check         = [ je_ for je_ in range( ie_ + 1 )] [::-1]
                if len(check) > 0 :
                    for je_ in check :
                        l2 = ledger[ je_ ][0]
                        for relative in ledger[je_][1] :
                            if D_i[relative][0] == D_i[potential_child][0] :
                                continue
                            relative_words = D_i[relative][1]
                            bIsChild = len(relative_words^pchild_words)>0 or (len(relative_words^pchild_words)==0 and l2==l1 )
                            if not bIsChild :
                                break
                if bIsChild :
                    if potential_child in CPlist :
                        if CPlist[potential_child][-1]>relative_idx(pchild_words,parent_words):
                            CPlist[potential_child] = [parent_id , potential_child , relative_idx(pchild_words,parent_words) ]
                    else :
                        CPlist[potential_child] = [parent_id , potential_child , relative_idx(pchild_words,parent_words) ]
                    PClist .append ( [parent_id , potential_child ] )
    D_i[root_name] = tuple( ('full unit',S_M,len(S_M)) )
    pcl_ = []

    if bSingleChild:
        PClist = [ (v[0],v[1]) for k,v in CPlist.items() ]
    return ( [PClist,D_i] )


def scipylinkages ( distm:np.array , command:str='min' , bStrKeys=True ) -> dict :
    from scipy.spatial.distance  import squareform
    from scipy.cluster.hierarchy import linkage as sclinks
    from scipy.cluster.hierarchy import fcluster
    Z = sclinks( squareform(distm) , {'min':'single','max':'complete'}[command] )
    CL = {}
    for d in Z[:,2] :
        row = fcluster ( Z ,d, 'distance' )
        sv_ = sorted(list(set(row)))
        cl  = {s:[] for s in sv_}
        for i in range( len( row ) ) :
            cl[row[i]].append(i)
        for v_ in list( cl.values() ) :
            if tuple(v_) not in CL:
                CL[tuple(v_)] = d
    if bStrKeys :
        L = {}
        for item in CL.items():
            L['.'.join( lint2lstr(item[0])  )] = item[1]
        CL = L
    return ( CL )


def link0_ ( D:np.array , method:str = 'min' ) -> list :
    def func( r:float , c:float , lab:str='min' ) -> float :
        if lab == 'max' :
            return ( r if r > c else c )
        if lab == 'min' :
            return ( r if r < c else c )

    nmind = np.argmin(D) # SIMPLE TIEBREAKER
    ( i,j )  = ( int(nmind/len(D)) , nmind%len(D) )
    k = j - int(i<j)
    l = i - int(j<i)

    pop1 = nppop(D,i,j)
    pop2 = nppop(pop1[-1],k,l)
    lpr  = list(pop2[0])
    d    = lpr.pop(l)
    lpr  = np.array(lpr)
    lpc  = pop2[1]
    nvec = np.array([*[D[0,0]],*[ func(r,c,method) for (r,c) in zip(lpr,lpc) ]])
    DUM  = np.eye(len(nvec))*0
    DUM[ 0  , : ] = nvec
    DUM[ :  , 0 ] = nvec
    DUM[ 1: , 1:] = pop2[-1]
    return ( [ DUM , (i,j) , d ]  )


def linkages_tiers ( D:np.array , method:str = 'min' ) -> dict :
    N   = len(D)
    dm  = np.max(D)*1.1
    idx = list()
    for i in range(N):  D[i,i] = dm ; idx.append( tuple((i,)) )
    cidx     = []
    sidx     = set()
    res      = [D]
    linkages = dict()
    while ( len(res[0]) > 1 ) :
        res          = link0_ ( res[0] , method )
        found_cidx   = tuple( [ idx[i] for i in res[1] ])
        idx = [ *[found_cidx], *[ix_ for ix_ in idx if not ix_ in set(found_cidx) ] ]
        linkages[ found_cidx ] = res[-1]
    for i in range(N) :
        linkages[ (i,) ] = 0
        D[i,i] = 0
    return ( linkages )


def linkages ( distm:np.array , command:str='min' ,
               bStrKeys:bool = True , bUseScipy:bool = False ,
               bMemSec=True ) -> dict :
    distm = np.array(distm)
    if bMemSec :
        distm = distm.copy()
    if bUseScipy :
        linkages_ = scipylinkages ( distm ,command=command , bStrKeys = False )
    else :
        linkages_ = linkages_tiers ( D = distm , method = command )
    if bStrKeys :
        L = {}
        for item in linkages_.items():
            L['.'.join( lint2lstr(item[0])  )] = item[1]
        linkages_ = L
    return ( linkages_ )


