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
import sys

import typing

class Node ( object ) :
    def __init__ ( self ) :
        self.id_          :str   = ""
        self.label_       :str   = ""
        self.description_ :str   = ""
        self.level_       :int   = 0      # NODES ARE MYOPIC
        self.metrics_     :list  = list()
        self.links_       :list  = list()
        self.ascendants_  :list  = list() # INWARD LINKS  , DIRECT ASCENDENCY  ( 1 LEVEL )
        self.descendants_ :list  = list() # OUTWARD LINKS , DIRECT DESCENDENCY ( 1 LEVEL )
        self.data_        :dict  = dict() # OTHER THINGS SHOULD BE ALL INFORMATION FLOATING IN USERSPACE

    def can_it_be_root ( self, n:int=1 ) -> bool :
        return ( len(self.ascendants_) < n )

    def supplement ( self, n:super ) -> None :
        self.label_       = n.label_
        self.description_ = n.description_
        self.level_       = n.level_
        self.metrics_     = [ *self.metrics_     , *n.metrics_     ]
        self.links_       = [ *self.links_       , *n.links_       ]
        self.ascendants_  = [ *self.ascendants_  , *n.ascendants_  ]
        self.descendants_ = [ *self.descendants_ , *n.descendants_ ]
        self.data_        = { **self.data_, **n.data_ }

    def assign_all ( self, identification : str ,
                           links : type(list(str())) ,
                           label : str = "" ,
                           description : str = "" ) -> object :
        # ASSIGNS ALL META DATA AND BIPOLAR LINKS
        self.set_id( identification )
        self.add_label( label )
        self.add_description( description )
        self.add_links( links , bClear=True )
        return ( self )

    def set_level ( self,level:int ) -> None :
        self.level_ = level

    def set_metrics ( self , metrics:list ) -> None :
        self.metrics_ = [ *self.metrics_ , *metrics ]

    def level ( self ) -> None :
        return ( self.level_ )

    def get_data ( self ) -> dict :
        return ( self.data_ )

    def overwrite_data ( self, data:dict ) -> None :
        self.data_ = data

    def set_id ( self, identification:str ) -> None :
        self.id_ = identification

    def add_label ( self, label : str ) -> None :
        self.label_ = label

    def add_description ( self, description : str ) -> None :
        self.description_ = description

    def identification ( self ) -> str :
        return ( self.id_ )

    def label ( self ) -> str :
        return ( self.label_ )

    def description ( self ) -> str :
        return ( self.description_ )

    def add_link ( self, identification:str , bClear:bool = False , linktype:str = 'links' ) -> None :
        edges = self.get_links( linktype )
        if bClear :
            edges = list()
        edges .append( identification )
        self.links_ = list(set( edges ))
    #
    # NOTE : list[str] type declaration is not working in Python3.8
    def add_links ( self, links:type(list(str())), bClear:bool = False , linktype:str = 'links' ) -> None :
        edges = self.get_links( linktype )
        if bClear :
            edges = list()
        for e in links :
            edges .append ( e )
        self.links_ = list(set( edges ))

    def get_links ( self , linktype:str='links' ) -> type(list(str())) :
        if not linktype in set([ 'links' , 'ascendants' , 'descendants' ]):
            print ( ' \n\n!!FATAL!!\t' + ', '.join([ 'links' , 'ascendants' , 'descendants' ]) \
                  + '\t ARE THE ONLY VALID EDGE TYPES (linktype)' )
            exit ( 1 )
        if linktype == 'links' :
            return ( self.links_ )
        if linktype == 'ascendants' :
            return ( self.ascendants_  )
        if linktype == 'descendants' :
            return ( self.descendants_ )

    def show ( self ) -> None :
        s_inf = "NODE [" + self.identification() \
                   + "," + self.label() + "] - " \
                   + self.description() + "\nEDGES:"
        for linktype in [ 'links' , 'ascendants' , 'descendants' ] :
            s_inf += '\n['+linktype+'] : '
            for l in self.get_links(linktype=linktype) :
                s_inf += l + '\t'
        for item in self.get_data().items() :
            s_inf += '\n'+str(item[0])+'\t'+str(item[1])
        print ( s_inf )

class NodeGraph ( Node ) :
    # https://github.com/richardtjornhammar/RichTools/commit/c4b9daa78f2a311995d142b0e74fba7c3fdbed20#diff-0b990604c2ec9ebd6f320ebe92099d46e0ab8e854c6e787fac2f208409d112d3
    def __init__( self ) :
        self.root_id_       = ''
        self.desc_          = "SUPPORTS DAGS :: NO STRUCTURE ASSERTION"
        self.num_edges_     = 0
        self.num_vertices_  = 0
        self.graph_map_     = dict()

    def keys ( self )   -> list :
        return( self.graph_map_.keys() )

    def values ( self ) -> list :
        return( self.graph_map_.values() )

    def items ( self )  -> list :
        return( self.graph_map_.items() )

    def list_roots ( self ) ->  type(list(str())) :
        roots = [] # BLOODY ROOTS
        for name,node in self.items():
            if node.can_it_be_root() :
                roots.append( name )
        return ( roots )

    def get_node ( self, nid : str ) -> Node :
        return ( self.graph_map_[nid] )

    def set_root_id ( self, identification : str ) -> None :
        self.root_id_ = identification

    def get_root_id ( self ) -> str :
        return ( self.root_id_ )

    def add ( self, n : Node ) -> None :
        if n.identification() in self.graph_map_ :
            self.graph_map_[ n.identification() ].supplement( n )
        else :
            self.graph_map_[ n.identification() ] = n
            if len ( self.graph_map_ ) == 1 :
                self.set_root_id( n.identification() )

    def get_dag ( self ) -> dict :
        return ( self.graph_map_ )

    def get_graph ( self ) -> dict :
        return ( self.graph_map_ )

    def show ( self ) -> None :
        print ( self.desc_ )
        for item in self.get_dag().items() :
            print ( '\n' + item[0] + '::' )
            item[1].show()

    def complete_lineage ( self , identification : str ,
                           order:str    = 'depth'      ,
                           linktype:str = 'ascendants' ) -> dict :
        # 'ascendants' , 'descendants'
        root_id = identification
        results = self.search( order=order , root_id=identification , linktype=linktype )
        results['path'] = [ idx for idx in results['path'] if not idx==identification ]
        return ( results )

    def search ( self , order:str = 'breadth', root_id:str = None ,
                 linktype:str = 'links', stop_at:str = None ) -> dict :
        #
        path:list   = list()
        visited:set = set()
        if root_id is None :
            root_id = self.get_root_id()
        S:list      = [ root_id ]
        if not order in set(['breadth','depth']) :
            print ( 'order MUST BE EITHER breadth XOR depth' )
            exit ( 1 )

        if order == 'breadth' :
            while ( len(S)>0 ) :
                v = S[0] ; S = S[1:]
                ncurrent:Node = self.get_node(v)
                visited       = visited|set([v])
                path.append( ncurrent.identification() )
                #
                # ADDED STOP CRITERION FOR WHEN THE STOP NODE IS FOUND
                if not stop_at is None :
                    if stop_at == v :
                        S = []
                        break
                links         = ncurrent.get_links(linktype)
                for w in links :
                    if not w in visited and len(w)>0:
                        S.append( w ) # QUE

        if order == 'depth' :
            while ( len(S)>0 ) :
                v = S[0] ; S = S[1:]
                if not v in visited and len(v)>0 :
                    visited       = visited|set([v])
                    ncurrent:Node = self.get_node(v)
                    links         = ncurrent.get_links(linktype)
                    for w in links :
                        if not w in visited and len(w)>0:
                            S = [*[w],*S] # STACK
                    path.append( ncurrent.identification() )
                    #
                    # ADDED STOP CRITERION FOR WHEN THE STOP NODE IS FOUND
                    if not stop_at is None :
                        if stop_at == v :
                            S = []
                            break

        return ( { 'path':path , 'order':order , 'linktype':linktype } )

    def connectivity ( self, distm:np.array , alpha:float , n_connections:int=1 , bOld:bool=True ) -> list :
        #
        # AN ALTERNATIVE METHOD
        # DOES THE SAME THING AS THE CONNECTIVITY CODE IN MY
        # CLUSTERING MODULE (in src/impetuous/clustering.py )
        # OR IN https://github.com/richardtjornhammar/RichTools/blob/master/src/cluster.cc
        # ADDED TO RICHTOOLS HERE: https://github.com/richardtjornhammar/RichTools/commit/74b35df9c623bf03570707a24eafe828f461ed90#diff-25a6634263c1b1f6fc4697a04e2b9904ea4b042a89af59dc93ec1f5d44848a26
        # CONNECTIVITY SEARCH FOR (connectivity) CONNECTIVITY
        #
        # THIS ROUTINE RETURNS A LIST BELONGING TO THE CLUSTERS
        # WITH THE SET OF INDICES THAT MAPS TO THE CLUSTER
        # THIS METHOD IS NOW ALSO IN THE clustering.py MODULE
        # AND IS CALLED connectedness
        # THIS CLASS WILL EMPLOY THE JIT connectivity IMPLEMENTATION
        # IN THE FUTURE BECAUSE IT IS SUPERIOR
        #
        if len ( distm.shape ) < 2 :
            print ( 'PLEASE SUBMIT A SQUARE DISTANCE MATRIX' )
            exit(1)
        #
        if bOld : # WATER CLUSTERING ALGO FROM 2009
            from graphtastic.clustering import connectivity as connections
            results = connections ( distm , alpha )
            L = [set() for i in range(len(results[0]))]
            for c in results['clustercontacts'] :
                L[c[0]] = L[c[0]]|set([c[1]])
            return ( L )
        #
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

    def distance_matrix_to_pclist ( self , distm:np.array ,
                                    cluster_connections:int = 1 ,
                                    hierarchy_connections:int = 1 ,
                                    bNonRedundant:bool = True  ) -> list :
        #
        # FASTER PCLIST CONSTRUCTION ROUTINE
        # RETURNS LIST USEFUL FOR HIERARCHY GENERATION
        # SHOULD BE EASIER TO PARALLELIZE WITH JIT
        # lambda p:set -> bool INVALID SYNTAX #
        logic = lambda p,c : len(p&c) >= hierarchy_connections and len(p^c)>0
        if not bNonRedundant :
            logic = lambda p,c : len(p&c) >= hierarchy_connections
        #
        nm = np.shape(distm)
        if not len ( nm ) == 2 :
            print ( "DISTANCE MATRIX MUST BE A SQUAREFORM MATRIX" )
            exit(1)
        if not nm[0] == nm[1]:
            print ( "DISTANCE MATRIX MUST BE A SQUAREFORM MATRIX" )
            exit(1)
        R = sorted( list(set( distm.reshape(-1) ) ) )
        prev_clusters = []
        PClist = []
        for r in R :
            # NOTE: THE LOCAL ROUTINE
            present_clusters = self.connectivity ( distm , r , cluster_connections )
            parent_child  = [ (p,c,r) for c in prev_clusters for p in present_clusters \
                          if logic(p,c)  ]
            prev_clusters = present_clusters
            PClist = [ *PClist, *parent_child ]
        return ( PClist )

    def distance_matrix_to_absolute_coordinates ( self , D:np.array , bSquared:bool = False, n_dimensions:int=2 , bLocal:bool=True ) -> np.array :
        #
        # SAME AS IN THE IMPETUOUS cluster.py EXCEPT THE RETURN IS TRANSPOSED
        # AND distg.m IN THE RICHTOOLS REPO
        # C++ VERSION HERE https://github.com/richardtjornhammar/RichTools/commit/be0c4dfa8f61915b0701561e39ca906a9a2e0bae
        #
        if not bLocal : # JITTED VERSION
            from graphtastic.fit import distance_matrix_to_absolute_coordinates as distm2coords
            return ( distm2coords ( D , bSquared = bSquared, n_dimensions=n_dimensions ) )

        if not bSquared :
            D = D**2.
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

    def distance_matrix_to_graph_dag ( self , distm:np.array , n_:int=1 , bVerbose:bool=False , names:list=None ) -> None :
        #
        # CONSTRUCTS THE HIERACHY FROM A DISTANCE MATRIX
        # SIMILAR TO THE ROUTINES IN hierarchical.py IN THIS IMPETUOUS REPO
        #
        if len ( distm.shape ) < 2 :
            print ( 'PLEASE SUBMIT A SQUARE DISTANCE MATRIX' )
            exit(1)
        lookup = dict()
        m_ = len(distm)
        for I in range(m_) :
            lookup[I] = I
        if not names is None :
            if len ( names ) == m_ :
                for I,N in zip(range(len(names)),names):
                    lookup[I] = N
        pclist = self.distance_matrix_to_pclist( distm )
        for pc_ in pclist :
            lpc0 = [ lookup[l] for l in list(pc_[0]) ]
            lpc1 = [ lookup[l] for l in list(pc_[1]) ]
            asc = str(lpc0)
            des = str(lpc1)
            asc_met = pc_[2]
            self.add_ascendant_descendant(asc,des)
            self.get_graph()[asc].set_metrics([asc_met])
            self.get_graph()[asc].get_data()['analyte ids'] = lpc0
            self.get_graph()[des].get_data()['analyte ids'] = lpc1
        for key in self.keys() :
            if self.get_graph()[key].can_it_be_root(n_):
                self.set_root_id ( key )
        if bVerbose :
            self.show()
            print ( self.get_root_id() )
            self.get_graph()[self.get_root_id()].show()

    def graph_analytes_to_approximate_distance_matrix ( self ,
             analyte_identifier:str = 'analyte ids',
             alpha:float = 1. ) -> np.array :

        root_data    = self.get_graph()[ self.get_root_id() ].get_data()

        if analyte_identifier in root_data:
            all_analytes = root_data[ analyte_identifier ]
        else:
            print ( 'ERROR COULD NOT FIND GLOBAL IDENTIFIER INFORMATION:' , analyte_identifier )
            exit (1)
        m_ = len( all_analytes )
        lookup = { a:i for a,i in zip(all_analytes,range(m_)) }
        CM = np.ones(m_*m_).reshape(m_,m_)
        for item in self.get_graph().items() :

            item_data = item[1].get_data()

            if analyte_identifier in item_data : # IMPROVE SPEED HERE
                for q in item_data[analyte_identifier] :
                    for p in item_data[analyte_identifier] : # STOP
                        CM[lookup[p],lookup[q]]+=1
        #
        # CONSTRUCT APPROXIMATION
        # LEVELS MISSING EVEN VIA DEFAULT
        approximate_distm  = 1./CM - np.mean(np.diag(1./CM))
        approximate_distm *= 1-np.eye(m_)
        return ( np.abs(approximate_distm) , lookup )

    def add_ascendant_descendant ( self, ascendant:str, descendant:str ) -> None :
        n = Node()
        n.set_id(ascendant)
        n.add_label("")
        n.add_description("")
        n.add_links([descendant],linktype='links'       )
        n.add_links([descendant],linktype='descendants' )
        m = Node()
        m.set_id(descendant)
        m.add_label("")
        m.add_description("")
        m.add_links([ascendant],linktype='links'      )
        m.add_links([ascendant],linktype='ascendants' )
        self.add(n)
        self.add(m)

    def generate_ascendants_descendants_lookup ( self ) -> (type(list(str())),type(list(str()))) :
        all_names   = self.keys()
        descendants = [ ( idx , set( self.complete_lineage( idx,linktype='descendants')['path'] ) ) for idx in all_names ]
        ancestors   = [ ( idx , set( self.complete_lineage( idx,linktype='ascendants' )['path'] ) ) for idx in all_names ]
        return ( ancestors , descendants )

    def ascendant_descendant_file_to_dag ( self, relationship_file:str = './PCLIST.txt' ,
                                  i_a:int = 0 , i_d:int = 1 ,
                                  identifier:str = None , sep:str = '\t' ) -> (type(list(str())),type(list(str()))) :

        with open ( relationship_file,'r' ) as input :
            for line in input :
                if not identifier is None :
                    if not identifier in line :
                        continue

                lsp = line.replace('\n','').split( sep )
                ascendant  = lsp[i_a].replace('\n','')
                descendant = lsp[i_d].replace('\n','')

                self.add_ascendant_descendant( ascendant , descendant )

        ancestors , descendants = self.generate_ascendants_descendants_lookup()

        return ( ancestors , descendants )


    def calculate_node_level( self, node:Node , stop_at:str = None , order:str='depth' ) -> None :
        note__ = """
         SEARCHING FOR ASCENDANTS WILL YIELD A
         DIRECT PATH IF A DEPTH SEARCH IS EMPLOYED.
         IF THERE ARE SPLITS ONE MUST BREAK THE SEARCH.
         SPLITS SHOULD NOT BE PRESENT IN ASCENDING DAG
         SEARCHES. SPLIT KILLING IS USED IF depth AND
         stop_at ARE SPECIFIED. THIS CORRESPONDS TO
         DIRECT LINEAGE INSTEAD OF COMPLETE.
        """
        level = len( self.search( root_id=node.identification(), linktype='ascendants', order=order, stop_at=stop_at )['path'] ) - 1
        node.set_level( level )


    def hprint ( self, node:Node, visited:set,
                 I:int = 0, outp:str = "" , linktype:str = "descendants",
                 bCalcLevel = True ) -> (str,int) :
        I = I+1
        if bCalcLevel :
            self.calculate_node_level( node, stop_at = self.get_root_id() )

        head_string   = "{\"source\": \"" + node.identification() + "\", \"id\": " + str(I)
        head_string   = head_string + ", \"level\": " + str(node.level())
        desc_         = str(node.description())
        if len( desc_ ) == 0 :
            desc_ = "\"\""
        head_string   = head_string + ", \"description\": " + desc_
        dat = node.get_data().items()
        if len( dat )>0 :
            for k,v in dat :
                sv = str(v)
                if len(sv) == 0 :
                    sv = "\"\""
                head_string   = head_string + ", \"" + str(k) + "\": " + sv
        desc_h_str    = ", \"children\": ["
        desc_t_str    = "]"
        tail_string   = "}"

        visited = visited|set( [node.identification()] )
        outp    = outp + head_string
        links   = node.get_links(linktype)
        for w in links :
            if not w in visited and len(w)>0 :
                outp = outp + desc_h_str
                outp,I = self.hprint ( self.get_node(w), visited, I, outp, linktype  )
                outp = outp + desc_t_str
        outp = outp + tail_string
        return ( outp,I )

    def write_json ( self , jsonfile:str = 'rtree.json', bCalcLevel:bool = True ,
                     linktype:str = 'descendants', root_id:str = None ) -> str :
        I:int = 1
        if root_id is None :
            root_id = self.get_root_id()
        v = root_id
        node:Node = self.get_node(v)
        visited   = set()
        json_data_txt,I = self.hprint( node, visited,
                                       linktype   = linktype,
                                       bCalcLevel = bCalcLevel )
        of_ = open(jsonfile,'w')
        print ( json_data_txt,file=of_ )
        return( json_data_txt )


def ascendant_descendant_to_dag ( relationship_file:str = './PCLIST.txt' ,
                                  i_a:int = 0 , i_d:int = 1 ,
                                  identifier:str = None , sep:str = '\t' ) -> NodeGraph :
    RichTree = NodeGraph()
    ancestors , descendants = RichTree.ascendant_descendant_file_to_dag( relationship_file = relationship_file ,
        i_a = i_a , i_d = i_d ,identifier = identifier , sep = sep )

    return ( RichTree , ancestors , descendants )


def write_tree( tree:NodeGraph , outfile='tree.json', bVerbose=True ):
    if bVerbose:
        print ( 'YOU CAN CALL THE NodeGraph METHOD tree.write_json() FUNCTION DIRECTLY' )
    o_json = tree.write_json( outfile )
    return ( o_json )

def add_attributes_to_tree ( p_df , tree ):
    add_attributes_to_node_graph ( p_df , tree )
    return ( tree )

def parent_child_to_dag ( relationship_file:str = './PCLIST.txt' ,
             i_p:int = 0 , i_c:int = 1 , identifier:str = None ) :

    return ( ascendant_descendant_to_dag ( relationship_file = relationship_file,
                                      i_a = i_p , i_d = i_c , identifier=identifier ) )

def value_equalisation( X:np.array , method:str='average' ) -> np.array :
    X_ = (rankdata( X , method=method )-0.5)/len(set(X))
    return ( X_ )



if __name__ == '__main__' :
    contributions__ = { "Richard Tjörnhammar": "All methods" }
