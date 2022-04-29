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

import typing

def read_xyz( fname:str = 'argon.xyz', sep:str = ' ') -> list :
    coords = []
    with open(fname,'r') as input:
        for line in input:
            lsp = [u for u in line.replace('\n','').split(sep) if len(u)>0 ]
            print(lsp,len(lsp))
            if len(lsp) == 4:
                coords.append( ( lsp[0],[ float(c) for c in lsp[1:]] ) )
    return ( coords )

def encode_categorical( G:list = ['Male','Male','Female'] )-> dict :
    #
    # CREATES AN BINARY ENCODING MATRIX FROM THE SUPPLIED LIST
    #
    ugl = list(set(G)) ; n = len(ugl) ; m = len(G)
    lgu = { u:j for u,j in zip(ugl,range(n)) }
    enc_d = np.zeros(n*m).reshape(m,n)
    for i in range ( m ) :
        j = lgu[G[i]]
        enc_d[i,j] = 1
    return ( { 'encoding':enc_d , 'names':ugl } )
