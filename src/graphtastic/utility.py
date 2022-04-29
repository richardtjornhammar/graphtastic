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

def sign ( x:float ) :
    return ( 2*(x>=0)-1 )

def abs ( x:float ):
    return ( sign(x)*x )

contrast   = lambda A,B : ( A-B )/( A+B )
#
# OTHER
e_flatness = lambda x   : np.exp(np.mean(np.log(x),0))/np.mean(x,0)
e_contrast = lambda x   : 1 - e_flatness(x)
#
# SPURIOUS LOW VALUE REMOVAL
confred       = lambda x,eta,varpi : 0.5*x*(1+np.tanh((x-eta)/varpi))*(np.sqrt(x*eta)/(0.5*(eta+x)))
smoothbinred  = lambda x,eta,varpi : 0.5*(1+np.tanh((x-eta)/varpi))*(np.sqrt(x*eta)/(0.5*(eta+x)))
smoothmax     = lambda x,eta,varpi : x * self.smoothbinred(x-np.min(x),eta-np.min(x),varpi)
sabsmax       = lambda x,eta,varpi : x * self.smoothbinred(np.abs(x),np.abs(eta),varpi)


def frac_procentile ( vals:list[float]=[12.123, 1.2, 1000, 4] ):
    vals = np.array(vals).copy()
    N = len( vals );
    for i,j in zip( np.argsort(vals),range(N)):
        vals[i]=(j+0.5)/N
    return ( vals )


def get_procentile ( vals:list[float], procentile:float = 50 ):
    fp_   = procentile/100.0
    proc  = frac_procentile(vals)
    ma,mi = np.max(proc),np.min(proc)
    if fp_ > ma:
        fp_ = ma
    if fp_ < mi:
        fp_ = mi
    idx = np.argwhere(proc==fp_)
    if len ( idx ) == 1 :
        return ( vals[idx[0][0]] )
    else :
        i1 = np.argsort( np.abs(proc-fp_) )[:2]
        return ( sum([vals[i] for i in i1]) * 0.5 )


def padded_rolling_window ( N:int, tau:int ) :
        if tau==1 :
                return ( [ (i,None) for i in range( N ) ] )
        if len(ran)<tau :
                return ( [ (0,N) for v_ in ran ] )
        centered = lambda x:(x[0],x[1]) ;
        w = int( np.floor(np.abs(tau)*0.5) ) ;
        jid = lambda i,w,N:[int((i-w)>0)*((i-w)%N),int(i+w<N)*((i+w)%N)+int(i+w>=N)*(N-1)]
        idx = [ centered( jid(i,w,N) ) for i in range(N) ]
        return ( idx )


def factorial ( n:int ) -> int :
    return ( 1 if n<=0 else factorial(n-1)*n )


def invfactorial ( n:int ) -> float :
    if n<0 :
        return 0
    m = factorial(n)
    return ( 1/m )


def zernicke_amp ( r:float , n:int , m:int ) -> float :
    if ( not (r >= 0 and r <= 1)) or (m > n) :
        return ( 0 )
    def zer_R ( n , m , r ) :
        ip,im = ( n+m )/2 , ( n-m )/2
        z = 0
        for k in range( int( im ) ) :
            f = factorial(n-k)*invfactorial(k)*invfactorial(ip-k)*invfactorial(im-k)
            if f > 0 :
                z = z + (-1)**k * f * r**( n-2*k )
        return ( z )
    Rnm  = zer_R ( n,m,r )
    return ( Rnm )

def error ( message:str , severity:int=0 ):
    print ( errstr )
    if severity > 0 :
        exit(1)
    else :
        return

def seqsum ( c:np.array , n:int = 1 ) -> np.array :
    return ( c[n:] + c[:-n] )


def seqdiff ( c:np.array , n:int = 1 ) -> np.array :
    return ( c[n:] - c[:-n] )


def arr_contrast ( c:np.array , n:int=1, ctyp:str='c' ) -> np.array :
    if ctyp=='c':
        return ( np.append( (c[n:]-c[:-n])/(c[n:]+c[:-n]) ,  np.zeros(n)  ) )
    return ( np.append( np.zeros(n) , (c[n:]-c[:-n])/(c[n:]+c[:-n]) ) )


def all_conts ( c:np.array ) -> np.array :
    s   = c*0.0
    inv = 1.0/len(c)
    for i in range(len(c)):
        s += arr_contrast(c,i+1)*inv
    return ( s )


def mse ( Fs:np.array,Fe:np.array ) -> float :
    return ( np.mean( (Fs-Fe)**2 ) )


def coserr ( Fe:np.array , Fs:np.array ) -> float :
    return ( np.dot( Fe,Fs )/np.sqrt(np.dot( Fe,Fe ))/np.sqrt(np.dot( Fs,Fs )) )


def z2error ( model_data:np.array , evidence_data:np.array , evidence_uncertainties:np.array = None ) -> dict :
    # FOR A DESCRIPTION READ PAGE 71 (57 INTERNAL NUMBERING) of:
    # https://kth.diva-portal.org/smash/get/diva2:748464/FULLTEXT01.pdf
    # EQUATIONS 6.3 AND 6.4
    #
    Fe = evidence_data
    Fs = model_data
    N  = np.min( [ len(evidence_data) , len(model_data) ] )
    if not  len(evidence_data) == len(model_data):
        error ( " THE MODEL AND THE EVIDENCE MUST BE PAIRED ON THEIR INDICES" , 0 )
        Fe  = evidence_data[:N]
        Fs  = model_data[:N]

    dFe = np.array( [ 0.05 for d in range(N) ] )
    if not evidence_uncertainties is None :
        if len(evidence_uncertainties)==N :
            dFe = evidence_uncertainties
        else :
            error ( " DATA UNCERTAINTIES MUST CORRESPOND TO THE TARGET DATA " , 0 )

    def K ( Fs , Fe , dFe ) :
        return ( np.sum( np.abs(Fs)*np.abs(Fe)/dFe**2 ) / np.sum( (Fe/dFe)**2 ) )

    k = K ( Fs,Fe,dFe )
    z2e = np.sqrt(  1/(N-1) * np.sum( ( (np.abs(Fs) - k*np.abs(Fe))/(k*dFe) )**2 )  )
    cer = coserr(Fe,Fs)
    qer = z2e/cer

    return ( {'qerr': qer , 'z2e': z2e , 'cer': cer ,'N':N } )

import math
def isItPrime( N:int , M:int=None,p:int=None,lM05:float=None )->bool :
    if p is None :
        p = 1
    if M is None :
        M = N
    if lM05 is None:
        lM05 = math.log(M)*0.5
    if ((M%p)==0 and p>=2) :
        return ( N==2 )
    else :
       if math.log(p) > lM05:
           return ( True )
       return ( isItPrime(N-1,M=M,p=p+1,lM05=lM05) )


# FIRST APPEARENCE:
# https://gist.github.com/richardtjornhammar/ef1719ab0dc683c69d5a864cb05c5a90
def fibonacci(n):
    if n-2>0:
        return ( fibonacci(n-1)+fibonacci(n-2) )
    if n-1>0:
        return ( fibonacci(n-1) )
    if n>0:
       return ( n )


def f_truth(i): # THE SQUARE SUM OF THE I:TH AND I+1:TH FIBONACCI NUMBER ARE EQUAL TO THE FIBONACCI NUMBER AT POSITION 2i+1
    return ( fibonacci(i)**2+fibonacci(i+1)**2 == fibonacci(2*i+1))


def lint2lstr ( seq:list[int] ) -> list[str] :
    if isinstance ( seq,(list,tuple,set)) :
        yield from ( str(x) for y in seq for x in lint2lstr(y) )
    else :
        yield seq


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
    return ( a )


def smom ( v:np.array , p:int ) -> np.array :
    n = len(v)
    X = [ np.mean(v) ]
    X .append( np.std(v) )
    z = (v-X[0])/X[1]
    if p>2 :
        for q in range( 3 , p+1 ) :
            X.append( np.sum(z**q)/n )
    return ( np.array(X)[:p] )


def nppop(A:np.array, irow:int=None, jcol:int=None ) -> list[np.array] :
    # ASSUMES ROW MAJOR ORDER
    rrow:np.array() = None
    rcol:np.array() = None
    N = len(A)
    M0,M1 = np.shape(A)
    if not irow is None :
        rrow = A[irow,:]
        A    = np.delete(A,range(N*irow,N*(irow+1))).reshape(-1,N)
        M0   = M0-1
    if not jcol is None :
        rcol = A[:,jcol]
        A    = np.delete(A,range(jcol,len(A.reshape(-1)),N) )
        M1   = M1-1
    return ( [rrow,rcol,A.reshape(M0,M1)] )

# Numerical Analysis by Burden and Faires
# GOLUBS SVD PAPER
#
def kth_householder ( A:np.array , k:np.array ) :
    # THE K:TH HOUSHOLDER ITERATION
    A  = np .array( A )
    n_ , m_ = np .shape(A)
    if n_ < 2 :
        return ( A )
    k0 = k
    k1 = k+1

    alpha = ( 2*(A[k1][k0]<0)-1 )
    alpha = alpha * np.sqrt( sum([ a**2 for a in A.T[k0][k1:] ]) )
    r  = np.sqrt ( 0.5*(alpha**2-A[k1][k0]*alpha) )
    v_ = [ 0 for z in range(k1) ] ; v_ .append( (A[k1][k0]-alpha)*0.5/r )
    [ v_ .append ( (0.5/r) * A[j][k0] ) for j in range(k1+1,n_) ]
    v_ = np.array( v_ )
    Pk = np.eye( n_ ) - 2*np.array( [ v*w for v in v_ for w in v_ ] ).reshape(n_,n_)
    Qk = Pk

    if n_ != m_ :
        alpha = ( 2*(A[k0][k1]<0)-1 )
        alpha = alpha * np.sqrt( sum([ a**2 for a in A[k0][k1:] ]) )
        r  = np.sqrt ( 0.5*(alpha**2-A[k0][k1]*alpha) )
        w_ = [ 0 for z in range(k1) ] ; w_ .append( (A[k0][k1]-alpha)*0.5/r )
        [ w_ .append ( (0.5/r) * A[k0][j] ) for j in range(k1+1,m_) ]
        w_ = np.array( w_ )
        Qk = np.eye( m_ ) - 2*np.array( [ v*w for v in w_ for w in w_ ] ).reshape(m_,m_)

    Ak = np.dot( np.dot( Pk,A ),Qk )
    return ( Pk,Ak,Qk )


def householder_reduction ( A:np.array ) :
    A = np.array( A )
    n = np.min( np.shape( A ) )
    if n < 2 :
        return ( A )
    P0 , A0 , Q0 = kth_householder( A,k=0 )
    if n==2 :
        return ( P0 , A0 , Q0.T )
    for k in range( 1 , n-1 ) : # ends at n-2
        P1 , A1 , Q1 = kth_householder( A0,k=k )
        A0 = A1
        P0 = np.dot( P0 , P1 )
        Q0 = np.dot( Q0 , Q1 )
    U  = P0
    S  = A1
    VT = Q0.T
    return ( U , S , VT )


def rich_rot ( a:float , b:float ) :
    if a==0 and b==0 :
        c = 0
        s = 0
        r = 0
    else :
        r = np.sqrt( a*a + b*b )
        if a == 0 :
            s = r / b
            c = 0
        else :
            s = b / r
            c = ( r - s*b ) / a
    return ( c , s , r )


def diagonalize_2b2( A:np.array , tol:float = 1E-13 , maxiter:int = 100 ) :
    M   = A[:2,:2].copy()
    M0  = A[:2,:2].copy()
    k   = 0
    ERR = 1
    G_  = None
    H_  = None
    for k in range( maxiter ) :
        # LEFT
        c,s,r = rich_rot( M0[0,0],M0[1,0])
        G0    = np.array( [[c,s],[-s,c]] )
        M     = np.dot( G0 , M0 )
        # RIGHT
        M     = M.T
        c,s,r = rich_rot( M[0,0],M[1,0])
        H0    = np.array( [[c,s],[-s,c]] )
        M     = np.dot( H0 , M )
        # BUILD
        M0    = M.T
        ERR   = np.sqrt( M0[1,0]**2+M0[0,1]**2 )
        if G_ is None :
            G_ = G0
        else :
            G_ = np.dot(G0,G_)
        if H_ is None :
            H_ = H0
        else :
            H_ = np.dot(H0,H_)
        if ERR < tol :
            break
    return ( G_ , M0 , H_ )

def diagonalize_tridiagonal ( tridiagonal:np.array ,
            maxiter:int = 1000 ,
            tol:float   = 1E-16 ) :

        S       = tridiagonal.copy()
        n_ , m_ = np.shape( S )
        tol22   = tol*0.1
        maxi22  = int( np.ceil( maxiter*0.1 ))

        sI = skew_eye ( [ n_ , n_ ] )
        tI = skew_eye ( [ m_ , m_ ] )
        zI = skew_eye ( np.shape(S) )
        GI = sI.copy()
        HI = tI.copy()
        #
        sI_   = sI.copy()
        tI_   = tI.copy()
        shape = np.shape(S)
        nm_   = shape[0] if (shape[0]<=shape[1]) else shape[1] - 1

        for k in range ( maxiter ) :
            for i in range ( nm_ ) :
                sI_   = sI .copy()
                tI_   = tI .copy()
                A     = S[ i:i+2 , i:i+2 ].copy()
                G , Z , H = diagonalize_2b2 ( A , tol=tol22 , maxiter=maxi22 )
                sI_[ i:i+2 , i:i+2 ] = G
                GI = np.dot( sI_ , GI )
                tI_[ i:i+2 , i:i+2 ] = H
                HI = np.dot( tI_ , HI )
                S =  np.dot( np.dot( sI_ , S ) , tI_.T )
                for ir in range( 2,nm_+1-i ):
                    ii  = i
                    jj  = i+ir
                    idx = [ (ii,ii),(ii,jj),(jj,ii),(jj,jj) ]
                    jdx = [ (0,0),(0,1),(1,0),(1,1) ]
                    A   = np.array( [ S[i] for i in idx] ).reshape(2,2)
                    G , Z , H = diagonalize_2b2 ( A , tol=tol22 , maxiter=maxi22 )
                    sI_ = sI .copy()
                    tI_ = tI .copy()
                    H = H.T
                    for i_,j_ in zip(idx,jdx) :
                        sI_[i_] = G[j_]
                        tI_[i_] = H[j_]
                    tI_= tI_.T
                    GI = np.dot( sI_ , GI )
                    HI = np.dot( tI_ , HI )
                    S =  np.dot( np.dot( sI_ , S ) , tI_.T )
            #ERR = np.sum( S**2*(1-skew_eye([n_,m_]) ) )
            ERR = sum( np.diag(S[:nm_],-1)**2 ) + sum( np.diag(S[:nm_] ,1)**2 )
            if ERR < tol :
                break;
        # RETURNS THE MATRICES NEEDED TO CREATE THE INPUT DATA
        # WHERE R[1] IS THE SINGULAR VALUE VECTOR
        # DATA = np.dot( np.dot( R[0],R[1]),R[2] )
        return ( GI.T , S , HI )


def anSVD ( A:np.array , maxiter=1000 , tol=1E-30 ):
    P , Z , QT = householder_reduction( A )
    G , S , HT = diagonalize_tridiagonal( Z , maxiter=maxiter , tol=tol )
    U  = np.dot(P,G)
    VT = np.dot(HT,QT)
    return ( U,S,VT )


def seqdot( B:np.array ) :
    if len(B)  > 2 :
        return ( np.dot( B[0] , seqdot( B[1:] ) ) )
    if len(B)  > 1 :
        return ( np.dot( B[0] , B[1] ) )
    return ( B[0] )


def eigensolve_2b2 ( M:np.array ) :
    # MOHRS LILLA OLLE I SKOGEN GICK ...
    s1      = M[0,0]
    s2      = M[1,1]

    tau2    = M[1,0] * M[0,1]
    delta   = M[0,0] - M[1,1]
    phi     = M[0,0] + M[1,1]

    xi      = np.sqrt( delta**2+4*tau2 )
    lambda0 = 0.5*( phi + xi )
    lambda1 = 0.5*( phi - xi )
    tau01   = M[0,1]
    tau10   = M[1,0]

    def transf ( tau, delta, xi , pm=1 ) :
        nom0 = 0.5 * ( delta + pm*xi )/tau
        nom1 = 1
        c,s  = nom0 , nom1
        norm = np.sqrt(c*c+s*s)
        c    = c / norm
        s    = s / norm
        return ( np.array([[c,s],[s,-c]]) )

    e10p = transf ( tau=tau10 , delta=delta , xi=xi , pm =  1 )
    e10m = transf ( tau=tau10 , delta=delta , xi=xi , pm = -1 )

    return ( np.array([lambda0,lambda1]),e10p[0],e10m[0] )

import re
def find_category_variables( istr ) :
    return ( re.findall( r'C\((.*?)\)', istr ) )

def find_category_interactions ( istr:str ) :
    all_cats = re.findall( r'C\((.*?)\)', istr )
    interacting = [ ':' in c for c in istr.split(')') ][ 0:len(all_cats) ]
    interacting_categories = [ [all_cats[i-1],all_cats[i]] for i in range(1,len(interacting)) if interacting[i] ]
    return ( interacting_categories )

def subArraysOf ( Array:list,Array_:list=None ) -> list :
    if Array_ == None :
        Array_ = Array[:-1]
    if Array == [] :
        if Array_ == [] :
            return ( [] )
        return( subArraysOf(Array_,Array_[:-1]) )
    return([Array]+subArraysOf(Array[1:],Array_))

def permuter( inputs:list , n:int ) -> list :
    # permuter( inputs = ['T2D','NGT','Female','Male'] , n = 2 )
    return( [p[0] for p in zip(itertools.permutations(inputs,n))] )

def grouper ( inputs, n ) :
    iters = [iter(inputs)] * n
    return zip ( *iters )
