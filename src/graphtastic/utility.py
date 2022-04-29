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


def get_procentile ( vals:list[flost], procentile:float = 50 ):
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

    return ( {'qerr'. qer , 'z2e': z2e , 'cer': cer ,'N':N } )

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
