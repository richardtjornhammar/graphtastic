import numpy as np
import itertools
import typing

class Qvalues ( object ) :
    def __init__( self, pvalues:np.array , method:str = "UNIQUE" , pi0:np.array = None ) :
        from scipy.stats import rankdata
        self.rankdata = rankdata
        self.method   : str      = method
        self.pvalues  : np.array = pvalues
        self.qvalues  : np.array = None
        self.qpres    : np.array = None
        if method == "FDR-BH" :
            self.qpres = self.qvaluesFDRBH  ( self.pvalues )
        if method == "QVALS"  :
            self.qpres = self.qvaluesFDRBH  ( self.pvalues , pi0 )
        if method == "UNIQUE" :
            self.qpres = self.qvaluesUNIQUE ( self.pvalues , pi0 )

    def __str__ ( self ) :
        return ( self.info() )

    def __repr__( self ) :
        return ( self.info() )

    def help ( self ) :
        desc__ = "\n\nRANK CORRECTION FOR P-VALUES\nVIABLE METHODS ARE method = FDR-BH , QVALS , UNIQUE\n\n EMPLOYED METHOD: " + self.method
        return ( desc__ )

    def info ( self ) :
        desc__ = "\nMETHOD:"+self.method+"\n   q-values       \t     p-values\n"
        return ( desc__+'\n'.join( [ ' \t '.join(["%10.10e"%z for z in s]) for s in self.qpres ] ) )

    def get ( self ) :
        return ( self.qpres )

    def qvaluesFDRBH ( self , p_values_in:np.array = None , pi0:np.array = None ) :
        p_s = p_values_in
        if p_s is None :
            p_s = self.pvalues
        m = int(len(p_s))
        if pi0 is None :
            pi0 = np.array([1. for i in range(m)])
        qs_ = []
        ps = p_s
        frp_ = (self.rankdata( ps,method='ordinal' )-0.5)/m
        ifrp_ = [ ( (p<=f)*f + p*(p>f) ) for p,f in zip(ps,frp_) ]
        for ip,p0 in zip(range(m),pi0) :
            p_ = ps[ ip ] ; f_ = frp_[ip]
            q_ = p0 * p_ / ifrp_[ip]
            qs_.append( (q_,p_) )
        self.qvalues = np.array([q[0] for q in qs_])
        return np.array(qs_)

    def qvaluesUNIQUE ( self , p_values_in = None , pi0 = None ) :
        p_s = p_values_in
        if p_s is None :
            p_s = self.pvalues
        m = int(len(set(p_s)))
        n = int(len(p_s))
        if pi0 is None :
            pi0 = np.array([1. for i in range(n)])
        qs_ = []
        ps  = p_s
        frp_  = (self.rankdata( ps,method='average' )-0.5)/m
        ifrp_ = [ ( (p<=f)*f + p*(p>f) ) for p,f in zip(ps,frp_) ]
        for ip,p0 in zip( range(n),pi0 ) :
            p_ = ps[ ip ] ; f_ = frp_[ip]
            q_ = p0 * p_ / ifrp_[ip]
            qs_.append( (q_,p_) )
        self.qvalues = np.array([q[0] for q in qs_])
        return np.array(qs_)

class Pvalues ( object ) :
    def __init__( self, data_values:np.array , method:str = "NORMAL" ) :
        from scipy.stats import rankdata
        self.rankdata = rankdata
        self.method   : str        = method
        self.dvalues  : np.array   = data_values
        self.pvalues  : np.array   = None
        self.dsdrvalues : np.array = None
        self.dpres    : np.array   = None
        if method == "RANK DERIV E" :
            self.dpres = self.pvalues_dsdr_e ( self.dvalues , True)
        if method == "RANK DERIV N" :
            self.dpres = self.pvalues_dsdr_n ( self.dvalues , True )
        if method == "NORMAL" :
            self.dpres = self.normal_pvalues ( self.dvalues , True )
        self.pvalues = self.dpres[0]

    def __str__ ( self ) :
        return ( self.info() )

    def __repr__( self ) :
        return ( self.info() )

    def help ( self ) :
        desc__ = "\n\nRANK DERIVATIVE P-VALUES\nVIABLE METHODS ARE method = NORMAL, RANK DERIV E, RANK DERIV N \n\n EMPLOYED METHOD: " + self.method
        return ( desc__ )

    def info ( self ) :
        desc__ = "\nMETHOD:"+self.method+"\n   p-values       \t     ds-values\n"
        return ( desc__+'\n'.join( [ ' \t '.join(["%10.10e"%z for z in s]) for s in self.dpres.T ] ) )

    def get ( self ) :
        return ( self.qpres )

    def sgn ( self, x:float) -> int :
        return( - int(x<0) + int(x>=0) )
    
    def nn ( self, N:int , i:int , n:int=1 )->list :
        t = [(i-n)%N,(i+n)%N]
        if i-n<0 :
            t[0] = 0
            t[1] += n-i
        if i+n>=N :
            t[0] -= n+i-N
            t[1] = N-1
        return ( t )

    def normal_pvalues ( self, v:np.array , bReturnDerivatives:bool=False  ) -> np.array :
        ds = v # TRY TO ACT LIKE YOU ARE NORMAL ...
        N = len(v) 
        M_ , Var_ = np.mean(ds) , np.std(ds)**2
        from scipy.special import erf as erf_
        loc_Q   = lambda X,mean,variance : [ 1. - 0.5*( 1. + erf_(  (x-mean)/np.sqrt( 2.*variance ) ) ) for x in X ]
        rv = loc_Q ( ds,M_,Var_ )
        if bReturnDerivatives :
            rv = [*rv,*ds ]
        return ( np.array(rv).reshape(-1,N) )

    def pvalues_dsdr_n ( self, v:np.array ,
                         bReturnDerivatives:bool=False ,
                         bSymmetric:bool=True ) -> np.array :
        #
        N = len(v)
        vsym = lambda a,b : a*self.sgn(a) if b else a
        import scipy.stats as st
        rv = st.rankdata(v,'ordinal') - 1
        vr = { int(k):v for k,v in zip(rv,range(len(rv)))}
        ds = []
        for w,r in zip(v,rv) :
            nr  = self.nn(N,int(r),1)
            nv  = [ vr[j] for j in nr ]
            s_  = [ v[j] for j in sorted(list(set( [ *[vr[int(r)]] , *nv ] )) ) ]
            dsv = np.mean( np.diff(s_) )
            ds.append( vsym( dsv , bSymmetric) ) # DR IS ALWAYS 1
        M_,Var_ = np.mean(ds) , np.std(ds)**2
        from scipy.special import erf as erf_
        loc_Q   = lambda X,mean,variance : [ 1. - 0.5*( 1. + erf_(  (x-mean)/np.sqrt( 2.*variance ) ) ) for x in X ]
        rv = loc_Q ( ds,M_,Var_ )
        if bReturnDerivatives :
            rv = [*rv,*ds ]
        return ( np.array(rv).reshape(-1,N) )

    def pvalues_dsdr_e ( self, v:np.array ,
                         bReturnDerivatives:bool=False ,
                         bSymmetric:bool=True ) -> np.array :
        #
        N = len(v)
        vsym = lambda a,b : a*self.sgn(a) if b else a
        import scipy.stats as st
        rv = st.rankdata(v,'ordinal') - 1
        vr = { int(k):v for k,v in zip(rv,range(len(rv)))}
        ds = []
        for w,r in zip(v,rv) :
            nr  = self.nn(N,int(r),1)
            nv  = [ vr[j] for j in nr ]
            s_  = [ v[j] for j in sorted(list(set( [ *[vr[int(r)]] , *nv ] )) ) ]
            dsv = np.mean( np.diff(s_) )
            ds.append( vsym( dsv , bSymmetric) ) # DR IS ALWAYS 1
        M_ = np.mean ( ds )
        loc_E  = lambda X,L_mle : [ np.exp(-L_mle*x) for x in X ]
        ev = loc_E ( ds,1.0/M_)   # EXP DISTRIBUTION P
        if bReturnDerivatives :
            rv = [*ev,*ds ]
        return ( np.array(rv).reshape(-1,N) )

import scipy.stats as st
def group_significance (  GroupAnalytes:list[str] , SigAnalytes:list[str] ,
                          AllAnalytes:list[str]   , alternative:str = 'two-sided' ) :
    # FISHER ODDS RATIO CHECK
    # CHECK FOR ALTERNATIVE :
    #   'greater'   ( ENRICHMENT IN GROUP )
    #   'two-sided' ( DIFFERENTIAL GROUP EXPERSSION )
    #   'less'      ( DEPLETION IN GROUP )
    SigAnalytes    = set(SigAnalytes)
    AllAnalytes    = set(AllAnalytes)
    Analytes       = set(GroupAnalytes)

    notAnalytes    = AllAnalytes - Analytes
    notSigAnalytes = AllAnalytes - SigAnalytes
    AB  = len(Analytes&SigAnalytes)    ; nAB  = len(notAnalytes&SigAnalytes)
    AnB = len(Analytes&notSigAnalytes) ; nAnB = len(notAnalytes&notSigAnalytes)
    oddsratio , pval = st.fisher_exact([[AB, nAB], [AnB, nAnB]], alternative=alternative )
    return ( pval , oddsratio )      

from scipy.stats import rankdata
def quantify_density_probability ( rpoints:np.array , bMore:bool=False ) :
    # USING NORMAL DISTRIBUTION
    # DETERMINE P VALUES
    loc_pdf = lambda X,mean,variance : [ 1./np.sqrt(2.*np.pi*variance)*np.exp(-((x-mean)/(2.*variance))**2) for x in X ]
    from scipy.special import erf as erf_
    loc_cdf = lambda X,mean,variance : [      0.5*( 1. + erf_(  (x-mean)/np.sqrt( 2.*variance ) ) ) for x in X ]
    loc_Q   = lambda X,mean,variance : [ 1. - 0.5*( 1. + erf_(  (x-mean)/np.sqrt( 2.*variance ) ) ) for x in X ]
    M_,Var_ = np.mean(rpoints),np.std(rpoints)**2
    #
    # INSTEAD OF THE PROBABILTY DENSITY WE RETURN THE FRACTIONAL RANKS
    # SINCE THIS ALLOWS US TO CALCULATE RANK STATISTICS FOR A PROJECTION
    n = len(set(rpoints)) 
    corresponding_density = (rankdata (rpoints,'average') -0.5) / n
    corresponding_pvalue  = loc_Q  ( rpoints,M_,Var_ )
    if bMore:
        cor_pdf = loc_pdf  ( rpoints,M_,Var_ )
        cor_cdf = loc_cdf  ( rpoints,M_,Var_ )
    #
    if bMore:
        return ( corresponding_pvalue , corresponding_density , cor_pdf , cor_cdf )
    return ( corresponding_pvalue , corresponding_density )
