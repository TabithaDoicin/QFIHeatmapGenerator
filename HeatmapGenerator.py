from mpmath import mpf
from mpmath import mp
import mpmath
import sys as s
import numpy as np
import scipy as sp
import pandas as pd
import time
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import pickle
from pathlib import Path
import math

mp.dps = 50

Dg=1
De=100
M = np.min([Dg,De])
N = np.max([Dg,De])
wc = 1
wa = 0.25

ep1=0.2
ep2=0.2

minT=1e-7
maxT=1e3
numT=200
Tlist = np.geomspace(minT, maxT, numT)

gprefactor=0.8 * 1/(np.sqrt(100))

totallines=1250
totalsets=8
workers=8



method = 'oscillatorordered' #'energyordered' or 'oscillatorordered'
theta = 1
Individuallynormalised = False
print('g = ' + str(gprefactor))
print('method = ' + method)

def mode_eigs_wishart(Dg, De, normalised, beta=2, c=1.0):
    M = np.min([Dg,De])
    N = np.max([Dg,De])
    
    if N==M:
        z=np.array([0.0])
        z_lag, _ = sp.special.roots_genlaguerre(M-1, 1.0)
        z = np.concatenate(([0.0], z_lag))
    else:
        alpha = N - M - 1
        z, _ = sp.special.roots_genlaguerre(M,alpha)
    
    roots_sorted = np.flip(np.sort(z))
    if normalised ==True:
        lambdas_mode = roots_sorted / roots_sorted[0]
        return lambdas_mode
    elif normalised==False:
        lambdas_mode = roots_sorted
        return lambdas_mode

def AA_energies_uptodark(wc,wa,Xq,O,Dg,De,Dmin,Dplu,Dk,geff=1, ordered = False):
    M = np.min([Dg,De])
    N = np.max([Dg,De])
    X = Xq
    laglist = 0#[sp.special.laguerre(t,False) for t in range(math.ceil(O+X[0]*geff**2))]
   
    evalsD = []
    for n in range(O):
        for k in range(N-M):
            if Dg>De:
                pm = -1
            elif Dg<De:
                pm = 1
            else:
                pass
            E = n*wc + pm*wa/2 + Dk[k]
            evalsD.append(E)

    evalsB = []
    for q in range(len(X)):
        indicator = math.floor(O + wa/2 + geff**2 *X[q]) 
        for t in range(indicator):
            Em = wc*(t-geff**2*X[q]/wc**2) + 0.5*(Dplu[q])\
                            -0.5*(wa+Dmin[q])*0#np.exp(-2*X[q]*geff**2/wc**2)\
                                            #*np.real(laglist[t](4*X[q]*geff**2/wc**2))
            Ep = wc*(t-geff**2*X[q]/wc**2) + 0.5*(Dplu[q])\
                            +0.5*(wa+Dmin[q])*0#np.exp(-2*X[q]*geff**2/wc**2)\
                                            #*np.real(laglist[t](4*X[q]*geff**2/wc**2))
            evalsB.append(Em)
            evalsB.append(Ep)
    
    if ordered==True:
        ls = np.array([x for x in np.sort(evalsB+evalsD)])
        return ls[~np.isnan(ls)]
    else:
        evals = evalsB+evalsD
        return evals
        
def generate_qfi_list_theor2(wc, wa, Xq, Tlist, Dg, De, Dmin=0, Dplu=0, Dk=0, gprefactor=1, Ocutoff = 0):
    laguerrelist = [sp.special.laguerre(n,False) for n in range(math.floor(Ocutoff+gprefactor**2+2))]
    M = int(np.min([Dg,De]))
    N = int(np.max([Dg,De]))
    if Dg>De:
        p=mpf(-1)
    elif De>Dg:
        p=mpf(1)
    else:
        p=mpf(0)
        
    if Dmin==0:
        Dmin = [mpf(0) for k in range(M)]
    else:
        Dmin = [mpf(k) for k in Dmin]
        
    if Dplu==0:
        Dplu = [mpf(0) for k in range(M)]
    else:
        Dplu = [mpf(k) for k in Dplu]
        
    if Dk==0:
        Dk = [mpf(0) for k in range(N-M)]
    else:
        Dk = [mpf(k) for k in Dk]
    
    X = [mpf(k) for k in Xq]
    g = mpf(gprefactor) #mpf(Xq[0])
    wf = mpf(wc)
    wa = mpf(wa)
    T = [mpf(k) for k in Tlist]
    num=len(Tlist)
    QFI = np.empty([num],dtype=object)
    
    for t in range(len(T)):
        Oindicatorbrightlist = [math.floor(Ocutoff + wa/2 + gprefactor**2 *X[q]+1) for q in range(M)]
        indicatordark =  Ocutoff
        beta = 1/T[t]
        
        gamD = [] #indep of q
        exgamD = [] #indep of q
        
        gamB = np.empty([M,max(Oindicatorbrightlist)+1],dtype=object)  #q,n both hmm
        GamB = np.empty([M,max(Oindicatorbrightlist)+1],dtype=object)
        exgamB = np.empty([M,max(Oindicatorbrightlist)+1],dtype=object)
        
        CoshGamB = np.empty([M,max(Oindicatorbrightlist)+1],dtype=object)
        TanhGamB = np.empty([M,max(Oindicatorbrightlist)+1],dtype=object)
        #darkstates
        for h in range(indicatordark):
            gamD_h = mpf(-1)*(mpf(h)*wf + wa*p*mpf(0.5))
            exgamD_h = mpmath.exp(beta*gamD_h)
            gamD.append(gamD_h); exgamD.append(exgamD_h)
        #brightbits
        for q in range(M):
            for h in range(Oindicatorbrightlist[q]+1):
                gamB[q,h] = g*g*X[q]/wf - Dplu[q]*mpf(0.5) - wf*mpf(h)
                GamB[q,h] = mpf(0.5)*(wa+Dmin[q])*mpmath.exp(mpf(-2)*g*g*X[q]/(wf*wf))*mpf(laguerrelist[h](float(4*g*g*X[q]/(wf*wf))))
                exgamB[q,h] = mpmath.exp(beta*gamB[q,h])
                CoshGamB[q,h] = mpmath.cosh(beta*GamB[q,h])
                TanhGamB[q,h] = mpmath.tanh(beta*GamB[q,h])
        Z = mpf(N-M)*sum(exgamD[h] for h in range(Ocutoff)) + mpf(2)*sum(sum(exgamB[k,h]*CoshGamB[k,h] for h in range(Oindicatorbrightlist[k])) for k in range(M))
        S1 = mpf(2)*sum(sum(exgamB[k,h] * GamB[k,h] * GamB[k,h] / CoshGamB[k,h] for h in range(Oindicatorbrightlist[k])) for k in range(M))
        S2 = mpf(N-M)*sum(gamD[h]*gamD[h]*exgamD[h] for h in range(Ocutoff)) + mpf(2)*sum(sum(exgamB[k,h]*CoshGamB[k,h]*mpmath.power(gamB[k,h]+GamB[k,h]*TanhGamB[k,h],2) for h in range(Oindicatorbrightlist[k])) for k in range(M))
        S3 = mpf(N-M)*sum(gamD[h]*exgamD[h] for h in range(Ocutoff)) + mpf(2)*sum(sum(exgamB[k,h]*CoshGamB[k,h]*(gamB[k,h]+GamB[k,h]*TanhGamB[k,h]) for h in range(Oindicatorbrightlist[k])) for k in range(M))
        QFI1 = S1/Z
        QFI2 = S2/Z
        QFI3 = -(S3*S3)/(Z*Z)
        QFI[t] = float( (QFI1 + QFI2 + QFI3)/(mpmath.power(T[t],4)) )
    return QFI

def logsumexp_mp(xs):
    # xs: list of mp.mpf
    m = max(xs)
    return m + mp.log(mp.fsum(mp.e**(x - m) for x in xs))

def generate_qfi_list_fromE(Elist, Tlist):
    E=np.array(Elist)
    Tl = [mpf(k) for k in Tlist]
    qfi_list = []
    Z_list = []
    for T in Tl:
        logweights  = [-(e/T) for e in E] 
        logZ  = logsumexp_mp(logweights)
        p = [mp.e**(logw-logZ) for logw in logweights]
        
        mu1 = mp.fsum(pi * ei for pi, ei in zip(p, E))
        mu2 = mp.fsum(pi * (ei * ei) for pi, ei in zip(p, E))
        var = mu2 - mu1 * mu1

        Z_list.append(mp.e**logZ)
        qfi_list.append(var/(T**mpf(4)))
    return qfi_list

def seperation(geff_list,Xq,wc,wa):
    E1 = [-g**2*Xq[0]/wc**2 - 0.5*wa*np.exp(-2/wc**2 * g**2 * Xq[0]) for g in geff_list]
    E2 = [-g**2*Xq[1]/wc**2 - 0.5*wa*np.exp(-2/wc**2 * g**2 * Xq[1]) for g in geff_list]
    E3 = [-g**2*Xq[0]/wc**2 + 0.5*wa*np.exp(-2/wc**2 * g**2 * Xq[0]) for g in geff_list]
    E21= [E2[i]-E1[i] for i in range(len(geff_list))]
    E31= [E3[i]-E1[i] for i in range(len(geff_list))]
    seperationlist = [np.min([E21[i],E31[i]]) for i in range(len(geff_list))]
    return seperationlist[0]

class CmatRandomAF:
    def __init__(self, DG, DE, normalised, sd_independent=1/np.sqrt(2)):
        self.data = np.random.normal(0,1*sd_independent,size=(DG,DE)) + 1j * np.random.normal(0,1*sd_independent,size=(DG,DE))
        self.U, self.svdvals, self.Vt = sp.linalg.svd(self.data)
        if normalised == True:
            self.data = 1/self.svdvals[0]*self.data
            self.svdvals = 1/self.svdvals[0] * self.svdvals
        else:
            pass

def generate_detunings(ep1,ep2,wa,Dg,De,Cmat):
    M = int(np.min([Dg,De]))
    N = int(np.max([Dg,De]))
    delta_g = np.random.normal(0, ep1*wa/2, size=(1, Dg))[0]
    delta_e = np.random.normal(0, ep2*wa/2, size=(1, De))[0]
    Delta_g = [sum([delta_g[k]*np.abs(Cmat.U[k,j])**2 for k in range(Dg)]) for j in range(M)]
    Delta_e = [sum([delta_e[k]*np.abs(Cmat.Vt[k,j])**2 for k in range(De)]) for j in range(M)]
    Dmin = [Delta_e[i]-Delta_g[i] for i in range(M)]
    Dplu = [Delta_e[i]+Delta_g[i] for i in range(M)]
    if Dg>De or De>Dg:
        Dk = [0 for i in range(M,N)] #disregarding Dk since it is not correct to have it, look at paper...
    else:
        Dk = 0
    return Dmin, Dplu, Dk

def generate_subdataframe(totallines):
    df_parts = []
    elist_parts = []
    for i in range(totallines):
        print('Progress: '+str(i/totallines))
        Cmat=CmatRandomAF(Dg,De,Individuallynormalised)
        Xq = [svd**2 for svd in Cmat.svdvals]
        #sep = seperation([gprefactor],Xq,wc,wa)
        #sep_list = [sep for k in range(numT)]
        #Xq_list = [Xq for k in range(numT)]
        if M==1:
            Xqratio = [np.nan for k in range(numT)]
        elif M>1:
            Xqratio = [Xq[1]/Xq[0] for k in range(numT)]
        Dmin, Dplu, Dk = generate_detunings(ep1, ep2, wa, Dg, De, Cmat)
        if method == 'oscillatorordered':
            energylist = AA_energies_uptodark(wc, wa, Xq, 1, Dg, De, Dmin, Dplu, Dk, geff=gprefactor, ordered = False)
            qfi_values = generate_qfi_list_theor2(wc, wa, Xq, Tlist, Dmin, Dplu, Dk, gprefactor)
        elif method == 'energyordered':
            energylist = AA_energies_uptodark(wc, wa, Xq, theta, Dg, De, Dmin, Dplu, Dk, geff=gprefactor, ordered = False)
            qfi_values = generate_qfi_list_fromE(energylist, Tlist)
        else:
            raise ValueError('what? please set a valid method')
        line_df = pd.DataFrame({"Temp": Tlist, "QFI": qfi_values, "Xqratio": Xqratio})#, "Xq":Xq_list, "Seperation":sep_list})
        df_parts.append(line_df)
        df_parts.append(pd.DataFrame({"Temp": [np.nan], "QFI": [np.nan], "Xqratio": [np.nan]}))  # separator row
        elist_parts = elist_parts+energylist
    return pd.concat(df_parts, ignore_index=True), elist_parts

def populate_dataframes_parallel_cpu(totallines, totalsets):
    bigset = []
    energies = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(generate_subdataframe, totallines) for _ in range(totalsets)]
        for fut in as_completed(futures):
            dfs, es = fut.result()
            bigset.append(dfs)
            energies = energies + es
    return bigset, energies

def averageqfi():
    Xq = mode_eigs_wishart(Dg, De, Individuallynormalised)
    svddiagonals = [x**0.5 for x in Xq]
    avgqfi = generate_qfi_list_theor2(wc, wa, Xq, Tlist, Dmin=0, Dplu=0, Dk=0, gprefactor=gprefactor)
    return avgqfi

def main():
    print(mp)
    print("Creating dataframe...")
    bigset, energies = populate_dataframes_parallel_cpu(totallines, totalsets)
    qfidf = pd.concat(bigset, ignore_index=True)
    qfidf.to_csv('qfidataframe.csv', index=False)
    print('done biggy one :)')
    print('Starting average one...')
    avgqfi = averageqfi()
    with Path("energies.pkl").open("wb") as f:
        pickle.dump(energies, f, protocol=pickle.HIGHEST_PROTOCOL)
    with Path("avgqfi.pkl").open("wb") as f:
        pickle.dump(avgqfi, f, protocol=pickle.HIGHEST_PROTOCOL)
    with Path("tlist.pkl").open("wb") as f:
        pickle.dump(Tlist, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Finito!')
if __name__ == '__main__':
    main()
