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

mp.dps = 50

Dg=100
De=100

wc = 1
wa = 0.5

ep1=0
ep2=0.05

minT=1e-10
maxT=1e3
numT=500

totallines=100
totalsets=5

normalised = False

def generate_qfi_list_theor2(wc, wa, Xq, Tlist, Dmin=0, Dplu=0, Dk=0):
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
    g = mpf(1) #mpf(Xq[0])
    wf = mpf(wc)
    wa = mpf(wa)
    T = [mpf(k) for k in Tlist]
    num=len(Tlist)
    QFIlist1 = np.empty([num],dtype=object)
    QFIlist2 = np.empty([num],dtype=object)
    QFIlist3 = np.empty([num],dtype=object)
    QFI = np.empty([num],dtype=object)
    for t in range(len(T)):
        # per-q lists in mpf
        e1 = []
        e2 = []
        e3 = []
        ch = []
        sc = []
        th = []
        arg = []
        beta = 1/T[t]
        nmexp = []
        temp = []
        thing=[]
        for k in range(N-M):
            thing_k = (p*wa/mpf(2)+Dk[k])
            nmexp_k = mpmath.exp(-beta*thing_k)
            thing.append(thing_k); nmexp.append(nmexp_k)
        for q in range(M):
            e1_q = mpmath.exp((g*g)*X[q] * beta / wf)              # exp1
            e2_q = mpmath.exp(-1*mpf(2)*(g*g)*X[q] / (wf*wf))             # exp2
            e3_q = mpmath.exp(-beta/mpf(2) * Dplu[q])
            arg_q  = ((wa +Dmin[q])* beta / mpf(2)) * e2_q
            ch_q = mpmath.cosh(arg_q)
            th_q = mpmath.tanh(arg_q)
            e1.append(e1_q); e2.append(e2_q); ch.append(ch_q); th.append(th_q); sc.append(1/ch_q); e3.append(e3_q)
        Z = sum(nmexp[k] for k in range(N-M)) + mpf(2)*sum(e1[q]*e3[q]*ch[q] for q in range(M))
        S1 = mpf(1/2) * ( sum(((wa+Dmin[q])**mpf(2)) * e1[q] * (e2[q]**mpf(2)) * e3[q] * sc[q] for q in range(M)) ) 
        S2 =   sum(thing[k]*thing[k] * nmexp[k] for k in range(N-M))  + mpf(2) * sum( e1[q]*e3[q]*ch[q] * mpmath.power((g*g)*X[q]/wf - mpf(0.5)*Dplu[q] + mpf(0.5)*(wa+Dmin[q])*e2[q]*th[q], 2)  for q in range(M) )
        S3 =  -mpf(1) * sum(thing[k] * nmexp[k] for k in range(N-M))  + mpf(2)*sum( e1[q]*e3[q]*ch[q] * ( (g*g)*X[q]/wf - mpf(0.5)*Dplu[q] + mpf(0.5)*(wa+Dmin[q])*e2[q]*th[q] )  for q in range(M) ) 
        QFIlist1[t] = S1/Z
        QFIlist2[t] = S2/Z
        QFIlist3[t] = -(S3*S3)/(Z*Z)
        QFI[t] = float( (QFIlist1[t] + QFIlist2[t] + QFIlist3[t])/(mpmath.power(T[t],4)) )
    return QFI

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

def generate_subdataframe(totallines):
    Tlist = np.geomspace(minT, maxT, numT)
    df_parts = []
    for i in range(totallines):
        print('Progress: '+str(i/totallines))
        Cmat=CmatRandomAF(Dg,De,normalised)
        Xq = [svd**2 for svd in Cmat.svdvals]
        #sep = seperation([Xq[0]],Xq,wc,wa)
        #sep_list = [sep for k in range(numT)]
        #Xq_list = [Xq for k in range(numT)]
        
        qfi_values = generate_qfi_list_theor2(wc, wa, Xq, Tlist)
        
        line_df = pd.DataFrame({"Temp": Tlist, "QFI": qfi_values})#, "Xq":Xq_list, "Seperation":sep_list})
        df_parts.append(line_df)
        df_parts.append(pd.DataFrame({"Temp": [np.nan], "QFI": [np.nan], "Seperation": [np.nan]}))  # separator row
    
    return pd.concat(df_parts, ignore_index=True)

def populate_dataframes_parallel_cpu(totallines, totalsets):
    bigset = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(generate_subdataframe, totallines) for _ in range(totalsets)]
        for fut in as_completed(futures):
            bigset.append(fut.result())
    return bigset

def populate_dataframes_parallel(totallines, totalsets):
    bigset = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit tasks
        futures = [executor.submit(generate_subdataframe, totallines) for _ in range(totalsets)]
        time.sleep(0)
        # Collect results
        for future in concurrent.futures.as_completed(futures):
            df = future.result()
            bigset.append(df)
    return bigset

def main():
    print(mp)
    print("Creating dataframe...")
    bigset = populate_dataframes_parallel_cpu(totallines, totalsets)
    qfidf = pd.concat(bigset, ignore_index=True)  
    qfidf.to_csv('qfidataframe.csv', index=False)
    print('done :)')

if __name__ == '__main__':
    main()
