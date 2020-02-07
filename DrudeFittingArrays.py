# -*- coding: utf-8 -*-
#from scipy.optimize import leastsq
from scipy.optimize import least_squares
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
# this is a template for a single fit of a complex fucntion
# data provided. data stored in a three column text file, namely frequency, real, imag parts of the response function
path = 'C:\\Users\\Youcheng\\Desktop\\IT9_2T_Thermal\\'
name = 'SubCorr_2T_cut'
append = '.txt' 
filename = path + name +append

data = pd.read_csv(filename,delimiter="\t",header=None,dtype=np.float64)
data_array = np.array(data.values)

#x is the frequency, ie first column 
x = data_array[:,0]
#the number of frequency points
NF = x.shape[0]

#the frequency to pass to functions
#x1 =  np.concatenate((x,x))

#beta0-5 represents A_SC (related with frequency times imaginary part), sigma_sc, and tau_sc,  sigma_n, tau_n respectively. tau_n is set to be 0
  
def mdl1_func(beta,x):
   mdl1 = (beta[2]*(1+beta[1]**2*x**2)*(1+abs(beta[4])**2*beta[3]**2+beta[3]**2*x**2)+beta[0]*(abs(beta[4])**4*beta[3]**4+(1+beta[3]**2.*x**2)**2)+abs(beta[4])**2*(2*beta[3]**2-2*beta[3]**4*x**2))/(1+beta[1]**2*x**2)/(abs(beta[4])**4*beta[3]**4+(1+beta[3]**2*x*x)**2+abs(beta[4])**2*(2*beta[3]**2-2*beta[3]**4*x**2))   
   return mdl1
         
def mdl2_func(beta,x):  
   mdl2 = (x*(beta[2]* beta[3]*(1+ beta[1]**2*x**2)*(1- abs(beta[4])**2* beta[3]**2+ beta[3]**2*x**2)+beta[0]*beta[1]*(abs(beta[4])**4*beta[3]**4 +(1+ beta[3]**2*x**2)**2+ abs(beta[4])**2*(2* beta[3]**2-2* beta[3]**4*x**2))))/(1+ beta[1]**2*x**2)/( beta[3]**4* beta[4]**4+(1+ beta[3]**2*x**2)**2 + abs(beta[4])**2*(2* beta[3]**2-2* beta[3]**4*x**2)) 
   return mdl2
         
#previous we had a problem... I think it is because I need to combine y1 and y2 into y before passing it to this funciton                  
def residual_two_functions(beta, x, y_input):
    y1 = y_input[:NF]
    nt = y_input.shape[0]
    n = y1.shape[0]
    mdl = np.zeros((nt,1),dtype=float)
    mdl= mdl.flatten()
    diff = copy.copy(mdl) +0.0
    for i in range(nt):
        if i<n:
            if y_input[i]>0:
                mdl[i] = (beta[2]*(1+beta[1]**2*x[i]**2)*(1+beta[4]**2*beta[3]**2+beta[3]**2*x[i]**2)+beta[0]*(beta[4]**4*beta[3]**4+(1+beta[3]**2.*x[i]**2)**2)+beta[4]**2*(2*beta[3]**2-2*beta[3]**4*x[i]**2))/(1+beta[1]**2*x[i]**2)/(beta[4]**4*beta[3]**4+(1+beta[3]**2*x[i]*x[i])**2+beta[4]**2*(2*beta[3]**2-2*beta[3]**4*x[i]**2)) 
                diff[i] = (mdl[i]*10000.0-y_input[i]*10000.0)
                #print(diff[i])
            else:
                diff[i] =0            
        else:
            mdl[i] = (x[i-n]*(beta[2]* beta[3]*(1+ beta[1]**2*x[i-n]**2)*(1- beta[4]**2* beta[3]**2+ beta[3]**2*x[i-n]**2)+beta[0]*beta[1]*(beta[4]**4*beta[3]**4 +(1+ beta[3]**2*x[i-n]**2)**2+ beta[4]**2*(2* beta[3]**2-2* beta[3]**4*x[i-n]**2))))/(1+ beta[1]**2*x[i-n]**2)/( beta[3]**4* beta[4]**4+(1+ beta[3]**2*x[i-n]**2)**2 + beta[4]**2*(2* beta[3]**2-2* beta[3]**4*x[i-n]**2)) 
            diff[i] = (mdl[i]*10000.0-y_input[i]*10000)
    return diff

#define the total number of temperature points
NT=100

#define the place where you start 1 means starting with the first point
start = 1

#define the number temperature points that you want to go backwards
N = 100

#define the starting temperature, ending temperature
sT = 0.385822
eT =3.99249

#initial value of the fit, path, etc. 
best = np.zeros((6,N),dtype=float)
best_init =np.array([0.014420542678880639, 	1.687498574842766e-09, 0.002040444628945457, 1.3333575762246272e-10, 10332.07821640472]) 
bestsave =copy.copy(best)

#here is a for loop where the fitting of the data is iterated. range N means 0-(N-1)
for ind in range(N): 
    y1 = data_array[:,2*start+2*ind-1]
    y2 = data_array[:,2*start+2*ind]
    # x_scale=[1E7,0.1,1E8, 0.01, 1E13]
    #print(residual_two_functions(best_init, x, y1, y2))
    yt=np.concatenate((y1,y2),axis=0)    
    res = least_squares(residual_two_functions, best_init, method = 'lm', ftol=2.23e-16, xtol =2.5e-16, gtol =2.5E-16, args=(x,yt))
    #res = least_squares(residual_two_functions, best_init, bounds=([0,0,0,0,0],[0.1,2.687498574842766e-09,0.1,2e-10,np.inf]), ftol=2.23e-16, xtol =2.5e-16, gtol =2.5E-16, args=(x,yt))
    #diff_step =[9E-6, 1E-7,1E-7,1E-6, 1E-6]
    #method = 'lm',
    best[1:,ind] = res.x
    #print(res.cost)
    #print(res.status)
    print(best[1:,ind])
    best[0,ind] = sT+(start-1)*(eT-sT)/(NT-1)+(ind)*(eT-sT)/(NT-1)
    #print(best[:,ind]) 
    best_init=best[1:,ind]
    bestsave[1:,ind]=copy.copy(best[1:,ind])
    bestsave[0,ind]=best[0,ind]
    #save fitted data to file
    save_x = np.linspace(5E7,8E9,num=1601)
    save_y1 = mdl1_func(best[1:,ind],save_x)
    save_y2 = mdl2_func(best[1:,ind],save_x)  
    data_fit =pd.DataFrame({'Freq_fit' : pd.Series(save_x), 'Imag' : pd.Series(save_y2),'Real' : pd.Series(save_y1)})                                              
    savepath =path +name + '_' +str(ind) + '_fit' +append                                            
    data_fit.to_csv(savepath,sep='\t',mode = 'w', index=False, columns=['Freq_fit', 'Real', 'Imag'])
    #here I want to save these figures
    fig1 = plt.figure(figsize=(6,5))
    ax1 = fig1.add_axes([0.2, 0.15, 0.73, 0.75])
    line1, =ax1.plot(x, y1, linewidth=3, color='k')
    ax1.plot(x, y2, linewidth=3, color='k',linestyle='dashed')
    ax1.plot(save_x, save_y1, linewidth=3, color='r')
    ax1.plot(save_x, save_y2, linewidth=3, color='r',linestyle='dashed')
    plt.xlabel(r'$\omega/2\pi$ (Hz)',fontsize=20)
    plt.ylabel(r'$G_{1,2}$ ($\Omega^{-1}/sq$)',fontsize=20)
    ax1.set_xlim([0,8E9])
    ax1.xaxis.set_tick_params(width=2)
    ax1.yaxis.set_tick_params(width=2)
    plt.tick_params(axis='both', labelsize=15)
    ax1.legend(handles=[line1],title=str(best[0,ind]) + 'K',loc='upper right',frameon =False)
    fig1.savefig(path + 'fit_1T_'+ str(best[0,ind]) + '.jpg', dpi=300, format = 'jpg', frameon =False, transparent=True, bboxinches='tight')     
    plt.clf()
   
#save best values to a file
par = pd.DataFrame(bestsave)
par=par.T
savepath =path +'fit_parameters' + '_1T' +append                                                
par.to_csv(savepath,sep='\t',mode = 'w', index=False,header=False)
   
#I can change N so that the plot presented here will be a specific temprerature associated with N                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
#plt.plot(x, y1, linewidth=3, color='k')
#plt.plot(x, y2, linewidth=3, color='k',linestyle='dashed')
#plt.plot(save_x, save_y1, linewidth=3, color='r')
#plt.plot(save_x, save_y2, linewidth=3, color='r',linestyle='dashed')
#plt.show()
T=sT+(start-1)*(eT-sT)/(NT-1)+(N-1)*(eT-sT)/(NT-1)
print(T)                                         
