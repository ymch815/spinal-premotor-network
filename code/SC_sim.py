#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simulation of multiple trials
"""

import numpy as np
from funcs import neural_model
from funcs import joint_model
from funcs import metrics

from scipy.fft import rfft, rfftfreq

import os, time, sys

import pandas as pd

def init_pops(facilitation=None):

    
    EF = neural_model.WC_Neuron(Ja = 140.0, Jdp = 110.0, s0=0.011,   
                   taus = 3.9615, taua = 166.80, taudp = 1.8765, taur = 250.2, 
                   x0 = 0.9223, u0 = 0.1, uFlag = 1,UU = 0.03)
    IF = neural_model.WC_Neuron(Ja = 180.0, Jdp = 0.0, s0=0.012,
                   taus = 5.4210, taua = 62.55, taudp = 1000.0, taur = 225.0, 
                   x0 = 0.4796, u0 = 0.4, uFlag = 0)
    EX = neural_model.WC_Neuron(Ja = 140.0, Jdp = 110.0, s0=0.01,
                   taus = 3.9615, taua = 166.80, taudp = 1.8765, taur = 250.2, 
                   x0 = 0.9223, u0 = 0.1, uFlag = 1,UU = 0.03)
    IX = neural_model.WC_Neuron(Ja = 180.0, Jdp = 0.0, s0=0.01,
                   taus = 5.4210, taua = 62.55, taudp = 1000.0, taur = 225.0, 
                   x0 = 0.4796, u0 = 0.4, uFlag = 0)
    MF = neural_model.WC_Neuron(Ja = 140.0, Jdp = 0.0,
                   taus = 5.4210, taua = 500.4, taudp = 1000.0, taur = 225.0, 
                   x0 = 0.4796, u0 = 0.4, uFlag = 0)
    MX = neural_model.WC_Neuron(Ja = 140.0, Jdp = 0.0, 
                   taus = 5.4210, taua = 500.4, taudp = 1000.0, taur = 225.0, 
                   x0 = 0.4796, u0 = 0.4, uFlag = 0)
    
    if facilitation is not None:
        EF = neural_model.WC_Neuron(Ja = 140.0, Jdp = 110.0, s0=0.011,   
                   taus = 3.9615, taua = 166.80, taudp = 1.8765, taur = 250.2, 
                   x0 = 0.9223, u0 = 0.1, uFlag = 0,UU = 0.03)
        EX = neural_model.WC_Neuron(Ja = 140.0, Jdp = 110.0, s0=0.011,   
                   taus = 3.9615, taua = 166.80, taudp = 1.8765, taur = 250.2, 
                   x0 = 0.9223, u0 = 0.1, uFlag = 0,UU = 0.03)
        
    return [EF, IF, EX, IX, MF, MX]

def run_SC_sim(var='V1', modulation = 'ablation', var_min = 0.0, var_max = 1.0, noise = 0, Nrep = 1, Nsample = 20,
    outputvar = 'freq',facilitation=None,fft=None,connname=None,connval=None):
    
    '''
    initialize random states and generate random numbers based on number of parameters, Nrep, Nsample and noise mode
    noise = 0: no noise
    noise = 1: random numbers for every repetition and every variable
    noise = 2: random numbers for every repetition, but not every variable
    '''

    if noise == 0:
        random_scales = np.ones((7,Nrep,Nsample))
    elif noise == 1:
        prng = np.random.RandomState(1234)
        random_scales = 1.0 + prng.uniform(-1.0,1.0,size=(7,Nrep,Nsample))*0.10
    elif noise == 2:
        prng = np.random.RandomState(2345)
        random_scales_ = 1.0 + prng.uniform(-1.0,1.0,size=(7,Nrep))*0.10
        random_scales = np.repeat(random_scales_,Nsample,axis=1).reshape((7,Nrep,Nsample))

    V1F = 0.4545 # 
    V1X = 0.8181 # 
    V2bF = 0.5455 # 
    V2bX = 0.1819 # 
    
    pararan = np.linspace(var_min, var_max, Nsample)

    outputvarlist = outputvar.split('_')
    
    outputallF = []
    outputallX = []
    if fft is not None:
        fftresultall = []
    
    ListWCNeuron = init_pops()
 
    if facilitation is not None:
        ListWCNeuron = init_pops(facilitation=facilitation)
    
    V2b_inp_scale = 0
    V1_inp_scale = 0

    V3prop = 0.2

    varlist = var.split('_')

    paravalues = pd.DataFrame()
    
    for kk in range(Nrep):
        V1scale = 1
        V2bscale = 1
        Escale = 1
        I0E = 0.6
        I0I = 0.3
        freqarrF = []
        freqarrX = []

        ccratearr = []

        irflagarrF = []
        irflagarrX = []

        jointfreqarr = []

        phasediffarr = []

        for k in range(Nsample):
            # print information in each trial
            print('Repetition: %i, sampling: %i'%(kk,k))
            cdict = {}
            cdict['Repetition'] = kk
            cdict['Variable'] = pararan[k]
            cdict['Jee2'] = 0.0
            cdict['Jie2']= 52.0 * random_scales[0,kk,k]
            cdict['Jei2'] = 66.0 * random_scales[1,kk,k]
            # cdict['Jei2'] = 60.0
            cdict['Jii2'] = 0.0
            cdict['Jee'] = 6.5 * random_scales[2,kk,k]
            cdict['Jii'] = 55.0 * random_scales[3,kk,k]
            cdict['Jei'] = 77.0 * random_scales[4,kk,k]

            cdict['Jme']=195.0 * random_scales[5,kk,k]
            cdict['Jmi']=330.0 * random_scales[6,kk,k]

            if modulation == 'ablation':
                if 'V1' in varlist:
                    V1scale = pararan[k]
                if 'V2b' in varlist:
                    V2bscale = pararan[k]
                if 'V2a' in varlist:
                    Escale = V3prop + pararan[k]*(1.0-V3prop)
                print('V1scale = %.2f, V2bscale = %.2f, Escale = %.2f'%(V1scale,V2bscale,Escale))

            if modulation == 'activation':
                if 'V1' in varlist:
                    V1_inp_scale = pararan[k]
                if 'V2b' in varlist:
                    V2b_inp_scale = pararan[k]
                if 'V2a' in varlist:
                    I0E = pararan[k] + 0.6

            if modulation == 'modulation':
                for conns in varlist:
#                     if noise != 0:
#                         cdict[conns] = pararan[k]*(1.0 + np.random.uniform(-1.0,1.0)*0.10)
#                     elif noise == 0:
                    cdict[conns] = pararan[k]
                    print(conns,cdict[conns])
            if connname:
                cdict[connname] = float(connval)
                print(connname,cdict[connname])
            paravalues = pd.concat([paravalues,pd.DataFrame(cdict,index=[0])],ignore_index=True)

            IscaleX = (V1scale* V1X+ V2bscale*V2bX)
            IscaleF = (V1scale* V1F+ V2bscale*V2bF)

            I0I_F = 0.3+V1_inp_scale*V1F+V2b_inp_scale*V2bF
            I0I_X = 0.3+V1_inp_scale*V1X+V2b_inp_scale*V2bX
            print('I0E = %.1f, I0IF = %.1f, I0IX = %.1f,'%(I0E,I0I_F,I0I_X))

            ConnMat = np.array([[Escale*cdict['Jee2'], IscaleF*-cdict['Jei2'],  \
                                Escale*cdict['Jee'], IscaleX*-cdict['Jei'], 0.0, 0.0],
                               [Escale*cdict['Jie2'], IscaleF*-cdict['Jii2'], \
                               0.0, IscaleX*-cdict['Jii'], 0.0, 0.0],
                               [Escale*cdict['Jee'], IscaleF*-cdict['Jei'], Escale*cdict['Jee2'], \
                               IscaleX*-cdict['Jei2'], 0.0, 0.0],
                               [0.0, IscaleF*-cdict['Jii'], Escale*cdict['Jie2'], \
                               IscaleX*-cdict['Jii2'], 0.0, 0.0],
                               [Escale * cdict['Jme'], 0.0, 0.0, -IscaleX*cdict['Jmi'], 0.0, 0.0],
                               [0.0, -IscaleF*cdict['Jmi'],Escale * cdict['Jme'], 0.0, 0.0, 0.0]])

            dt = 0.1

            FinalNN = neural_model.WC_NN(ListWCNeuron,ConnMat, beta = 0.011, dt = 0.1)
            T = 30000

            if k == 0:
                FinalNN.init_dynamics(T, uselast=False)
            else:
                FinalNN.init_dynamics(T, uselast=False)
            inp_arr = [I0E, I0I_F, I0E, I0I_X, 0.0, 0.0]
            for i in range(FinalNN.Npts):
                FinalNN.update(inp_arr)
                FinalNN.record(i)

            T_plot = 15000
            fac = 1000/18/3.0

            time_sim, x_mat = joint_model.Jointmodel(np.tanh(FinalNN.marr[4][-int(T_plot/dt):]*fac), np.tanh(FinalNN.marr[5][-int(T_plot/dt):]*fac),dt=0.1)
            jointfreq, jointcc = metrics.peak_cal_joint(np.rad2deg(x_mat[:,4]),np.rad2deg(x_mat[:,4]),len(x_mat[:,4]),[75-5,75+5],0.1,
                            FinalNN.marr[4][-int(T_plot/dt):]*fac, FinalNN.marr[5][-int(T_plot/dt):]*fac,0.1)


            if fft is not None:
                # fft
                N = len(x_mat[:,4])
                # normalize the trace
                yf = rfft(x_mat[:,4]/np.max(x_mat[:,4]))
                xf = rfftfreq(N, 1/(1000/dt))

                fftresultall.append(np.abs(yf)[0:200])


            freqF,ccrateF,irflagF  = metrics.peak_cal_contract(FinalNN.marr[4],FinalNN.marr[4],FinalNN.marr[5],0.2,
                                      int(FinalNN.Npts/2),scale= 0.1,dt=0.1)
            freqX,ccrateX,irflagX  = metrics.peak_cal_contract(FinalNN.marr[5],FinalNN.marr[5],FinalNN.marr[4],0.2,
                                      int(FinalNN.Npts/2),scale= 0.1,dt=0.1)

            phasediff =  metrics.phase_diff(int(FinalNN.Npts/2),FinalNN.marr[4], FinalNN.marr[5],scale=0.1)

        
            print(jointfreq,jointcc,freqF,freqX,ccrateF,ccrateX,irflagF,irflagX)
            
            freqarrF.append(freqF)
            freqarrX.append(freqX)

            irflagarrF.append(irflagF)
            irflagarrX.append(irflagX)
            
            ccratearr.append(jointcc)

            jointfreqarr.append(jointfreq)

            phasediffarr.append(phasediff)

        for outputvar_ in outputvarlist:
            if outputvar_ == 'freq':
                outputallF.append([0 if x is None else x for x in freqarrF])
                outputallX.append([0 if x is None else x for x in freqarrX])
            if outputvar_ == 'ccrate':
                ccratearr = [1.0 if x is None else 1.0-x for x in ccratearr]
                outputallF.append(ccratearr)
                outputallX.append(ccratearr)
            if outputvar_ == 'irflag':
                outputallF.append(irflagarrF)
                outputallX.append(irflagarrX)
            if outputvar_=='jointfreq':
                outputallF.append([0 if x is None else x for x in jointfreqarr])
                outputallX.append([0 if x is None else x for x in jointfreqarr])
            if outputvar_=='phasediff':
                outputallF.append([0 if x is None else x for x in phasediffarr])
                outputallX.append([0 if x is None else x for x in phasediffarr])


            
    # save files
    fdToSave = 'test/var_%s_mode_%s/' %(var,modulation)
    if facilitation is not None:
        fdToSave = 'test/no_fac/var_%s_mode_%s/'%(var,modulation)
    if not os.path.exists(fdToSave):
            os.makedirs(fdToSave)
    outputallF = np.reshape(outputallF,(len(outputvarlist),Nrep,Nsample),order = 'F')
    outputallX = np.reshape(outputallX,(len(outputvarlist),Nrep,Nsample),order = 'F')

    for ind_, outputvar_ in enumerate(outputvarlist):
        filename = 'var_%s_mode_%s_%.1f_%.1f_noise_%i_%i_%i_%s'%(var,modulation,
                            var_min,var_max,noise,Nrep,Nsample,outputvar_)
        if connname is not None:
            filename = 'var_%s_mode_%s_%.1f_%.1f_noise_%i_%i_%i_%s_%i_%s'%(var,
                            modulation,var_min,var_max,noise,Nrep,Nsample,
                connname,connval,outputvar_)
        if outputvar_ in ['ccrate','jointfreq','phasediff']:
            np.savetxt(fdToSave+filename+'.txt',outputallF[ind_])
        else:
            np.savetxt(fdToSave+filename+'_F.txt',outputallF[ind_])
            np.savetxt(fdToSave+filename+'_X.txt',outputallX[ind_])
    filename = 'var_%s_mode_%s_%.1f_%.1f_noise_%i_%i_%i'%(var,modulation,var_min,
                                                    var_max,noise,Nrep,Nsample)
    if connname is not None:
        filename = 'var_%s_mode_%s_%.1f_%.1f_noise_%i_%i_%i_%s_%i'%(var,
        modulation,var_min,var_max,noise,Nrep,Nsample,
            connname,connval)
    paravalues.to_csv(fdToSave+filename+'_parameters.csv')
    if fft is not None:
        np.savetxt(fdToSave+filename+'_fftresult.txt',np.array(fftresultall))
        np.savetxt(fdToSave+filename+'_fftfreq.txt',np.array(xf[0:200]))

def get_argument(index, arg_type=str,default=None):
    if len(sys.argv) > index and sys.argv[index].lower() != "none":
        return arg_type(sys.argv[index])
    else:
        return default
    
if __name__ == "__main__":
    t = time.time() # Count the time of running

    # The 1st argument is the python filename
    print("The name of the program:", sys.argv[0]) 
    print("---------------------------------------------\n") 

    if len(sys.argv) < (1+8):
        print('  [Error] No input argument provided. Program ended!\n')
        exit()

    # Set Parameters
    var = str(sys.argv[1]) # str input, for multiple modulation, use '_' to seperate
    modulation = str(sys.argv[2]) # str input, ablation, activation or modulation (single modulation of specific connections)
    var_min = float(sys.argv[3]) # float input
    var_max = float(sys.argv[4]) # float input
    noise = int(sys.argv[5]) # int input, noise mode
    Nrep = int(sys.argv[6]) # int input
    Nsample = int(sys.argv[7]) # int input
    outputvar = str(sys.argv[8]) # str input, for multiple outputs, use '_' to seperate
    
    # optional arguments
    fft = get_argument(9,int)
    facilitation = get_argument(10,int)
    connname = get_argument(11,str)
    connval = get_argument(12,float)
    
    
    
    paraList = 'Variable_%s_mode_%s_%.1f_%.1f_noise_%i_rep_%i_sample_%i_output_%s_fac_%s_fft_%s_conn_%s_%s' % \
    (var, modulation, var_min, var_max, noise, Nrep, Nsample,outputvar,
     facilitation,fft,connname,connval)
    print(paraList+'  To Run!')
    
    run_SC_sim(var=var, modulation = modulation, var_min = var_min, var_max = var_max, \
    noise = noise, Nrep = Nrep, Nsample = Nsample, outputvar=outputvar,facilitation=facilitation,
    fft=fft,connname=connname,connval=connval)

    elapsed = time.time() - t
    print('\nRun time: %f\n' % elapsed)
