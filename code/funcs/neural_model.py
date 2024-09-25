#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
functions for neural model
"""

import numpy as np
from scipy import signal
from scipy.fft import rfft, rfftfreq

# wilson-cowan model with intrinsic properties
class WC_Neuron:
    def __init__(self, s0 = 0.01, a0 = 0.5, dp0 = 0.0, Ja = 0.0, Jdp = 0.0, 
               taus = 7.5, taua = 1000.0, taudp = 1000.0, taur = 225.0, 
               x0 = 0.4, u0 = 0.1, uFlag = 1, UU = 0.2):
        
        self.initial_set = {'s0':s0, 'a0':a0, 'dp0':dp0, 'x0':x0, 'u0':u0}
        
        self.s = s0 # synaptic gating
        self.a = a0 # adaptation current
        self.dp = dp0 # depolarization current
        
        self.Ja = Ja # strength of adaptation current
        self.Jdp = Jdp # strength of depolarization current
        
        self.taus = taus 
        self.taua = taua
        self.taudp = taudp
        self.taur = taur # time scale of synaptic facilitation 
        
        self.x = x0 # synaptic depression 
        self.u = u0 # synaptic facilitation 
        
        self.UU = UU 
        
        self.uFlag = uFlag # have facilitation or not
        
    def set_init(self): # set back to innitial values
        self.s = self.initial_set['s0'] # synaptic gating
        self.a = self.initial_set['a0'] # adaptation current
        self.dp = self.initial_set['dp0'] # depolarization current
        self.x = self.initial_set['x0'] # synaptic depression 
        self.u = self.initial_set['u0'] # synaptic facilitation 
        
        return self
        
    def fr(self, beta, sum_currents, I0): # firing rate
        return beta * self.lininter(sum_currents + I0 + self.dp - self.a)
    
    def lininter(self,x):
        return np.heaviside(x,0)*x
        
    def update(self, svarList, ConnList, beta, dt, I0): # update all variables at one time step
        
        sum_currents = 0
        for i, svars in enumerate(svarList):
            sum_currents += svars * ConnList[i]
        
        self.m = self.fr(beta, sum_currents, I0)
        
        dp_next = self.dp + dt * (-self.dp + self.Jdp * self.m)/self.taudp
        s_next = self.s + dt * (-self.s/self.taus + self.x*self.m*self.u)
        a_next = self.a + dt * (-self.a + self.Ja*self.m)/self.taua
        
        if self.uFlag == 1:  
            u_next = self.u + dt*((self.UU - self.u)/self.taur + self.UU*(1.0-self.u)*self.m)
            self.u = u_next

        self.s = s_next
        self.dp = dp_next
        self.a = a_next
        
        return self

# network of WCneurons
class WC_NN:
    def __init__(self, ListWCNeuron, ConnMat, beta = 0.011, dt = 0.2, 
                 toFeedback = None, fromFeedback = None, updatingrate = 0.0):
        self.ListWCNeuron = ListWCNeuron # list of WC_Neuron
        self.ConnMat = ConnMat # rows: to, cols: from
        self.beta = beta
        self.dt = dt
        self.Npops = len(ListWCNeuron)
        self.toFeedback = toFeedback
        if self.toFeedback is not None: 
            """
            only applies when you have 2 modules, specify the *index* of 2 population to compute difference in 
            toFeedback list
            specify populations receiving feedback in fromFeedback by *connectivity*
            """
            self.hist_diff = 0.0
            self.updatingrate = updatingrate
            self.fromFeedback = fromFeedback

        
    def update(self, I0): # update all populations in the network in one time step
        svarList = []
        for i,Neurons in enumerate(self.ListWCNeuron):
            svarList.append(self.ListWCNeuron[i].s)
        NewNeurons = []
        if self.toFeedback is not None:
            I0 = I0 + self.fbCurrent * self.fromFeedback
        
        for i, pops in enumerate(self.ListWCNeuron):
            NewNeurons.append(pops.update(svarList = svarList, 
                                          ConnList = self.ConnMat[i], 
                                          beta = self.beta, 
                                          dt = self.dt,
                                         I0 = I0[i]))
        self.ListWCNeuron = NewNeurons
        if self.toFeedback is not None:
            self.hist_diff = self.ListWCNeuron[self.toFeedback[0]].m - self.ListWCNeuron[self.toFeedback[1]].m
            self.fbCurrent += self.dt * self.updatingrate * self.hist_diff
    
    def init_dynamics(self,T,uselast=False): # initialize array to store variables
        # arrays to store dynamics
        self.Npts = int(T/self.dt)
        self.marr = np.zeros((self.Npops, self.Npts))
        self.sarr = np.zeros((self.Npops, self.Npts))
        self.aarr = np.zeros((self.Npops, self.Npts))
        self.uarr = np.zeros((self.Npops, self.Npts))
        self.dparr = np.zeros((self.Npops, self.Npts))
        if self.toFeedback is not None:
            self.fbCurrent_arr = np.zeros((1, self.Npts))
            self.fbCurrent = 0.0
        if uselast == False:
            initNeurons = []
            for i in range(len(self.ListWCNeuron)):
                initNeurons.append(self.ListWCNeuron[i].set_init())
            self.ListWCNeuron = initNeurons

    def record(self,curr): # record variable at one time point
        for kk, Neurons in enumerate(self.ListWCNeuron):
            self.marr[kk, curr] = self.ListWCNeuron[kk].m
            self.sarr[kk, curr] = self.ListWCNeuron[kk].s
            self.aarr[kk, curr] = self.ListWCNeuron[kk].a
            self.uarr[kk, curr] = self.ListWCNeuron[kk].u
            self.dparr[kk, curr] = self.ListWCNeuron[kk].dp
        if self.toFeedback is not None:
            self.fbCurrent_arr[0,curr] = self.fbCurrent
             
    def concat_NN(self, WC_NN2, InterMat_to, InterMat_from): # concatenate one WC_NN with another
        NewListWCNeuron = self.ListWCNeuron + WC_NN2.ListWCNeuron
        NewConnMat = np.zeros((len(NewListWCNeuron),len(NewListWCNeuron)))
        NewConnMat[:self.Npops,:self.Npops] = self.ConnMat
        NewConnMat[self.Npops:,self.Npops:] = WC_NN2.ConnMat
        NewConnMat[:self.Npops,self.Npops:] = InterMat_to
        NewConnMat[self.Npops:,:self.Npops] = InterMat_from
        return NewListWCNeuron,NewConnMat