#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
functions for computing metrics
"""

import numpy as np
from scipy import signal

# function for computing metrics from joint angle traces
def peak_cal_joint(dynamicsPeak,dynamicsAmp,L,minangles,dt, MFdynamics, MXdynamics,scale):
    # dt: ms

    # detect peaks for flexor side: minangles[0]
    peaks1=signal.find_peaks(dynamicsPeak[-L:],height=minangles[1],prominence = 1)
    # detect peaks for extensor side: minangles[1]
    peaks2=signal.find_peaks(-dynamicsPeak[-L:]+150,height=150-minangles[0],prominence = 1)
    
    NpeaksMF=len(signal.find_peaks(MFdynamics[-L:],prominence=scale*max(MFdynamics[-L:]))[0])
    NpeaksMX=len(signal.find_peaks(MXdynamics[-L:],prominence=scale*max(MXdynamics[-L:]))[0])

    freq, ccrate = None, None
    if (np.size(peaks1[0])>6) & (np.size(peaks2[0])>6):
        # merge peaks1 and peaks2
        locs = np.concatenate([peaks1[0],peaks2[0]])
        sides = np.concatenate([np.ones_like(peaks1[0]),np.zeros_like(peaks2[0])])
        sides_sort = sides[np.argsort(locs)]
#         print(locs, sides, sides_sort)

        Nbehavior = (np.sum(sides_sort[np.where(sides_sort==0)[0][:-1]+1]==1)+
                    np.sum(sides_sort[np.where(sides_sort==1)[0][:-1]+1]==0))/2.0
        freq = Nbehavior/(L*dt/1000.0)
        ccrate = Nbehavior/min(NpeaksMF-1,NpeaksMX-1)
#         print(NpeaksMF,NpeaksMX,Nbehavior)
        if abs(Nbehavior-min(NpeaksMF-1,NpeaksMX-1))<=1:
            ccrate = 1.0
 
    return freq, ccrate

# compute metrics from MF/MX dynamics in neural model
def peak_cal_contract(dynamicsPeak1,dynamicsAmp1,dynamicsPeak2,thre,L,scale,dt):
    # dt: ms
    peaks1=signal.find_peaks(dynamicsPeak1[-L:],prominence=scale*max(dynamicsPeak1[-L:]))
    peaks2=signal.find_peaks(dynamicsPeak2[-L:],prominence=scale*max(dynamicsPeak2[-L:]))
    freq, ccrate, irflag = None, None, 0
    
    # if peak height variance > 0.1 ave peak height: irflag = 1
    if (np.size(peaks1[0])>6):
        temp_ = peaks1[1]['prominences'][1:-1]

        if (np.max(temp_)-np.min(temp_))>np.mean(temp_)*0.05:
            irflag = 1
    
    # delete found peaks if contraction happens
    if (np.size(peaks1[0])>6) & (np.size(peaks2[0])>6):
        temp = dynamicsAmp1[-L+peaks1[0][-6]:-L+peaks1[0][-2]]
        peakwidth = np.size(temp[temp>0.1*max(temp)])/4.0

        thredist = peakwidth*thre
        mindists = []
        amprat = []
        for i,peakloc in enumerate(peaks1[0]):
            minloc = np.argmin(abs(peaks2[0]-peakloc))
            mindists.append(min(abs(peaks2[0]-peakloc)))
            amprat.append((peaks1[1]['prominences'][i]/peaks2[1]['prominences'][minloc]))
        mindists = np.array(mindists) 
        amprat = np.array(amprat)

        indsel = np.where((mindists>thredist)|(amprat>1.5))[0]
        ###
        peaks_remain = peaks1[0][indsel]
        if len(peaks_remain)>6:
            freq = 1.0/((peaks_remain[-2]-peaks_remain[0])/(len(peaks_remain)-2)*dt)*1000.0 # return freq in Hz
            ccrate = len(indsel)/len(peaks1[0])
    return freq, ccrate, irflag

# compute phase difference from MF/MX dynamics in neural model
def phase_diff(L,MFdynamics, MXdynamics,scale):

    peaks1=signal.find_peaks(MFdynamics[-L:],prominence=scale*max(MFdynamics[-L:]))
    peaks2=signal.find_peaks(MXdynamics[-L:],prominence=scale*max(MXdynamics[-L:]))
    phasediff = None

    # if peak height variance > 0.1 ave peak height: irflag = 1, return None
    if (np.size(peaks1[0])>6):
        temp_ = peaks1[1]['prominences'][1:-1]

        if (np.max(temp_)-np.min(temp_))>np.mean(temp_)*0.05:
            return phasediff
    if (np.size(peaks1[0])>6) & (np.size(peaks2[0])>6):
        # compute period for MF and MX
        pdMF = (peaks1[0][-2]-peaks1[0][-6])/4.0 # period MF
        pdMX = (peaks2[0][-2]-peaks2[0][-6])/4.0 # period MF
    
        # compute peak difference between MF and MX
        # find the last reliable MF
        init = peaks1[0][-2]
        # find the largest MX smaller than this
        last = peaks2[0][np.where(peaks2[0]<init)[0][-1]]
        phasediff = (init-last)/((pdMF+pdMX)/2.0)*2.0*np.pi

   
    return phasediff