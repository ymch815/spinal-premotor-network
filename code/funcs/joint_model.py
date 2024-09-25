#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
functions for biomechanical model
"""

import numpy as np


"""
"Nonlinear Muscle" model
cr. Akira Nagamori

https://github.com/anagamori/Fink2014-model/blob/main/biomechanical_model.ipynb

"""

def FL_function(L):
    beta = 1.55
    omega = 0.75
    rho = 2.12
    
    FL = np.exp(-np.power(abs((np.power(L,beta) - 1)/(omega)),rho));
    return FL

def FV_con_function(L,V):
    Vmax = -9.15*2;
    cv0 = -5.7;
    cv1 = 9.18;
    
    FVcon = (Vmax - V)/(Vmax + (cv0 + cv1*L)*V);
    return FVcon

def FV_ecc_function(L,V):
    av0 = -1.4;
    av1 = 0.0;
    av2 = 0.0;
    bv = 0.72;
    FVecc = (bv - (av0 + av1*L + av2*np.power(L,2))*V)/(bv+V);
    
    return FVecc

def Fpe_function(L,V):
    c1_pe1 = 23.0*6
    k1_pe1 = 0.046
    Lr1_pe1 = 1.17
    eta = 0.0001;
    
    Fpe1 = c1_pe1 * k1_pe1 * np.log(np.exp((L - Lr1_pe1)/(k1_pe1))+1) + eta*V;
    
    return Fpe1

def Af_function(U,L):
    a = (2.0-0.5)/(1.0-0.001);
    b = 0.5-(2.0-0.5)/(1.0-0.001)*.001;
    f_eff = a*U+b;

    a_f = 0.56
    n_f0 = 2.1
    n_f1 = 3.3
    n_f = n_f0 + n_f1*(1.0/L-1.0);
    Af = 1 - np.exp(-np.power(f_eff/(a_f*n_f),n_f));
    return Af

def muscle_length_flexor(a,b,theta):
    L = np.sqrt(np.power(a,2) - 2*a*b*np.cos(theta) + np.power(b,2))
    return L

def muscle_length_extensor(a,b,theta):
    if theta > 1.63:
        L = np.sqrt(np.power(a,2) + 2*a*b*np.cos(theta) + np.power(b,2))
    else:
        L = np.sqrt(np.power(a,2)-np.power(b,2)) + b*(1.63-theta)    
    return L

def torque_flexor(a,b,theta,Fm):
    T = Fm*a*b*np.sin(theta)/np.sqrt(np.power(a,2)-2*a*b*np.cos(theta)+np.power(b,2))
    return T

def torque_extensor(a,b,theta,Fm):
    if theta > 1.63:
        T = -Fm*a*b*np.sin(theta)/np.sqrt(np.power(a,2) + 2*a*b*np.cos(theta) + np.power(b,2))
    else:
        T = -Fm*b    
    return T  

def forward_dynamis(x,dx,u,I,B,a,b,Lce0_1,Lce0_2,F0_1,F0_2,Lmax_1,Lmax_2,Lt_1,Lt_2):
    # Minimum joint angle
    theta_min = 40*np.pi/180.0
    # Maximum joint angle
    theta_max = 110*np.pi/180.0
    
    # Muscle time constant (sec)
    tau_1 = 0.01;
    tau_2 = 0.01;
    
    L_1 = muscle_length_flexor(a,b,x[4])
    Lce_1 = (L_1 - Lt_1)/Lce0_1
    
    L_2 = muscle_length_extensor(a,b,x[4])
    Lce_2 = (L_2 - Lt_2)/Lce0_2
    
    FL_1 = FL_function(Lce_1)
    FL_2 = FL_function(Lce_1)
        
    Fpe_1 = Fpe_function(Lce_1/Lmax_1,0)
    Fpe_2 = Fpe_function(Lce_2/Lmax_2,0)

    Af_1 = Af_function(x[0],Lce_1)
    Af_2 = Af_function(x[2],Lce_2)
    
    Fm_1 = (FL_1*x[0]*Af_1+Fpe_1)*F0_1
    Fm_2 = (FL_2*x[2]*Af_2+Fpe_2)*F0_2
        
    T_stop_1 = 0.01*np.exp(-(x[4]-theta_min)/0.01)
    T_stop_2 = -0.01*np.exp((x[4]-theta_max)/0.01)
    
    dx[0] = x[1]
    dx[1] = -x[0]/(tau_1*tau_2) - x[1]*(tau_1+tau_2)/(tau_1*tau_2) + u[0]/(tau_1*tau_2)
    dx[2] = x[3]
    dx[3] = -x[2]/(tau_1*tau_2) - x[3]*(tau_1+tau_2)/(tau_1*tau_2) + u[1]/(tau_1*tau_2)
    dx[4] = x[5]
    dx[5] = (-torque_flexor(a,b,x[4],Fm_1) - torque_extensor(a,b,x[4],Fm_2) + T_stop_1 + T_stop_2 - B*x[5] )/I 

    return dx


def Jointmodel(mMF, mMX, dt):

    Fs = 1000.0/dt 
    h = 1/Fs
    duration = len(mMF)
    time_sim = np.arange(0,duration/Fs,step = 1/Fs)

    # Segment length (m)
    L = 0.01 
    # Distance between the rotational axis and the center of the segment (m)
    d = 0.001
    # Segment mass (kg)
    M = 0.0004
    # Segment inertia (kg*m^2)
    I = 1/12.0*np.power(L,2)*M + np.power(d,2)*M
    # Viscosity (N*m*s)
    B = 0.00001
    # Distance from the rotation axis to the muscle origin (m)
    a = 0.018 
    # Distance from the rotation axis to the muscle insertion (m)
    b = 0.001

    # Minimum joint angle (radians)
    theta_min = 40*np.pi/180.0
    # Maximum joint angle (radians)
    theta_max = 110*np.pi/180.0
    # Positive angle change => dorsiflexion 
    
    # Maximum muscle force (N)
    F0_1 = 2.13
    F0_2 = 2.13
    # Optimal muscle fiber length (m)
    Lce0_1 = 0.473*0.01 # Average fiber length of all dorsiflexors 
    Lce0_2 = 0.473*0.01 #0.389*0.01; # Average fiber length of all plantarflexors 
    # Maximum muscle fiber legnth at the anatomical limit of range of motion (i.e. minimum and maximum joint angles) (L_0)  
    Lmax_1 = 1.1;
    Lmax_2 = 1.1;

    # Muscle time constant (ms)
    tau_1 = 0.01;
    tau_2 = 0.01;

    # Moment arm (m)
    r_m1 = 0.001 #0.0006
    r_m2 = -0.001 #-0.0015
    
    # Length of flexor muscle at theta_min when it is most stretched 
    L_1_max = muscle_length_flexor(a,b,theta_max)
    # Length of muscle fiber at the joint angle
    Lm_max_1 = Lce0_1*Lmax_1
    # Compute the length of tendon/aponeurosis 
    Lt_1 = L_1_max - Lm_max_1

    # Length of flexor muscle at theta_min when it is most stretched 
    L_2_max = muscle_length_extensor(a,b,theta_min)
    # Length of muscle fiber at the joint angle
    Lm_max_2 = Lce0_2*Lmax_2
    # Compute the length of tendon/aponeurosis 
    Lt_2 = L_2_max - Lm_max_2

    

    # Length of flexor muscle at theta_min when it is most stretched 
    L_1_max = muscle_length_flexor(a,b,theta_max)
    # Length of muscle fiber at the joint angle
    Lm_max_1 = Lce0_1*Lmax_1
    # Compute the length of tendon/aponeurosis 
    Lt_1 = L_1_max - Lm_max_1

    # Length of flexor muscle at theta_min when it is most stretched 
    L_2_max = muscle_length_extensor(a,b,theta_min)
    # Length of muscle fiber at the joint angle
    Lm_max_2 = Lce0_2*Lmax_2
    # Compute the length of tendon/aponeurosis 
    Lt_2 = L_2_max - Lm_max_2
    
    # Find the equilibrium joint angle at rest
    theta_vec = np.linspace(theta_min,theta_max,1000)
    Lce_1 = np.zeros(len(theta_vec))
    Fpe_1 = np.zeros(len(theta_vec))
    T_1 = np.zeros(len(theta_vec))
    Lce_2 = np.zeros(len(theta_vec))
    Fpe_2 = np.zeros(len(theta_vec))
    T_2 = np.zeros(len(theta_vec))
    for i in range(0,len(theta_vec)):
        L_1 = muscle_length_flexor(a,b,theta_vec[i])
        Lce_1[i] = (L_1 - Lt_1)/Lce0_1
        Fpe_1[i] = Fpe_function(Lce_1[i]/Lmax_1,0)*F0_1
        T_1[i] = torque_flexor(a,b,theta_vec[i],Fpe_1[i])

        L_2 = muscle_length_extensor(a,b,theta_vec[i])
        Lce_2[i] = (L_2 - Lt_2)/Lce0_2
        Fpe_2[i] = Fpe_function(Lce_2[i]/Lmax_2,0)*F0_2
        T_2[i] = torque_extensor(a,b,theta_vec[i],Fpe_2[i])

    eq_theta = theta_vec[np.argmin(abs(abs(T_1)-abs(T_2)))]

    u_1 = mMF.reshape(-1,1)
    u_2 = mMX.reshape(-1,1)

    # Initialize parameters
    theta = eq_theta
    dtheta = 0
    Lm_1 = Lce_1[np.argmin(abs(abs(T_1)-abs(T_2)))]
    Lm_1 = Lm_1*Lce0_1
    Lm_2 = Lce_2[np.argmin(abs(abs(T_1)-abs(T_2)))]
    Lm_2 = Lm_2*Lce0_2
    x = np.array([0.0, 0.0, 0.0, 0.0, theta, dtheta])
    dx = np.zeros(6)
    x_mat = np.zeros((int(duration),len(x)))

    for t in range(int(duration)):
        u = np.concatenate(([u_1[t]],[u_2[t]]))
        f1 = forward_dynamis(x,dx,u,I,B,a,b,Lce0_1,Lce0_2,F0_1,F0_2,Lmax_1,Lmax_2,Lt_1,Lt_2)
        x1 = x + f1*h/2
        f2 = forward_dynamis(x1,dx,u,I,B,a,b,Lce0_1,Lce0_2,F0_1,F0_2,Lmax_1,Lmax_2,Lt_1,Lt_2)
        x2 = x + f2*h/2
        f3 = forward_dynamis(x2,dx,u,I,B,a,b,Lce0_1,Lce0_2,F0_1,F0_2,Lmax_1,Lmax_2,Lt_1,Lt_2)
        x3 = x + f3*h
        f4 = forward_dynamis(x3,dx,u,I,B,a,b,Lce0_1,Lce0_2,F0_1,F0_2,Lmax_1,Lmax_2,Lt_1,Lt_2)
        x = x + (f1+2*f2+2*f3+f4)*h/6
        x_mat[t,:] = x

    return time_sim, x_mat