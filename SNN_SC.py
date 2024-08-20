# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 11:11:14 2020
@author: Arezoo Alizadeh
"""

#%% Load libraries

from brian2 import *
import numpy as np
import pylab as plt
import sys
import gc
import pickle
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 
from datetime import datetime
from scipy.signal import convolve as scipy_convolve


#%% Input Current


def FEFcurrent(F0, d2, t,sigma,targetpos, Noise=False):
    
    """
    Generates the current to drive the frontal eye field neurons    
    """

    beta=0.03
    gamma = beta*60
    curr = np.zeros((len(d2), len(t)))
    gaussianfunc = F0*exp(-d2/(2*sigma**2))
    gammafunc = ((t/msecond)**gamma)*exp(-(t/msecond)*beta)
    if Noise:
        curr = gammafunc[:, np.newaxis]*gaussianfunc
        noise = np.random.normal(0, 0.05, (len(t), len(d2)))
        curr = curr + noise*curr
        curr = curr.clip(min=0)
    else:
        curr = gammafunc[:, np.newaxis]*gaussianfunc
        delay=np.zeros((5000,len(d2)))
        curr = np.concatenate([delay,curr])
    return curr
    

#%% Gaussian Distribution

def Gaussian(dist2, sigma, k):
    return k*exp(-dist2/(2*sigma**2))

#%% Smooth firing rate

def frate(st, window_width=8*ms, dt=defaultclock.dt):
    width_dt = int(window_width/dt)
    window = exp(-arange(-4 * width_dt, 4 * width_dt + 1) ** 2 * 1. / (2 * (width_dt) ** 2))
    rate = zeros(int(600./(dt/(1*ms))))
    if len(st)>0:
        st = [int(x/dt) for x in st]
        rate[st] = 1./dt
    return scipy_convolve(rate, window * 1. / sum(window), mode='same') # maybe trim_zeros can improve trailing zeros

#%%
def strainplot(st):
    figure()
    rate = frate(st, window_width=5*ms, dt=defaultclock.dt)
    plot(st/defaultclock.dt, np.ones(len(st))*(max(rate)), '.')
    plot(rate)
    
    
#%% Exponential Fit

def expfit(x, a, b, c):
    return a*np.exp(b*x)+c
    
def expfitinv(x, a, b, c):
    return np.log((x-c)/a)/b

    
def mapping(direction, amplitude, A=3, Bx=1.4, By=1.8):
    '''
    returns position of a neuron in cartesian coordinates
    log-polar afferent mapping
    '''
    theta = np.radians(direction)
    r = amplitude
    return Bx*log(sqrt(((r**2)+(2*A*r*cos(theta))+(A**2)))/A)#, By*arctan((r*sin(theta))/(r*cos(theta)+A))  

def dist2(pos1, pos2):
    '''
    returns the distance between two neuron positions
    '''
    return (pos1-pos2)**2  # why not sqrt((pos1-pos2)**2)

def tISI(st):
    if len(st) > 1:
        tISI = st[1:]-st[:-1]
        return tISI
    else:
        return [0]
        


#%% Input neuron's parameters (frontal eye field neurons)
C = 50*pF               # Membrane Capacitance (pF)
gL = 2*nS               # Leak Conductance (nS)
DeltaT = 2*mV           # Spike Slope Factor (mV)
taum = C/gL             # Membrane Time Constant (pF/nS=ms)
EL = -70*mV             # Leak Reversal Potential (mV)
Vr = -55*mV             # Reset potential (mV)
VT = -50*mV             # Spike Threshold (mV)
#Vcut = VT+5*DeltaT      # Spiking Threshold (mV)
Vcut = -30*mV      # Spiking Threshold (mV)
a = 0*nS                # Subthreshold Adaptation (nS)
b = 60*pA               # Spike-triggered Adaptation (nA)
tauw = 30*ms            # Adaptation Time Constant (ms)

# tauw  : Adaptation Time Constant (ms)
# a     : Subthreshold Adaptation (nS)
# b     : Spike-triggered Adaptation (pA)
# Vr    : Resting potential (mV)
# I     : Total Current (pA)


#%% Adex neuron model at input layer

eqs1="""
dvm/dt=(gL*(EL-vm)+gL*DeltaT*exp((vm-VT)/DeltaT)+I-w)/C : volt
dw/dt=(a*(vm-EL)-w)*(1./tauw)                           : amp
I = I_FEF(t,i)           		                        : amp
"""
            		                 
reset1="""
vm=Vr;
w+=b
"""

#inputNeuron = NeuronGroup(N, model=eqs1, threshold='vm>Vcut', reset=reset1)
#I_FEF = 0*amp

#%% SC parameters: bursting testing phase
C2 = 280*pF              # Membrane Capacitance (pF)
gL2 = 10*nS              # Leak Conductance (nS)
DeltaT2 = 2*mV           # Spike Slope Factor (mV)
taum2 = C2/gL2           # Membrane Time Constant (pF/nS=ms)
EL2 = -70*mV           # Leak Reversal Potential (mV)
VT2 = -50*mV           # Spike Threshold (mV)
Vr2 = VT2+5*mV           # Reset potential (mV)
#Vcut2 = VT2+5*DeltaT2    # Spiking Threshold (mV)
Vcut2 = -30*mV    # Spiking Threshold (mV)
a2 = 4*nS                # Subthreshold Adaptation (nS)
b2 = 0.08*nA             # Spike-triggered Adaptation (nA)
#tauw2 = 100*ms          # Adaptation Time Constant (ms)

Ee=0*mV                 # Excitatory Reversal Potential (mV)
Ei=-80*mV               # Inhibitory Reversal Potential (mV)
taue=5*ms               # Excitatory Conductance Decay (ms)
taui=10*ms              # Inhibitory Conductance Decay (ms)

#%% Adex neuron model at output layer

eqs2="""
dvm/dt=(gL2*(EL2-vm)+gL2*DeltaT2*exp((vm-VT2)/DeltaT2)+Isyn-w)/C2           : volt
dw/dt=(a2*(vm-EL2)-w)*(1./tauw2)                                            : amp
tauw2                                                                       : second
dge/dt=-ge*(1./taue)                                                        : siemens
dgi/dt=-gi*(1./taui)                                                        : siemens
Isyn = ge*(Ee-vm)+gi*(Ei-vm)                                                : amp  
"""

reset2="""
vm=Vr2;
w+=b2
"""

## neural network parameters

N=200  # number of neurons

#%% connectivity parameters
wmin=1
wmax=16
wstep=100
tauwmin=1
tauwmax=200
tauwstep=100 
sigmainh=0.7
sigmaexc=0.2
winh=1150e-3
wexc=160e-3
conscheme='lateral'
tdir=0


duration=550*ms # duration of stimuli
tt = linspace(0*second, duration, int(duration/defaultclock.dt))
defaultclock.dt = 0.01*ms
#%% Spiking neural network model

def saccadesim(tdir, tamp, sigmainh, sigmaexc, winh, wexc, conscheme='lateral', detail=True):

   
    for sigma in arange(1,28,2):
        
        start_scope()   # Brian simulator reset

        inputNeuron = NeuronGroup(N, model=eqs1, threshold='vm>Vcut', reset=reset1,method='euler')
        
        SCNeuron = NeuronGroup(N, model=eqs2, threshold='vm>Vcut2', reset=reset2, method='euler')
        
        targetpos = mapping(tdir, tamp)
        pos = np.linspace(0,5, N)
        distance2 = dist2(targetpos, pos)
        centreid = where(distance2==min(distance2))[0]
                 
        T1=55
        # tauws = (1+alpha)* np.linspace(T1,T1/60,N)
        weights = np.linspace(W1,W2, N)           
#            
        
        Synaptic_model="""
        dge/dt = -ge*(1/taue)                              :siemens 
        dgi/dt = -gi*(1/taui)                              :siemens 
        Isyn_post = ge*(Ee-vm)+gi*(Ei-vm)                  :amp (summed)  
        weight: siemens    
        """        
        
        conFEFSC= Synapses(inputNeuron,SCNeuron,'weight_syn: siemens', on_pre='ge+=weight_syn')
        conFEFSC.connect(j='i')
    
        conFEFSC.weight_syn = weights*wamp
        SCNeuron.tauw2 = tauws*ms
    
        latmap = np.append(pos[::-1], pos[1::])
        latmap = latmap**2
        #plot(latmap)
    
        if conscheme == 'lateral':
            conSCSCinh = Synapses(SCNeuron, SCNeuron, 'weight_inh :siemens', on_pre='gi+=weight_inh')
            conSCSCinh.connect()
            inhw = -Gaussian(latmap, sigmainh,winh)+1

            inhw[N-1]=0                
            for iii in range(N):
                conSCSCinh.weight_inh[iii,:] = inhw[N-1-iii:2*N-1-iii:]*(1-0.04*pos**2)*wamp*(1+alpha) ## considering synaptic scale for lateral connection (micro stimulation paper 2019)
                       
         ## excitatory connection   
            conSCSCexc = Synapses(SCNeuron, SCNeuron,'weight_exc : siemens', on_pre='ge+=weight_exc')
            conSCSCexc.connect()                
            excw = Gaussian(latmap, sigmaexc, wexc)
            excw[N-1]=0
            for iii in range(N):
#                    conSCSCexc.weight_exc[iii,:] = excw[N-1-iii:2*N-1-iii:]*wamp
                conSCSCexc.weight_exc[iii,:] = excw[N-1-iii:2*N-1-iii:]*(1-0.04*pos**2)*wamp*(1+alpha)


        current = FEFcurrent(0, distance2, tt,sigma, targetpos, Noise=False)
        I_FEF = TimedArray(current*amp,dt=0.01*ms) # importing input
        run(50*ms, report_period=20*second)
        
        inputSpikes = SpikeMonitor(inputNeuron) # reading output
        SCSpikes = SpikeMonitor(SCNeuron)       # reading output
    
        if detail:
            cV_SC = StateMonitor(SCNeuron, 'vm', record=[centreid])
            cw_SC = StateMonitor(SCNeuron, 'w', record=[centreid])
            cge_SC = StateMonitor(SCNeuron, 'ge', record=[centreid])
            cgi_SC= StateMonitor(SCNeuron, 'gi', record=[centreid])
            
#                
    #    reinit_default_clock()
        current = FEFcurrent(F0, distance2, tt,sigma, targetpos, Noise=False)
        I_FEF = TimedArray(current*amp,dt=0.01*ms)
        run(duration, report='text', report_period=20*second)
    
        
        i_FEF,t_FEF=inputSpikes.it    
        spikes_dict_FEF=inputSpikes.spike_trains()
        i_SC,t_SC=SCSpikes.it    
        spikes_dict_SC=SCSpikes.spike_trains()
        
        inpSpi = {}
        outSpi = {}
        for n in range(N):
            inpSpi[n] = spikes_dict_FEF[n]
            outSpi[n] = spikes_dict_SC[n]
            
 

        names='saccadesim'
        pickle.dump([inpSpi,outSpi,centreid],open(names,'wb'))
    
    return inpSpi, outSpi, centreid

    
mu = lambda x: x/gL2
Ast = lambda mu: mu+EL2-Vr2
log1 = lambda AST: log((DeltaT2*exp((Vcut2-VT2)/DeltaT2)+AST)/volt)
log2 = lambda AST: log((DeltaT2*exp((Vr2-VT2)/DeltaT2)+AST)/volt)
T = lambda x: (taum2*DeltaT2/Ast(mu(x)))*(((Vcut2-Vr2)/DeltaT2)-log1(Ast(mu(x)))+log2(Ast(mu(x))))

def grid(beta, F0, wmin, wmax, wstep, tauwmin, tauwmax, tauwstep):
    start_scope()
    defaultclock.dt = 0.01*ms
    duration = 550*ms
    t = linspace(0*second, duration, int(duration/defaultclock.dt))

    inputNeuron = NeuronGroup(N, model=eqs1, threshold='vm>Vcut', reset=reset1, method='euler')
    current = FEFcurrent(0, np.array([0]), t,sigma, targetpos)

    I_FEF = TimedArray(current*amp,dt=0.01*ms)

    weights = linspace(wmin, wmax, wstep)
    tauws   = linspace(tauwmin, tauwmax, tauwstep)
    
    numspikes = np.zeros((len(weights), len(tauws)))
    sacdur = np.zeros((len(weights), len(tauws)))
    peakrate = np.zeros((len(weights), len(tauws)))
    N = len(weights)*len(tauws)

    SCNeuron = NeuronGroup(N, model=eqs2, threshold='vm>Vcut2', reset=reset2, method='euler')
    conFEFSC= Synapses(inputNeuron,SCNeuron,'weight_syn : siemens',on_pre='ge+=weight_syn')
    conFEFSC.connect()
    
    weightind=[]
    tauwind=[]

    for i in range(N):
        weightind1, tauwind1 = divmod(i, len(tauws))
        weightind.append(weightind1)
        tauwind.append(tauwind1)
            
    SCNeuron.tauw2= tauws[tauwind]*ms
    conFEFSC.weight_syn[0,:]=weights[weightind]*wamp  

    run(50*ms)
#    reinit_default_clock()

    # Monitors
    inputSpikes = SpikeMonitor(inputNeuron)
    outputSpikes = SpikeMonitor(SCNeuron)
     
    print ("Simulation Starts!")
#    reinit_default_clock()
    current = FEFcurrent(F0, np.array([0]), t)
    I_FEF = TimedArray(current*amp,dt=0.01*ms) 
    run(duration, report='text')

    i_FEF,t_FEF=inputSpikes.it    
    spikes_dict_FEF=inputSpikes.spike_trains()
    i_SC,t_SC=outputSpikes.it    
    spikes_dict_SC=outputSpikes.spike_trains()


    spikes = {}
    out = {}
    print ("Simulation complete!\n")
    for i in range(N):
              
        sys.stdout.write("Progress: %d%%   \r" % (i*100./N) )
        sys.stdout.flush()              
        weightind, tauwind = divmod(i, len(tauws))       
        numspikes[weightind, tauwind] = len(spikes_dict_SC[i])        
#        startTime = datetime.now()
        peakrate[weightind, tauwind] = max(frate(spikes_dict_SC[i]))
#        print(datetime.now() - startTime)       
        spikes[i] = spikes_dict_SC[i]       
#    strainplot(spikes_dict_FEF[80])
  
    return numspikes, peakrate, spikes, current

def neurondetails(beta, F0, weight, tww):
    start_scope()
    defaultclock.dt=0.01*ms
    duration = 500*ms
    t = linspace(0*second, duration, int(duration/defaultclock.dt))
    current = FEFcurrent(beta, F0, np.array([0]), t)

    inputNeuron = NeuronGroup(1, model=eqs1, threshold=Vcut, reset=reset1)
    inputNeuron.I_FEF = 0*amp

    SCNeuron = NeuronGroup(1, model=eqs2, threshold=Vcut2, reset=reset2)
    
   
    conFEFSC= Synapses(inputNeuron,SCNeuron, Synaptic_model,on_pre='ge+=weight')
    SCNeuron[0].tauw2 = tww*ms  # what is tww?
    con[0,0] = weight*wamp

    print ("Ignore first spikes!")
    run(3*ms)
#    reinit_default_clock()

    # Monitors
    inputSpikes = SpikeMonitor(inputNeuron)
    outputSpikes = SpikeMonitor(SCNeuron)
    SCw = StateMonitor(SCNeuron, 'w', record=True)
    SCv = StateMonitor(SCNeuron, 'vm', record=True)    
    SCge = StateMonitor(SCNeuron, 'ge', record=True)

    print ("Simulation Starts!")
    inputNeuron.I_FEF = TimedArray(current*amp,ft=0.01*ms)
    run(duration, report='text')
    
    strainplot(inputSpikes[0])

    return inputSpikes, outputSpikes, SCw, SCv, SCge

def getConnections(wmin, wmax, wstep, tauwmin, tauwmax, tauwstep, visual=True):
    '''
    Crude search for peak firing rates and number of spikes
    in the parameter space of synaptic weights and adaptation time constant
    generates a grid of wstep*tauwstep neurons
    with linearly distributed values between wmin-wmax and tauwmin-tauwmax
    feeds all with the same FEF spike train
    and checks the SC spike trains for different w&tauw values    
    '''
    
    start_scope()
    defaultclock.dt=0.01*ms
#    delay=28*ms
    max_cur=arange(800,810,10)   
    mag_list=800/max_cur


    
    for i,mag in enumerate(mag_list):
        ns, pr, spi, cur = grid(beta, F0, wmin, wmax, wstep, tauwmin, tauwmax, tauwstep,mag)
        names = "grid"+'-I'+str(max_cur[i])+'.pickle'
        pickle.dump([ns,pr,spi,cur],open(names,'wb'))

        tauws = np.linspace(tauwmin, tauwmax, tauwstep)
        ws = np.linspace(wmin, wmax, wstep)
        wind = where(ns==20)[0]
        tauwind = where(ns==20)[1]
        if len(wind)>0:
            
            wf = ws[wind]
            tauwf = tauws[tauwind]
        
            fit = polyfit(tauwf,wf, 2)   # A: it should be polynomial curve
            fit_fn = poly1d(fit)  

        
            
            if visual:    
                WS, TAUWS = np.meshgrid(ws, tauws)
                fig = plt.figure(figsize=(12,9))
        
                V = arange(300, 900, 50)
                CS = plt.contour(WS, TAUWS, pr.T, V)
                plt.clabel(CS, inline=1, fontsize=13)
                x2 = plot(ws[wind], tauws[tauwind], '.', label='20 spikes')
        
                #plot(wf, fit_fn(wf), '.k')
                plt.plot(fit_fn(tauwf), tauwf, '.k')
                plt.xlabel('Top-down projections (nS)',fontsize=40)
                plt.ylabel('Adaptation time Constant (ms)',fontsize=40);
                plt.rc('xtick', labelsize=30) 
                plt.rc('ytick', labelsize=30) 
                plt.savefig("W&tauw"+"I"+str(max_cur[i]))
        
            names = "getConnections"+str(max_cur[i])+'.pickle'
        #    with open(name, 'wb') as f:
            pickle.dump([fit_fn],open(names,'wb'))

    return fit_fn



def simulate(sigmainh=1.2, sigmaexc=0.4, winh=50e-3, wexc=160e-3, conscheme='lateral', dump=False):
    
    targetamp = np.array([2, 5, 9, 14, 20, 27, 35])


    inp = {}
    out = {}
    cent = np.zeros(len(targetamp))
    totspi = np.zeros(len(targetamp))
    actneurons = np.zeros(len(targetamp))
    peakFR = np.zeros(len(targetamp))
    centspi = np.zeros(len(targetamp))
    
    for i, val in enumerate(targetamp):
        inp[i], out[i], cent[i] = saccadesim(0, val, sigmainh, sigmaexc, winh, wexc, conscheme, detail=dump)
        totspi[i] = 0
        actneurons[i] = 0
        for j in range(len(out[i])):
            totspi[i] += len(out[i][j])
            if len(out[i][j]) != 0:
                actneurons[i] += 1
        peakFR[i] = max(frate(out[i][cent[i]]))
        centspi[i] = len(out[i][cent[i]])
        print( 'central cell fired: '+str(centspi[i])+' spikes, peak firing rate: '+str(peakFR[i])+' total # of spikes: '+str(totspi[i])+' from '+str(actneurons[i])+' cells')
    
    if dump:

       pickle.dump([targetamp, inp, out, cent, totspi, actneurons, peakFR, centspi], open('simulate.pickle','wb'))

    return inp, out, cent, totspi, actneurons, peakFR, centspi


def diffcurrent(winh, max_cur):
    
    neurons_spi_currents = []
    neurons_spi = np.zeros(200)
    totspi = np.zeros(len(max_cur))
    actneurons = np.zeros(len(max_cur))
    peakFR_cent = np.zeros(len(max_cur))
    centspi = np.zeros(len(max_cur))
    
    for i,value in enumerate(max_cur):
        names = 'saccadesim_'+'I='+str(value)+'winh=0.1'+'.pickle'
        [i_FEF,t_FEF,inpSpi,outSpike,centreid] = pickle.load(open(names,'rb'))
        
        outSpi = list(outSpike.values())
    
        for n in arange(0,200):
            neurons_spi[n] = len(outSpi[n])
            totspi[i] += len(outSpi[n])
           
            if len(outSpi[n]) != 0:
               actneurons[i] += 1
           
        neurons_spi_currents.append(neurons_spi)    
        peakFR_cent[i] = max(frate(outSpi[centreid[0]]))
        centspi[i] = len(outSpi[centreid[0]])
        
        print( 'Current='+str(value)+'  central cell fired:'+str(centspi[i])+'  spikes, central neuron''s peak firing rate:'+ str(peakFR_cent[i])+'  total # of spikes:'+ str(totspi[i])+'  from '+ str(actneurons[i])+' cells')
        
    name = 'diffcurrent'+'.pickle'
    pickle.dump([max_cur,totspi, actneurons, peakFR_cent, centspi,neurons_spi_currents], open(name,'wb'))
        
        
        ######## plot ##############
    plot(totspi,'o',label='# of spike',markersize=15)
    plot(centspi*10,'o',label='10 * # of central neuron spike',markersize=15)
#        plot(peakFR_cent,label='Central neuron peak firing rate',linewidth=5)
    plot(actneurons*10,'o',label='10 * # of active neuron',markersize=15)
#        plot(neurons_spi_currents,label='')        
    plt.xlabel('FEF Input ID', fontsize=50) 
    plt.ylabel('# of spike', fontsize=50) 
    plt.rc('xtick', labelsize=40) 
    plt.rc('ytick', labelsize=40) 
    plt.legend(fontsize=30)

       

def ratesfig(out, cent):
    for i in range(len(cent)):
        st = out[i][cent[i]]
        rate = frate(st)
        plot(st/defaultclock.dt, np.ones(len(st))*(max(rate)), '.')
        text(16000, max(rate), str(len(st))+'spikes')
        plot(rate)
#    xlim([7500, 20000])

def rasterit(out, c='b'):
    for i in range(len(out)):
        plot(out[i], np.ones(len(out[i]))*i, '.'+c)

def firings(out, cent):
    for i in range(20):
        plot(frate(out[cent+i]))
        plot(frate(out[cent-i]))
#    xlim([7000, 17000])

def findweights():
    for i in np.linspace(3, 10, 29):
        for j in np.linspace(3, 10, 29):
            if i>j:
                name = 'si'+str(1.5)+'se'+str(0.5)+'wi'+str(j)+'we'+str(i)
                print ('\n'+name+' Started!')
                inp, out, cent, totspi, actneurons = simulate(winh=j*1e-3, wexc=i*1e-3)
                figure()
                ratesfig(out, cent)
            #xlim([3000, 20000])
                savefig('img/'+name+'rates.png')
                figure()
                rasterit(out[1], 'r')
                rasterit(out[4], 'b')
                rasterit(out[6], 'g')
            #xlim([0.03, 0.2])
                ylim([0,200])
                savefig('img/'+name+'raster.png')
                figure()
                firings(out[2], cent[2])
                savefig('img/'+name+'scaled'+str(cent[2])+'.png')
                figure()
                firings(out[4], cent[4])
                savefig('img/'+name+'scaled'+str(cent[4])+'.png')
                figure()
                firings(out[5], cent[5])
                savefig('img/'+name+'scaled'+str(cent[5])+'.png')
                close('all')
                print (name+' Done!'+'\n\n')

def children(pop, perf):
    newpop = np.zeros(shape(pop))
    pop = pop[perf.argsort()]
    for i in range(len(pop)):
        if i<2:
            newpop[i] = pop[i]
        else:
            p1 = np.random.randint(6)
            p2 = np.random.randint(6)
            while p1==p2:
                p2 = np.random.randint(6)
            p1c = chrom(pop[p1])
            p2c = chrom(pop[p2])
            c1, c2 = crossover(p1c, p2c)
            child1 = val(mutate(c1))
            child2 = val(mutate(c2))
            while any(newpop==child1) and any(newpop==child2):
                c1, c2 = crossover(p1c,p2c)
                child1 = val(mutate(c1))
                child2 = val(mutate(c2))
            if any(newpop==child1):
                newpop[i] = child2
            else:
                newpop[i] = child1

    return newpop

def mutate(chrom):
    if rand()<0.05:
        chrom[np.random.randint(0, len(chrom))] = np.random.randint(0, 10) 
    return chrom
           
def val(chrom):
    ind = np.zeros((2))
    for i in range(15):
        ind[1] += chrom[-i-1]*(10**(i+1))
        ind[0] += chrom[-16-i]*(10**(i+1))
    return ind*1e-13

def chrom(ind):
    chromarray = np.zeros((30))
    for i in range(15):
        chromarray[i] = int(mod(ind[0]*(10**(i-2)), 10))
        chromarray[15+i] = int(mod(ind[1]*(10**(i-2)),10))
    return chromarray

def crossover(chrom1, chrom2):
    copoint = np.random.random_integers(1, len(chrom1)-1)
    child1 = np.append(chrom1[:copoint], chrom2[copoint:])
    child2 = np.append(chrom2[:copoint], chrom1[copoint:])
    return child1, child2


## genetic algorithm

def genetic():
    pop = np.random.rand(10,2)
    pop[:,0] = pop[:,0]*1e3
    pop[:,1] = pop[:,1]*1e3
    pop = sort(pop)

    targetamp = np.array([2, 5, 9, 14, 20, 27, 35])
    peakfiring = Fp(targetamp)
    centralspikes = np.ones(len(targetamp))*20.
    check = 0
    checkprev = 1
    counter = 0
    while abs(check-checkprev)>0.001:
        counter+=1
        perfpop = np.zeros((10,5))
        checkprev = check
        for j, ind in enumerate(pop):
            print('\nGENERATION :: '+str(counter)+' INDIVIDUAL :: '+str(j)+' VAL :: '+str(pop[j]))
            inp, out, cent, totspi, actneuron, peakfr, centspi = simulate(winh=ind[0]*1e-3 , wexc=ind[1]*1e-3)
            rmspeak = sqrt(mean_squared_error(peakfiring, peakfr))
            rmscentspi = sqrt(mean_squared_error(centspi, centralspikes))
            corr = zeros(len(targetamp))
            for sim, central in enumerate(cent):
                frs = np.zeros((21, 60000))
                frs[10] = frate(out[sim][central])
                #peakfrtiming = np.zeros((31))
                #peakfrtiming[15] = find(frs[15]==max(frs[15]))
                for i in range(10):
                    frs[i+11] = frate(out[sim][central+i+1])
                    #peakfrtiming[i+16] = find(frs[i+16]==max(frs[i+16]))[0]
                    frs[9-i] = frate(out[sim][central-i-1])
                    #peakfrtiming[14-i] = find(frs[14-i]==max(frs[14-i]))[0]
                corr[sim] = mean(corrcoef(frs)[10,:])
            correx = np.ones(shape(corr))
            rmscorr = sqrt(mean_squared_error(corr, correx))
            
       
            
            perfpop[j,0] = rmspeak
            perfpop[j,1] = rmscentspi
            perfpop[j,2] = rmscorr
            
            
            perfpop[j,3] = 0.1*rmspeak+10*rmscentspi+100*rmscorr
            
            print('GENERATION :: '+str(counter)+' INDIVIDUAL :: '+str(j)+' SCORE :: '+str(perfpop[j,3]))
            print('PERFPOP :: '+str(perfpop[j,:]))
        idx = perfpop[:,3].argsort()[0]
        check = perfpop[:,3][idx]
        print ('Population :: \n')
        print (pop)
        print ('Performance :: \n')
        print (perfpop)

        print( '\nIndividual: '+str(idx)+' in generation '+str(counter)+' performed best! score:'+str(check)+'\n Best variables for (w_inh, w_exc)'+str(pop[idx]))
        pop = children(pop, perfpop[:,3])


sigma00 = 3        # ms 
F00 = 800          # spikes/s
beta00 = 0.07       # ms/deg
def Fp(R):        # spikes/s
    return F00/sqrt(1+(beta00*R))
def sigmadur(R):  # ms
    return sigma00*(1+(beta00*R))
def T0(R, gamma): # ms
    return sigmadur(R)*gamma/e
    newpop = np.zeros(shape(pop))
def gamma(R):     # 1
    return 30/sigmadur(R)
index = lambda wind, tauwind: wind*tauwstep+tauwind


def runThisCode():
    getConnections(visual=True)

