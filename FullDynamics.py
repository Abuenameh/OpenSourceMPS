import time

import sys
import MPSPyLib as mps
import numpy as np
import concurrent.futures
import subprocess
import os
import datetime
# import gtk
# import gobject
import matplotlib.pyplot as plt
from mathematicaosmps import mathformat
from matplotlib import cm
from speed import gprogress
from speed import progress
import pylab

start = time.time()

numthreads = 6

def execute(command):
    """ Execute the given command on the command line."""
    # print subprocess.list2cmdline(command)
    return subprocess.call(subprocess.list2cmdline(command), shell=True)

def runmps(infile):
    appname='Execute_MPSMain'
    cmdline = [appname]
    cmdline += [infile]
    execute(cmdline)


def run(MainFiles):
    count = 0;
    with concurrent.futures.ThreadPoolExecutor(max_workers=numthreads) as executor:
        futures = [executor.submit(runmps, infile) for infile in MainFiles]
        for future in progress(concurrent.futures.as_completed(futures), size=len(futures)):
            # print count
            # count += 1
            pass
    gtk.main_quit()


N = 1000
g13 = 2.5e9
g24 = 2.5e9
Delta = -2.0e10
alpha = 1.1e7
eps = 0
delta = 1e12

Ng = np.sqrt(N) * g13;

Wi = 7.9e10
Wf = 1.1e12

def JW(W):
    lenW = len(W)
    J = np.zeros(lenW)
    for i in range(0, lenW-2):
        J[i] = alpha * W[i] * W[i+1] / (np.sqrt(Ng * Ng + W[i] * W[i]) * np.sqrt(Ng * Ng + W[i+1] * W[i+1]))
    J[lenW-1] = alpha * W[lenW-2] * W[lenW-1] / (np.sqrt(Ng * Ng + W[lenW-2] * W[lenW-2]) * np.sqrt(Ng * Ng + W[lenW-1] * W[lenW-1]))
    # J[lenW-1] = alpha * W[lenW-1] * W[0] / (np.sqrt(Ng * Ng + W[lenW-1] * W[lenW-1]) * np.sqrt(Ng * Ng + W[0] * W[0]))
    return J


def UW(W):
    return -2*(g24 * g24) / Delta * (Ng * Ng * W * W) / ((Ng * Ng + W * W) * (Ng * Ng + W * W))

def JWi(W):
    return alpha * W ** 2 / (Ng ** 2 + W ** 2)

def UWi(W):
    return -g24 ** 2 / Delta * (Ng ** 2 * W ** 2) / ((Ng ** 2 + W ** 2) ** 2)

PostProcessOnly = int(sys.argv[1])

# if not PostProcessOnly:
#     import gtk
#     import gobject

# Build operators
nmax = 3
Operators = mps.BuildBoseOperators(nmax, nFlavors=4)
Operators['n2'] = np.dot(Operators['nbtotal'], Operators['nbtotal'])
Operators['interaction'] = 0.5 * (np.dot(Operators['nbtotal'], Operators['nbtotal']) - Operators['nbtotal'])
Operators['na'] = Operators['nb_0']
Operators['nS12'] = Operators['nb_1']
Operators['nS13'] = Operators['nb_2']
Operators['nS14'] = Operators['nb_3']
Operators['a'] = Operators['b_0']
Operators['S12'] = Operators['b_1']
Operators['S13'] = Operators['b_2']
Operators['S14'] = Operators['b_3']
Operators['ad'] = Operators['bdagger_0']
Operators['S12d'] = Operators['bdagger_1']
Operators['S13d'] = Operators['bdagger_2']
Operators['S14d'] = Operators['bdagger_3']
B = np.sqrt(Ng**2 + Wi**2)
A = np.sqrt(4*B**2 + delta**2)
Operators['p'] = (1 / B) * (Ng * Operators['S12'] - Wi * Operators['a'])
Operators['pd'] = (1 / B) * (Ng * Operators['S12d'] - Wi * Operators['ad'])
Operators['pp'] = np.sqrt(2/(A*(A+delta)))*(Wi*Operators['S12']+Ng*Operators['a']+((A+delta)/2)*Operators['S13'])
Operators['ppd'] = np.sqrt(2/(A*(A+delta)))*(Wi*Operators['S12d']+Ng*Operators['ad']+((A+delta)/2)*Operators['S13d'])
Operators['pm'] = np.sqrt(2/(A*(A-delta)))*(Wi*Operators['S12']+Ng*Operators['a']-((A-delta)/2)*Operators['S13'])
Operators['pmd'] = np.sqrt(2/(A*(A-delta)))*(Wi*Operators['S12d']+Ng*Operators['ad']-((A-delta)/2)*Operators['S13d'])
Operators['HW'] = np.dot(Operators['S12d'], Operators['S13']) + np.dot(Operators['S13d'], Operators['S12'])
Operators['Hg13'] = np.dot(Operators['S13'], Operators['ad']) + np.dot(Operators['a'], Operators['S13d'])
# Operators['Hg24'] = np.dot(Operators['S14'], Operators['ad']) + np.dot(Operators['a'], Operators['S14d'])
# Operators['Hg24'] = np.dot(Operators['S12d'], np.dot(np.dot(Operators['S14'], Operators['S14']), Operators['ad'])) + np.dot(Operators['a'], np.dot(np.dot(Operators['S14d'], Operators['S14d']), Operators['S12']))
Operators['Hg24'] = np.dot(Operators['S12d'], np.dot(Operators['S14'], Operators['ad'])) + np.dot(Operators['a'], np.dot(Operators['S14d'], Operators['S12']))
Operators['penalty'] = np.dot(Operators['pmd'], Operators['pm'])
Operators['ntotal'] = Operators['na'] + Operators['nS12'] + Operators['nS13'] + 2 * Operators['nS14']
# Operators['F'] = Operators['n2'] - np.dot(Operators['nbtotal'], Operators['nbtotal'])
#Define Hamiltonian MPO
H = mps.MPO()
# H.AddMPOTerm(Operators, 'bond', ['bdagger', 'b'], hparam='J', weight=-1.0)
# H.AddMPOTerm(Operators, 'site', 'interaction', hparam='U', weight=1.0)
H.AddMPOTerm(Operators, 'site', 'nS12', hparam='e', weight=1.0)
H.AddMPOTerm(Operators, 'site', 'nS13', hparam='d', weight=1.0)
H.AddMPOTerm(Operators, 'site', 'nS14', hparam='De', weight=1.0)
H.AddMPOTerm(Operators, 'site', 'HW', hparam='W', weight=1.0)
H.AddMPOTerm(Operators, 'site', 'Hg13', hparam='g', weight=1.0)
H.AddMPOTerm(Operators, 'site', 'Hg24', hparam='g24', weight=1.0)
H.AddMPOTerm(Operators, 'bond', ['ad', 'a'], hparam='a', weight=1.0)
H.AddMPOTerm(Operators, 'site', 'penalty', hparam='pen', weight=1.0)

Operators['p1'] = np.dot(Operators['S12d'], Operators['a']) + np.dot(Operators['ad'], Operators['S12'])

#ground state observables
myObservables = mps.Observables()
#Site terms
# myObservables.AddObservable(Operators, 'nbtotal', 'site', 'n')
# myObservables.AddObservable(Operators, 'n2', 'site', 'n2')
# myObservables.AddObservable(Operators, 'na', 'site', 'na')
# myObservables.AddObservable(Operators, 'nS12', 'site', 'nS12')
# myObservables.AddObservable(Operators, 'nS13', 'site', 'nS13')
# myObservables.AddObservable(Operators, 'nS14', 'site', 'nS14')
# myObservables.AddObservable(Operators, 'p1', 'site', 'p1')
#correlation functions
# myObservables.AddObservable(Operators,['nbtotal','nbtotal'],'corr','nn')
# myObservables.AddObservable(Operators, ['bdagger', 'b'], 'corr', 'spdm')

dynObservables = mps.Observables()
# dynObservables.AddObservable(Operators, 'nbtotal', 'site', 'n')
# dynObservables.AddObservable(Operators, 'n2', 'site', 'n2')

maxsweeps = 7
myConv = mps.MPSConvergenceParameters(max_num_sweeps=maxsweeps)
myConv.AddModifiedConvergenceParameters(0,['max_bond_dimension','local_tol'],[50,1E-14])
# myKrylovConv=mps.KrylovConvergenceParameters(MaxnLanczosIterations=40,lanczos_tol=1E-8)
# myKrylovConv=mps.KrylovConvergenceParameters(MaxnLanczosIterations=40,lanczos_tol=1E-8)
myKrylovConv=mps.KrylovConvergenceParameters()

# Wi = 7.9e10
# Wf = 1.1e12

def func(x):
    if x < 0:
        return 0
    elif x < 0.5:
        return 2 * x ** 2
    elif x < 1:
        return -2*(x-1) ** 2 + 1
    else:
        return 1

def Wt(t):
    # return Wi
    return (Wi - Wf) * func(2*(1 - 1e6*t) - 0.4) + Wf

def Wt2(t):
    return Wt(t+3e-7)

def Ji(t):
    # return JWi(Wi)
    return JWi(Wt(t))

def Ui(t):
    # return UWi(Wi)
    return UWi(Wt(t))

def g24t(t):
    return g24

def et(t):
    return eps

def dt(t):
    return delta

def Det(t):
    return Delta + eps

def at(t):
    return alpha#0#alpha

def gt(t):
    return Ng

def pent(t):
    return 0

Quenches=mps.QuenchList()
# Quenches.AddQuench(H, ['e'], 1e-8, 1e-9, [qwet], ConvergenceParameters=myKrylovConv)
Quenches.AddQuench(H, ['W', 'g24', 'e', 'De', 'a', 'g', 'd', 'pen'], 1e-7, 1e-9, [Wt, g24t, et, Det, at, gt, dt, pent], ConvergenceParameters=myKrylovConv)
# Quenches.AddQuench(H, ['J', 'U'], 1e-7, 1e-9, [Ji, Ui], ConvergenceParameters=myKrylovConv)
# Quenches.AddQuench(H, ['J'], 1, 1e-3, [Jfunc], ConvergenceParameters=myKrylovConv)

# B = np.sqrt(Ng**2 + Wi**2)
# A = np.sqrt(4*B**2 + delta**2)
# Operators['p'] = (1 / B) * (Ng * Operators['S12'] - Wi * Operators['a'])
# Operators['pd'] = (1 / B) * (Ng * Operators['S12d'] - Wi * Operators['ad'])
# Operators['pp'] = np.sqrt(2/(A*(A+delta)))*(Wi*Operators['S12']+Ng*Operators['a']+((A+delta)/2)*Operators['S13'])
# Operators['ppd'] = np.sqrt(2/(A*(A+delta)))*(Wi*Operators['S12d']+Ng*Operators['ad']+((A+delta)/2)*Operators['S13d'])
# Operators['pm'] = np.sqrt(2/(A*(A-delta)))*(Wi*Operators['S12']+Ng*Operators['a']-((A-delta)/2)*Operators['S13'])
# Operators['pmd'] = np.sqrt(2/(A*(A-delta)))*(Wi*Operators['S12d']+Ng*Operators['ad']-((A-delta)/2)*Operators['S13d'])
Operators['np'] = np.dot(Operators['pd'], Operators['p'])
Operators['np2'] = np.dot(Operators['np'], Operators['np'])
Operators['npp'] = np.dot(Operators['ppd'], Operators['pp'])
Operators['npm'] = np.dot(Operators['pmd'], Operators['pm'])
Operators['na2'] = np.dot(Operators['na'], Operators['na'])
Operators['nS122'] = np.dot(Operators['nS12'], Operators['nS12'])
Operators['nS132'] = np.dot(Operators['nS13'], Operators['nS13'])
Operators['nS142'] = np.dot(Operators['nS14'], Operators['nS14'])
myObservables.AddObservable(Operators, 'np', 'site', 'np')
myObservables.AddObservable(Operators, 'npp', 'site', 'npp')
myObservables.AddObservable(Operators, 'npm', 'site', 'npm')
myObservables.AddObservable(Operators, 'np2', 'site', 'np2')
myObservables.AddObservable(Operators, 'na', 'site', 'na')
myObservables.AddObservable(Operators, 'nS12', 'site', 'nS12')
myObservables.AddObservable(Operators, 'nS13', 'site', 'nS13')
myObservables.AddObservable(Operators, 'nS14', 'site', 'nS14')
myObservables.AddObservable(Operators, 'na2', 'site', 'na2')
myObservables.AddObservable(Operators, 'nS122', 'site', 'nS122')
myObservables.AddObservable(Operators, 'nS132', 'site', 'nS132')
myObservables.AddObservable(Operators, 'nS142', 'site', 'nS142')
dynObservables.AddObservable(Operators, 'np', 'site', 'np')
dynObservables.AddObservable(Operators, 'np2', 'site', 'np2')
dynObservables.AddObservable(Operators, 'na', 'site', 'na')
dynObservables.AddObservable(Operators, 'nS12', 'site', 'nS12')
dynObservables.AddObservable(Operators, 'nS13', 'site', 'nS13')
dynObservables.AddObservable(Operators, 'nS14', 'site', 'nS14')
dynObservables.AddObservable(Operators, 'na2', 'site', 'na2')
dynObservables.AddObservable(Operators, 'nS122', 'site', 'nS122')
dynObservables.AddObservable(Operators, 'nS132', 'site', 'nS132')
dynObservables.AddObservable(Operators, 'nS142', 'site', 'nS142')


L = 4

parameters = []
parameters.append({
    'job_ID': 'BH_',
    'unique_ID': str(time.time()),#'0',
    'Write_Directory': 'Temp/',
    'Output_Directory': 'Output/',
    'L': L,
    'e': eps,
    'd': delta,
    'De': Delta + eps,
    'a': 0,#alpha,
    'g24': g24,
    'g': Ng,
    'W': Wi,
    'pen': 1e12,
    'Abelian_generators': ['ntotal'],
    'Abelian_quantum_numbers': [L],
    'verbose': 0,
    # 'n_excited_states': 5,
    'MPSObservables': myObservables,
    'eMPSObservables': myObservables,
    'MPSConvergenceParameters': myConv,
    'Quenches': Quenches,
    'DynamicsObservables': dynObservables
})

MainFiles=mps.WriteFiles(parameters,Operators,H,PostProcess=PostProcessOnly)
mps.runMPS(MainFiles)

Outputs = mps.ReadDynamicObservables(parameters)
Outputs2 = mps.ReadStaticObservables(parameters)

print Outputs2[0]['np']
# print Outputs2[0]['npp']
# print Outputs2[0]['npm']
print Outputs2[0]['np2']
print Outputs2[0]['na']
print Outputs2[0]['nS12']
print Outputs2[0]['nS13']
print Outputs2[0]['nS14']
print Outputs2[0]['na2']
print Outputs2[0]['nS122']
print Outputs2[0]['nS132']
print Outputs2[0]['nS142']

print np.sum(Outputs2[0]['np'])+np.sum(Outputs2[0]['npp'])+np.sum(Outputs2[0]['npm'])
print np.sum(Outputs2[0]['na'])+np.sum(Outputs2[0]['nS12'])+np.sum(Outputs2[0]['nS13'])+2*np.sum(Outputs2[0]['nS14'])

# print Outputs
# print Outputs[0]['na']
# print Outputs[0]['nS12']
# print Outputs[0]['nS13']
# print Outputs[0]['nS14']
# print Outputs[0]['np']

# print Outputs2[0]
# for d in Outputs2:
#     print d['energy']
#     print d['np']

# quit()

# print (1 / (Ng**2 + Wi**2))*(Ng**2 * Outputs[0]['nS12'][1] + Wi**2 * Outputs[0]['na'][1] - Ng*Wi*Outputs[0]['p1'][1])
# print (Ng**2 * Outputs[0]['nS12'][1] + Wi**2 * Outputs[0]['na'][1] - Ng*Wi*Outputs[0]['p1'][1])

# quit()

# print Outputs[0][0]['time']
# print Outputs[0][0]['n']
# print Outputs[0][0]['n2']
# print Outputs[0][-1]['time']
# print Outputs[0][-1]['n']
# print Outputs[0][-1]['n2']


t = []
# n = []
# n2 = []
# F = []
np0 = []
np02 = []
na = []
nS12 = []
nS13 = []
nS14 = []
na2 = []
nS122 = []
nS132 = []
nS142 = []
for p in Outputs[0]:
    t.append(p['time'])
    # F.append(p['n2'][1] - p['n'][1] ** 2)
    # n.append(p['np'][1])
    # n2.append(p['np2'][1])
    # F.append(n2[-1] - n[-1]**2)
    # na.append(p['na'][1])
    # nS12.append(p['nS12'][1])
    # F.append(np.array(p['n2']) - np.array(p['n']) ** 2)
    np0.append(p['np'])
    np02.append(p['np2'])
    na.append(p['na'])
    nS12.append(p['nS12'])
    nS13.append(p['nS13'])
    nS14.append(p['nS14'])
    na2.append(p['na2'])
    nS122.append(p['nS122'])
    nS132.append(p['nS132'])
    nS142.append(p['nS142'])

# print t
# print n
# print n2
# print F
# print na
# print nS12
# print na
# print nS12
# print nS13
# print nS14

resi = 16
f = open('res.{0}.txt'.format(resi), 'w')
f.write('t[{0}]={1};\n'.format(resi, mathformat(t)))
# f.write('n[{0}]={1};\n'.format(resi, mathformat(n)))
# f.write('n2[{0}]={1};\n'.format(resi, mathformat(n2)))
# f.write('F[{0}]={1};\n'.format(resi, mathformat(F)))
f.write('np[{0}]={1};\n'.format(resi, mathformat(np0)))
f.write('np2[{0}]={1};\n'.format(resi, mathformat(np02)))
f.write('na[{0}]={1};\n'.format(resi, mathformat(na)))
f.write('nS12[{0}]={1};\n'.format(resi, mathformat(nS12)))
f.write('nS13[{0}]={1};\n'.format(resi, mathformat(nS13)))
f.write('nS14[{0}]={1};\n'.format(resi, mathformat(nS14)))
f.write('na2[{0}]={1};\n'.format(resi, mathformat(na2)))
f.write('nS122[{0}]={1};\n'.format(resi, mathformat(nS122)))
f.write('nS132[{0}]={1};\n'.format(resi, mathformat(nS132)))
f.write('nS142[{0}]={1};\n'.format(resi, mathformat(nS142)))

end = time.time()
runtime = str(datetime.timedelta(seconds=end-start))
print runtime
f.write('runtime[{0}]=\"{1}\";\n'.format(resi, runtime))
f.flush()

npi = np.array(np0)[:,1]
np2i = np.array(np02)[:,1]
nai = np.array(na)[:,1]
nS12i = np.array(nS12)[:,1]
nS13i = np.array(nS13)[:,1]
nS14i = np.array(nS14)[:,1]
na2i = np.array(na2)[:,1]
nS122i = np.array(nS122)[:,1]
nS132i = np.array(nS132)[:,1]
nS142i = np.array(nS142)[:,1]

Fp = np2i - npi**2
Fa = na2i - nai**2
FS12 = nS12i - nS12i**2
FS13 = nS13i - nS13i**2
FS14 = nS14i - nS14i**2

# pylab.plot(t, F)
# pylab.plot(t, n, t, n2, t, F)
# pylab.plot(t, na, t, nS12)
pylab.figure()
pylab.plot(t, nai, t, nS12i, t, nS13i, t, nS14i, t, na2i, t, nS122i, t, nS132i, t, nS142i)
pylab.figure()
pylab.plot(t, Fa, t, FS12, t, FS13, t, FS14)
pylab.figure()
pylab.plot(t, npi, t, np2i)
pylab.figure()
pylab.plot(t, Fp)
pylab.show()

quit()

L = 50
Wlist = [2e10]#[2e10, 2e11]
Nlist = [2]#[1, 2, 3]
Nlist = range(1, 2*L+1)

# jobID = 'BH_' + str(int(time.time())) + '_'
jobID = 'BH_'
id = 0

seed = int(sys.argv[3])
delta = float(sys.argv[4])
np.random.seed(seed);
xi = 1 + 2 * delta * np.random.random(L) - delta

position = {}

parameters = []
for N in Nlist:
    for W in Wlist:
        parameters.append({
            'job_ID': jobID,
            'unique_ID': str(id),
            'Write_Directory': 'Temp/',
            'Output_Directory': 'Output/',
            'L': L,
            'J': JW(W*xi),
            'U': UW(W*xi),
            'Abelian_generators': ['nbtotal'],
            'Abelian_quantum_numbers': [N],
            'verbose': 0,
            'MPSObservables': myObservables,
            'MPSConvergenceParameters': myConv
        })
        position[id] = [Wlist.index(W), Nlist.index(N)]
        id += 1

MainFiles=mps.WriteFiles(parameters,Operators,H,PostProcess=PostProcessOnly)

runtime = np.NaN

if not PostProcessOnly:
    gobject.timeout_add(1000, run, MainFiles)
    gtk.gdk.threads_init()

    start = datetime.datetime.now()
    gtk.main()
    end = datetime.datetime.now()
    runtime = end - start

if True:#PostProcess:
    resi = int(sys.argv[2])
    if sys.platform == 'darwin':
        respath = '/Users/Abuenameh/Documents/SimulationResults/BH-OSMPS/'
    elif sys.platform == 'linux2':
        respath = '/home/ubuntu/Dropbox/Amazon EC2/Simulation Results/BH-OSMPS/'
    elif sys.platform == 'win32':
        respath = 'C:/Users/abuenameh/Dropbox/Server/BH-DMRG/'
    if not os.path.exists(respath):
        try:
            os.makedirs(respath)
        except:
            pass
    resipath = respath + 'res.' + str(resi)
    resfile = resipath + '.txt'
    while os.path.isfile(resfile):
        resi += 1
        resipath = respath + 'res.' + str(resi)
        resfile = resipath + '.txt'
    if sys.platform == 'linux2':
        respath = '/media/ubuntu/Results/BH-MPS/'
        if not os.path.exists(respath):
            try:
                os.makedirs(respath)
            except:
                pass
        resipath = respath + 'res.' + str(resi)
        resfile = resipath + '.txt'
        while os.path.isfile(resfile):
            resi += 1
            resipath = respath + 'res.' + str(resi)
            resfile = resipath + '.txt'
        resipath = respath + 'res.' + str(resi)
        resfile = resipath + '.txt'
    resf = open(resfile, 'w')
    print resfile

    dims = [len(Wlist), len(Nlist)]
    ndims = dims + [L]
    cdims = dims + [L, L]

    Jres = np.zeros(ndims)
    Ures = np.zeros(ndims)

    E0res = np.zeros(dims)
    fcres = np.zeros(dims)
    nres = np.zeros(ndims)
    n2res = np.zeros(ndims)
    cres = np.zeros(cdims)
    nnres = np.zeros(cdims)

    mumres = np.zeros(dims)
    mupres = np.zeros(dims)

    Jres.fill(np.NaN)
    Ures.fill(np.NaN)

    E0res.fill(np.NaN)
    fcres.fill(np.NaN)
    nres.fill(np.NaN)
    n2res.fill(np.NaN)
    cres.fill(np.NaN)
    nnres.fill(np.NaN)

    mumres.fill(np.NaN)
    mupres.fill(np.NaN)

    Outputs = mps.ReadStaticObservables(parameters)

    res = ''
    res += 'Lres[{0}]={1};\n'.format(resi, L)
    res += 'nmax[{0}]={1};\n'.format(resi, nmax)
    res += 'delta[{0}]={1};\n'.format(resi, delta)
    res += 'Wres[{0}]={1};\n'.format(resi, mathformat(Wlist))
    res += 'Nres[{0}]={1};\n'.format(resi, mathformat(Nlist))

    try:
        Outputs0 = mps.GetObservables(Outputs, 'unique_ID', '0')[0]
        maxsweeps = Outputs0['max_num_sweeps']
        warmupdim = Outputs0['warmup_bond_dimension']
        warmuptol = Outputs0['warmup_tol']
        bonddim = Outputs0['bond_dimension']
        maxdim = Outputs0['max_bond_dimension']

        res += 'maxsweeps[{0}]={1};\n'.format(resi, maxsweeps)
        res += 'warmupdim[{0}]={1};\n'.format(resi, warmupdim)
        res += 'warmuptol[{0}]={1};\n'.format(resi, warmuptol)
        res += 'bonddim[{0}]={1};\n'.format(resi, bonddim)
        res += 'maxdim[{0}]={1};\n'.format(resi, maxdim)

    except Exception as e:
        print e.message

    for i in range(0, id):
        iW, iN = position[i]
        Outputsi = mps.GetObservables(Outputs, 'unique_ID', str(i))

        for Output in Outputsi:
            spdm = Output['spdm']
            spdmeigs, U = np.linalg.eigh(spdm)
            maxeig = np.max(spdmeigs)
            eigsum = np.sum(spdmeigs)
            fc = maxeig / eigsum
            E0 = Output['energy']
            n = Output['n']
            n2 = Output['n2']
            converged = Output['converged']
            if converged:
                Jres[iW][iN] = Output['J']
                Ures[iW][iN] = Output['U']
                E0res[iW][iN] = Output['energy']#E0
                fcres[iW][iN] = fc
                nres[iW][iN] = Output['n']#n
                n2res[iW][iN] = Output['n2']#n2
                cres[iW][iN] = Output['spdm']#spdm
                nnres[iW][iN] = Output['nn']

    for iW in range(0, len(Wlist)):
        E0s = E0res[iW]
        for iN in range(1, len(Nlist)):
            mumres[iW][iN] = E0s[iN] - E0s[iN - 1]
        for iN in range(0, len(Nlist)-1):
            mupres[iW][iN] = E0s[iN + 1] - E0s[iN]


    res += 'Jres[{0}]={1};\n'.format(resi, mathformat(Jres))
    res += 'Ures[{0}]={1};\n'.format(resi, mathformat(Ures))
    res += 'E0res[{0}]={1};\n'.format(resi, mathformat(E0res))
    res += 'fcres[{0}]={1};\n'.format(resi, mathformat(fcres))
    res += 'nres[{0}]={1};\n'.format(resi, mathformat(nres))
    res += 'n2res[{0}]={1};\n'.format(resi, mathformat(n2res))
    res += 'cres[{0}]={1};\n'.format(resi, mathformat(cres))
    res += 'nnres[{0}]={1};\n'.format(resi, mathformat(nnres))
    res += 'mumres[{0}]={1};\n'.format(resi, mathformat(mumres))
    res += 'mupres[{0}]={1};\n'.format(resi, mathformat(mupres))
    res += 'runtime[{0}]=\"{1}\";\n'.format(resi, runtime)

    resf.write(res)
    resf.flush()
    os.fsync(resf.fileno())

quit()



