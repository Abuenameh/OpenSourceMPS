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
from mathematica import mathformat
from matplotlib import cm
from speed import gprogress
from speed import progress

numthreads = 2

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


N = 1000;
g13 = 2.5e9;
g24 = 2.5e9;
Delta = -2.0e10;
alpha = 1.1e7;

Ng = np.sqrt(N) * g13;

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

# t0 = time.time()
#
#
# def plotIt(jvalues, muvalues, dependentvalues):
#     plt.rc('text', usetex=True)
#     plt.rc('font', family='serif')
#     plt.scatter(jvalues, muvalues, c=dependentvalues, cmap=cm.jet)
#     plt.xlim((np.min(jvalues), np.max(jvalues)))
#     plt.ylim((np.min(muvalues), np.max(muvalues)))
#     plt.xlabel(r"\textbf{tunneling}  " r"$t/U$", fontsize=16)
#     plt.ylabel(r"\textbf{chemical potential}  " r"$\mu/U$", fontsize=16)
#     cbar = plt.colorbar()
#     cbar.set_label(r"\textbf{Quantum Depletion}", fontsize=16)
#     plt.show()


PostProcessOnly = int(sys.argv[1])

if not PostProcessOnly:
    import gtk
    import gobject

# Build operators
nmax = 6
Operators = mps.BuildBoseOperators(nmax)
Operators['n2'] = np.dot(Operators['nbtotal'], Operators['nbtotal'])
Operators['interaction'] = 0.5 * (np.dot(Operators['nbtotal'], Operators['nbtotal']) - Operators['nbtotal'])
#Define Hamiltonian MPO
H = mps.MPO()
H.AddMPOTerm(Operators, 'bond', ['bdagger', 'b'], hparam='J', weight=-1.0)
H.AddMPOTerm(Operators, 'site', 'interaction', hparam='U', weight=1.0)

#ground state observables
myObservables = mps.Observables()
#Site terms
myObservables.AddObservable(Operators, 'nbtotal', 'site', 'n')
myObservables.AddObservable(Operators, 'n2', 'site', 'n2')
#correlation functions
myObservables.AddObservable(Operators,['nbtotal','nbtotal'],'corr','nn')
myObservables.AddObservable(Operators, ['bdagger', 'b'], 'corr', 'spdm')

maxsweeps = 7
myConv = mps.MPSConvergenceParameters(max_num_sweeps=maxsweeps)

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

# U = 1.0
# tlist = [[t+0.01,t-0.02,t-0.03,t+0.04,t+0.03,t-0.01] for t in np.linspace(0,0.4,20)]
# # tlist = [[t, t, t, t, t, t] for t in np.linspace(0,0.4,20)]
# # tlist = [t for t in np.linspace(0,0.4,20)]
# tlist = [0,0.01]
# # parameters = []
# L = 6
# Nlist = np.linspace(1, 2*L, 2*L)
# Nlist = [1,2,3]
#
# # count = 0
#
# for N in Nlist:
#     for t in tlist:
#         parameters.append({
#             #Directories
#             'job_ID': 'Bose_Hubbard',
#             # 'unique_ID': 't_' + str(t[0]) + 'N_' + str(N),
#             'unique_ID': 't_' + str(t) + 'N_' + str(N),
#             'Write_Directory': 'Temp/',
#             'Output_Directory': 'Output/',
#             #System size and Hamiltonian parameters
#             'L': L,
#             't': t,#[0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01],#t,
#             'U': U,#[1,1,1,1,1,1],#U,
#             #Specification of symmetries and good quantum numbers
#             'Abelian_generators': ['nbtotal'],  #Working At Unit Filling
#             'Abelian_quantum_numbers': [N],
#             #Convergence parameters
#             'verbose': 0,
#             'MPSObservables': myObservables,
#             'MPSConvergenceParameters': myConv
#         })
#         # count += 1
#
#
# #Write Fortran-readable main files=b
# MainFiles=mps.WriteFiles(parameters,Operators,H,PostProcess=PostProcess)
# #Run the simulations if we are not just Post
# if not PostProcess:
#     mps.runMPS(MainFiles,RunDir='./')


# print time.time() - t0
#
# #Postprocessing and plotting
# if PostProcess:
#     alldata = [np.array([]), np.array([]), np.array([])]
#     Outputs = mps.ReadStaticObservables(parameters)
#     for t in tlist:
#         mulist = []
#         energylist = []
#         depletionlist = []
#         Outputs2 = mps.GetObservables(Outputs, 't', [t])
#         # Outputs2 = mps.GetObservables(Outputs, 't', t[0])
#         tinternal = t * np.ones(len(Nlist))
#         # tinternal = t[0] * np.ones(len(Nlist))
#
#         for Output in Outputs2:
#             spdm = Output['spdm']
#             spdmeigs, U = np.linalg.eigh(spdm)
#             maxeig = np.max(spdmeigs)
#             eigsum = np.sum(spdmeigs)
#             depletion = 1 - (maxeig / eigsum)
#             depletionlist.append(depletion)
#             energy = Output['energy']
#             energylist.append(energy)
#             converged = Output['converged']
#             print Output['t']
#
#             print 'Simulation Converged?:', converged
#         mulist.append(energylist[0])
#
#         for i in range(1, len(energylist), 1):
#             mu = energylist[i] - energylist[i - 1]
#             mulist.append(mu)
#         alldata[0] = np.append(alldata[0], tinternal)
#         alldata[1] = np.append(alldata[1], mulist)
#         alldata[2] = np.append(alldata[2], depletionlist)
#
#     plotIt(alldata[0], alldata[1], alldata[2])
#


