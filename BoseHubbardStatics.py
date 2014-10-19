import MPSPyLib as mps
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time

t0=time.time()

def plotIt(jvalues,muvalues,dependentvalues):
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')		
	plt.scatter(jvalues,muvalues,c=dependentvalues,cmap=cm.jet)
	plt.xlim((np.min(jvalues),np.max(jvalues)))
	plt.ylim((np.min(muvalues),np.max(muvalues)))
	plt.xlabel(r"\textbf{tunneling}  " r"$t/U$",fontsize=16)
	plt.ylabel(r"\textbf{chemical potential}  " r"$\mu/U$",fontsize=16)
	cbar = plt.colorbar()
	cbar.set_label(r"\textbf{Quantum Depletion}",fontsize=16)
	plt.show() 

PostProcess=True

#Build operators
Operators=mps.BuildBoseOperators(6)
Operators['interaction']=0.5*(np.dot(Operators['nbtotal'],Operators['nbtotal'])-Operators['nbtotal'])
#Define Hamiltonian MPO
H=mps.MPO()
H.AddMPOTerm(Operators,'bond',['bdagger','b'],hparam='t',weight=-1.0)
H.AddMPOTerm(Operators,'site','interaction',hparam='U',weight=1.0)

#ground state observables
myObservables=mps.Observables()
#Site terms
myObservables.AddObservable(Operators, 'nbtotal','site','n')
#correlation functions
myObservables.AddObservable(Operators,['nbtotal','nbtotal'],'corr','nn')
myObservables.AddObservable(Operators,['bdagger','b'],'corr','spdm')

myConv=mps.MPSConvergenceParameters(max_num_sweeps=7)

U=1.0
tlist=np.linspace(0,0.4,20)
parameters=[]
L=10
Nlist=np.linspace(1,11,11)

for N in Nlist:
    for t in tlist:
        parameters.append({ 
                  #Directories
                  'job_ID'                    :   'Bose_Hubbard_statics',
                  'unique_ID'                 : 't_'+str(t)+'N_'+str(N),
                  'Write_Directory'           : 'TMP/', 
                  'Output_Directory'          : 'OUTPUTS/', 
                  #System size and Hamiltonian parameters
                  'L'                         : L,
                  't'                         : t, 
                  'U'                         : U, 
                  #Specification of symmetries and good quantum numbers
                  'Abelian_generators'        : ['nbtotal'],
	          #Working At Unit Filling
                  'Abelian_quantum_numbers'   : [N],
                  #Convergence parameters
                  'verbose'                   : 1, 
                  'MPSObservables'            : myObservables,
                  'MPSConvergenceParameters'  : myConv
          	  })
    

#Write Fortran-readable main files
MainFiles=mps.WriteFiles(parameters,Operators,H,PostProcess=PostProcess)
#Run the simulations if we are not just Post
if not PostProcess:
    mps.runMPS(MainFiles,RunDir='./')

print time.time()-t0

#Postprocessing and plotting
if PostProcess:
    alldata=[np.array([]),np.array([]),np.array([])]
    Outputs=mps.ReadStaticObservables(parameters)
    for t in tlist:
        mulist=[]
        energylist=[]
        depletionlist=[]
        Outputs2=mps.GetObservables(Outputs,'t',t)
        tinternal=t*np.ones(len(Nlist))
		
        for Output in Outputs2:
            spdm=Output['spdm']
            spdmeigs, U = np.linalg.eigh(spdm)
            maxeig=np.max(spdmeigs)
            eigsum=np.sum(spdmeigs)
            depletion=1-(maxeig/eigsum)
            depletionlist.append(depletion)
            energy=Output['energy']
            energylist.append(energy)
            converged=Output['converged']
            print Output['t']

            print 'Simulation Converged?:',converged
        mulist.append(energylist[0])

        for i in range(1,len(energylist),1):
            mu=energylist[i]-energylist[i-1]
            mulist.append(mu)
        alldata[0]=np.append(alldata[0],tinternal)
        alldata[1]=np.append(alldata[1],mulist)
        alldata[2]=np.append(alldata[2],depletionlist)
		
    plotIt(alldata[0],alldata[1],alldata[2])



