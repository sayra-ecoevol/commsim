from scipy import *
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json
colors = sns.color_palette()
from matplotlib.backends import backend_pdf as bpd
#import relevant files from community simulator
from community_simulator import *
from community_simulator import Community,essentialtools,usertools,visualization,analysis
from community_simulator.usertools import *
from community_simulator.visualization import *


def validate_simulation_sa743(com_in,pro_time,iteration):
    """
    Check accuracy, convergence, and noninvadability of community instance com_in.
    N0 indicates which species were present at the beginning of the simulation.
    """
    com = com_in.copy()
    #failures = np.sum(np.isnan(com.N.iloc[0]))
    survive = com.N>0
    com.N[survive] = 1
    if type(com.params) is not list:
        params_list = [com.params for k in range(len(com.N.T))]
    else:
        params_list = com.params
    dlogNdt_survive = pd.DataFrame(np.asarray(list(map(com.dNdt,com.N.T.values,com.R.T.values,params_list))).T,
                                   index=com.N.index,columns=com.N.columns)
    dlogNdt_survive.to_csv(r"dNdt_gen_after"+str(pro_time)+ " for_gen = "+ str(iteration)+".dat", sep='\t')
    accuracy = np.max(abs(dlogNdt_survive))

    return accuracy.mean()

file = open(r"psoc_setup.txt", "r")
contents = file.read()
dictionary0 = eval(contents)
file.close()
#print((dictionary0))

#store the values of the setup
n_families = dictionary0['n_families'];
n_species = dictionary0['n_species'];
n_classes =dictionary0['n_classes'];
n_resources = dictionary0['n_resources'];
maintenance = dictionary0['maintenance'];
tau = dictionary0['tau'];
nspecies_sampled =dictionary0['nspecies_sampled'];
scale_set = dictionary0['scale_set'] #this is needed for conversion to actual cell counts from the coded abundances
nofwells = n_species+1
iter = dictionary0['iterations']
propagation_time = dictionary0['prop_times']

#prop1 = propagation_time[0]
#prop2 = propagation_time[1]
#prop3 = propagation_time[2]
#prop4 = propagation_time[3]
#prop5 = propagation_time[4]

#assumptions in the model
file = open(r"psoc_assump.txt", "r")
contents = file.read()
a_default = eval(contents)
file.close()
assumptions = a_default
#print(assumptions)

init_state = MakeInitialState(assumptions)
def dNdt(N,R,params):
    return MakeConsumerDynamics(assumptions)(N,R,params)
def dRdt(N,R,params):
    return MakeResourceDynamics(assumptions)(N,R,params)
dynamics = [dNdt,dRdt]



#print(init_state)
M = init_state[2]
#print(M)
Stot = init_state[4]
#print(Stot)


params = MakeParams(assumptions)

cons = params[2]['c']

table = MakeMatrices(assumptions)


commfunction = pd.read_csv(r"phi_global.csv", header=None)
phii = np.zeros((1,Stot))
#print(phii)

for j in range(Stot):
    phii[0,j] = commfunction.iat[j,0]

#print(phii)


plate1 = Community(init_state,dynamics,params,parallel=False) #on Windows, set parallel to False
#print(plate1.N)



c_param = np.array(cons)

for i in range(nofwells):
    plate1.params[i]['c'] = c_param

#print(cons)


plt.figure()
fig,ax=plt.subplots()
sns.heatmap(cons,vmin=0,square=True,linewidths=.5,xticklabels=False,yticklabels=False,cbar=True,ax=ax)
ax.set_title('consumer matrix')
plt.savefig(r"consumption_map"+ ".png")
plt.close()
#plt.show()

N0 = plate1.N
plate1.scale = scale_set
#c_param = np.array((n_species,n_resources))
#M = n_resources
#c_mean = assumptions['muc']/M
#c_var = assumptions['sigc']**2/M
#c_param = c_mean + np.random.randn(n_species,n_resources)*np.sqrt(c_var)
con_mat = pd.DataFrame(cons)
con_mat.to_csv(r"consumption_matrix.dat", sep='\t')
plate1.N.to_csv(r"N0.dat", sep='\t')

f = open(r"setup.dat", "w")
f.write("{\n")
for k in dictionary0.keys():
    f.write("'{}':'{}'\n".format(k, dictionary0[k]))
f.write("}")
f.close()

f = open(r"assumptions.dat", "w")
f.write("{\n")
for k in assumptions.keys():
    f.write("'{}':'{}'\n".format(k, assumptions[k]))
f.write("}")
f.close()


# In[14]:


supply_grad = assumptions['supply_grad']
commfunc = np.zeros((iter,1))
generation = np.zeros((iter,1))
failure = np.zeros((iter,1))


# In[ ]:


tol = 1e-2
start_time = time.time()
for k in range(iter):
    #copy the first well column to all other wells
    for i in range(1,nofwells):
        plate1.N['W'+str(i)]=plate1.N['W0']

    for i in range(0,nofwells-1):
        plate1.N.iat[i,i+1]= 1 - plate1.N.iat[i,0]

    N0 = plate1.N
    plate1.N.to_csv(r"N0_for_gen = "+ str(k)+".dat", sep='\t')
    #note that the plate1 in community class does not have an attribite N0
    for j in range(n_resources):
        for i in range(nofwells):
            plate1.R.iat[j,i]= (j+1)*supply_grad

    plate1.R.to_csv(r"R0.dat", sep='\t')
    plate1.scale=scale_set
    plate1.Propagate(propagation_time[0])
    plate1.N.to_csv(r"N_after_ "+str(propagation_time[0])+"s_for_gen = "+ str(k)+".dat", sep='\t')
    acc = validate_simulation_sa743(plate1,propagation_time[0],k)

    i=100
    while acc > tol:
        plate1.Propagate(propagation_time[i])
        plate1.N.to_csv(r"N_after_ "+str(propagation_time[i])+"s_for_gen = "+ str(k)+".dat", sep='\t')
        acc = validate_simulation_sa743(plate1,propagation_time[i],k)
        i=i+1

    for i in range(n_species):
        for j in range(nofwells):
            if plate1.N.iat[i,j] < tol:
                plate1.N.iat[i,j]= 0
            else:
                plate1.N.iat[i,j]=1

    functionarray = phii.dot(plate1.N)
    order = functionarray.argsort()

    comm = order[0,nofwells-1]
    commfunc[k] = functionarray[0,comm]

    N0upd = N0['W'+str(comm)]

    for j in range(n_species):
        plate1.N['W0'] = N0upd

    generation[k]=k+1

outp = pd.DataFrame(np.asarray(np.concatenate((generation,commfunc,failure), axis=1)))
outp.to_csv(r"commfunc_run.dat", sep='\t')

time_el = time.time() - start_time
with open(r"commfunc_run.dat", 'a') as file:
    file.write('time elapsed = ' + str(time_el) + 'seconds')


from matplotlib.ticker import MaxNLocator
plt.figure()
plt.plot(generation,commfunc,'o',color='red')
plt.xlabel('generation')
plt.ylabel('community function')
plt.grid(True, color = "black", linewidth = "1.2", linestyle = "--")
#plt.minorticks_on()
#ax.xaxis.set_minor_locator(MaxNLocator(integer=True))
plt.savefig(r"run"+ ".png")
#plt.show()


# In[ ]:





# In[ ]:
