from setup_3 import *
from constraints_3 import *
from dss_3 import *
import datetime
import time
import cvxpy as cp
from Ycalcs import calc_Ybus
from itertools import combinations
import gc


def feeder_init(modelpath, loadfolder, loadpath, timesteps, timestepcur, subkVbase_phg, subkVAbase, PVforecast, Psat_nodes=[],Qsat_nodes=[]):
    modeldata = pd.ExcelFile(modelpath)
    actpath = loadpath
    
    # set dummy values for undefined variables
    date = datetime.datetime.now()
    month = date.month
    day = date.day
    hour = date.hour
    minute = date.minute

    refphasor = np.ones((3,2))
    refphasor[:,0]=1
    refphasor[:,1]=[0,4*np.pi/3,2*np.pi/3]

    #get feeder
    feeder_init = feeder(modelpath,loadfolder,loadpath,actpath,timesteps,timestepcur,
                         subkVbase_phg,subkVAbase,refphasor,Psat_nodes,Qsat_nodes,PVforecast)
    feeder_init
    
    phase_size = 0
    for key, inode in feeder_init.busdict.items():
        if inode.type == 'SLACK' or inode.type == 'Slack' or inode.type == 'slack':
            for ph in inode.phases:
                phase_size += 1
    
    return phase_size, feeder_init

def lin_optimization(myfeeder, enable_actuators, verbose=True,eps_rel=1e-3,eps_abs=1e-10, max_iter = 50000): #write 'none' if doesnt exist    
    # set tuning parameters of objectives: 0 = off
    # lam1 - phasor target, lam2 - phase balancing, lam3 - voltage volatility
    # lamcost - const function, lamdistpf - transmission/distribution power flow
    lam1 = 0
    lam2 = 0
    lam3 = 0
    lam4 = 0
    
    lamcost = 1
    lamdistpf = 50
    
    # turn additional objectives on/off
    costfn_on_off = 0 # 0=off, 1=on  
    distpf_on_off = 0 # 0=off, 1=on
    
    #  phasor target settings:
    if lam1 > 0:
        target_key = '8'
        Vmag_match = [1.05]*3
        Vang_match = [0 - np.radians(1), 4/3*np.pi - np.radians(1), 2/3*np.pi - np.radians(1)] 
    
    # phase balancing settings:
    # no settings to change for this objective
    
    # voltage volatility settings:
    if lam3 > 0:
    # choose locations where you wish to control volatility:
        # either 'all', 'none', or a list of target nodes (i.e. ['680','692'])
        target_loc = []
    
    # cost function settings:
    if costfn_on_off == 1:
        actcostdata = pd.read_excel(actcostpath, index_col=0)
        
    # power flow target settings:
    if distpf_on_off == 1:      
        '# ~~~ INPUT NODE ~~~ #'  
        nodeloc = 77 # [INPUT HERE]
        '______________________'
        
        dfdtarg = pd.read_csv(genpath, index_col = 0)
        dfdtarg.index += 1
        Pstr, Qstr, Pkey, Qkey = f'{nodeloc}_P', f'{nodeloc}_Q', [], []
        for head in dfdtarg.columns:
            if Pstr in head:
                Pkey.append(head)
            if Qstr in head:
                Qkey.append(head)
        netpflow_targ = np.transpose(dfdtarg[Pkey].values/subkVAbase)
        netqflow_targ = np.transpose(dfdtarg[Qkey].values/subkVAbase)
        print(netpflow_targ[:][:][0:3]*subkVAbase)
        print(netqflow_targ[:][:][0:3]*subkVAbase)
        
    else:
        lamdistpf = 0
    
    
    # Run optimization problem and generate targets
    
    
    obj = 0
    #pdb.set_trace()
    Vangnom = [0, 4/3*np.pi, 2/3*np.pi] 
    
    for ts in range(0,myfeeder.timesteps):
        for key, bus in myfeeder.busdict.items():
        #for key, bus in myfeeder.busdict['633']:
            #print(bus.name)
            
    # objective 1 - phasor target  
            if lam1 > 0:
                if key == target_key:
                    
                    #pdb.set_trace()
                    #normalize to 2pi rad by adjusting to 2Pi / 2pi
                    if (bus.phasevec == np.ones((3,timesteps))).all():
                        obj = obj + lam1*((cp.square(bus.Vang_linopt[0,ts]-Vang_match[0]) + cp.square(bus.Vang_linopt[1,ts]-Vang_match[1]) + cp.square(bus.Vang_linopt[2,ts]-Vang_match[2])))
                        obj += lam1*((cp.square(bus.Vmagsq_linopt[0,ts]-Vmag_match[0]**2) + cp.square(bus.Vmagsq_linopt[1,ts]-Vmag_match[1]**2) + cp.square(bus.Vmagsq_linopt[2,ts]-Vmag_match[2]**2)))


    # objective 2 - phase balancing
            if lam2 > 0:
                if (bus.phasevec == np.array([[1],[1],[0]])).all():
                    obj = obj + lam2*cp.square(bus.Vmagsq_linopt[0,ts]-bus.Vmagsq_linopt[1,ts])
                    obj = obj + lam2*cp.square((bus.Vang_linopt[0,ts]-bus.Vang_linopt[1,ts])-(Vangnom[0]-Vangnom[1]))
                if (bus.phasevec == np.array([[1],[0],[1]])).all():
                    obj = obj + lam2*cp.square(bus.Vmagsq_linopt[0,ts]-bus.Vmagsq_linopt[2,ts])
                    obj = obj + lam2*cp.square((bus.Vang_linopt[0,ts]-bus.Vang_linopt[2,ts])-(Vangnom[0]-Vangnom[2]))
                if (bus.phasevec == np.array([[0],[1],[1]])).all():
                    obj = obj + lam2*cp.square(bus.Vmagsq_linopt[1,ts]-bus.Vmagsq_linopt[2,ts])
                    obj = obj + lam2*cp.square((bus.Vang_linopt[1,ts]-bus.Vang_linopt[2,ts])-(Vangnom[1]-Vangnom[2]))
                if (bus.phasevec == np.ones((3,timesteps))).all():
                    obj = obj + lam2*(cp.square(bus.Vmagsq_linopt[0,ts]-bus.Vmagsq_linopt[1,ts]) +
                                      cp.square(bus.Vmagsq_linopt[0,ts]-bus.Vmagsq_linopt[2,ts]) +
                                      cp.square(bus.Vmagsq_linopt[1,ts]-bus.Vmagsq_linopt[2,ts]))
                    obj = obj + lam2*(cp.square((bus.Vang_linopt[0,ts]-bus.Vang_linopt[1,ts])-(Vangnom[0]-Vangnom[1])) +
                                      cp.square((bus.Vang_linopt[0,ts]-bus.Vang_linopt[2,ts])-(Vangnom[0]-Vangnom[2])) +
                                      cp.square((bus.Vang_linopt[1,ts]-bus.Vang_linopt[2,ts])-(Vangnom[1]-Vangnom[2])))
                    
    # objective 4 - target all nominal - currently set up only for 3phase networks
            if lam4 > 0:
    # =============================================================================
    #                 if (bus.phasevec == np.array([[1],[1],[0]])).all():
    #                     obj = obj + lam2*(cp.square(1-bus.Vmagsq_linopt[0,ts]) + 
    #                                       cp.square(1-bus.Vmagsq_linopt[1,ts]))
    #                     obj = obj + lam2*cp.square((bus.Vang_linopt[0,ts]-bus.Vang_linopt[1,ts])-(Vangnom[0]-Vangnom[1]))
    #                 if (bus.phasevec == np.array([[1],[0],[1]])).all():
    #                     obj = obj + lam2*(cp.square(1-bus.Vmagsq_linopt[0,ts]) + 
    #                                       cp.square(1-bus.Vmagsq_linopt[2,ts]))
    #                     obj = obj + lam2*cp.square((bus.Vang_linopt[0,ts]-bus.Vang_linopt[2,ts])-(Vangnom[0]-Vangnom[2]))
    #                 if (bus.phasevec == np.array([[0],[1],[1]])).all():
    #                     obj = obj + lam2*(cp.square(1-bus.Vmagsq_linopt[1,ts]) + 
    #                                       cp.square(1-bus.Vmagsq_linopt[2,ts]))
    #                     obj = obj + lam2*cp.square((bus.Vang_linopt[1,ts]-bus.Vang_linopt[2,ts])-(Vangnom[1]-Vangnom[2]))
    # =============================================================================
                if (bus.phasevec == np.ones((3,timesteps))).all():
                    obj += lam4*(cp.square(1-bus.Vmagsq_linopt[0,ts]) + 
                                 cp.square(1-bus.Vmagsq_linopt[1,ts]) +
                                 cp.square(1-bus.Vmagsq_linopt[2,ts]))
                    obj += lam4*(cp.square(Vangnom[0]-bus.Vang_linopt[0,ts]) +
                                 cp.square(Vangnom[1]-bus.Vang_linopt[1,ts]) +
                                 cp.square(Vangnom[2]-bus.Vang_linopt[2,ts]))

    # objective 3 - voltage volitility
        if lam3 > 0:
            for ts in range(1,myfeeder.timesteps):
                for key, bus in myfeeder.busdict.items():
                    if isinstance(target_loc,list):
                        for targkey in target_loc:
                            if bus.name == 'bus'+targkey:
                                obj += lam3*cp.square(bus.Vmagsq_linopt[0,ts]-bus.Vmagsq_linopt[0,ts-1]+
                                                      bus.Vmagsq_linopt[1,ts]-bus.Vmagsq_linopt[1,ts-1]+
                                                      bus.Vmagsq_linopt[2,ts]-bus.Vmagsq_linopt[2,ts-1])
                    else:
                        obj += lam3*cp.square(bus.Vmagsq_linopt[0,ts]-bus.Vmagsq_linopt[0,ts-1]+
                                              bus.Vmagsq_linopt[1,ts]-bus.Vmagsq_linopt[1,ts-1]+
                                              bus.Vmagsq_linopt[2,ts]-bus.Vmagsq_linopt[2,ts-1])
                            
    # voltage volatility doesn't really make sense for only 2 timesteps? compute over horizon? 
    # OR (preferred) find way to store value from previous iteration?
        # is this equivalent even though not all cvx vars anymore?
        # initialize by minimizing volatility over first 10 timesteps...
    # alternative method:
        # consider previous timestep as well as next X timesteps
        # weigh previous timestep more heavily than future timesteps
    
    # TODO: vmagprev
    #Vmag_prev = {}
    #Vmag_prev[key] = np.ones((3,myfeeder.timesteps))
    # objective 3.2 [HIL} - voltage volatility
    '''
    for ts in range(0,myfeeder.timesteps):
        for key, bus in myfeeder.busdict.items():
            if key in perf_nodes:
                if bus.name == 'bus_' + key:
                    obj += lam3*cp.square(bus.Vmagsq_linopt[0,ts]-Vmag_prev[key][0]+
                                          bus.Vmagsq_linopt[1,ts]-Vmag_prev[key][1]+
                                          bus.Vmagsq_linopt[2,ts]-Vmag_prev[key][2])
    '''
                
    # add cost function to actuators       
    if costfn_on_off == 1:
        for ts in range(0,myfeeder.timesteps):  
                    
            for key, inode in myfeeder.busdict.items():
                for iact in inode.actuators:
                    key_str = str(key)
                    for idx in range(0,3):
                        obj += cp.square(iact.Pgen[idx,ts:ts+1] * actcostdata[key_str][ts])
                        
    # distribution level power flow objective to meet transmission SPBC target
        # in this iteration, it is done as meeting power flows directly defined by the transmission level SPBC
        # but in later iterations the goal is to have an intermediary feedback controller that translates the voltage target
        # into a power flow
        
    if distpf_on_off == 1:
                
        '# ~~~ SLACK LINE FLOW ~~~ #'
        for key, inode in myfeeder.busdict.items():
            if inode.type == 'SLACK' or inode.type == 'Slack' or inode.type == 'slack':
                conn = inode.edges_out
                break
        for ts in range(0,myfeeder.timesteps):
             for idx in range(0,3):
                 obj += lamdistpf*cp.square(conn[0].P_linopt[idx,ts:ts+1] - netpflow_targ[idx,ts:ts+1])
                 obj += lamdistpf*cp.square(conn[0].Q_linopt[idx,ts:ts+1] - netqflow_targ[idx,ts:ts+1])


    '~~~~~~~~~~~~~~~~'
    'CVX OPT SETTINGS'
    '~~~~~~~~~~~~~~~~'

    objective = cp.Minimize(obj)
    constraints = cvx_set_constraints(myfeeder,enable_actuators) # Second argument turns actuators on/off
    prob = cp.Problem(objective, constraints)
    #result = prob.solve()  
    result = prob.solve(solver="OSQP", verbose=verbose, eps_rel=eps_rel, eps_abs=eps_abs, max_iter=max_iter)

    return prob, result


def initial_power_injections(feeder):

    num_buses = len(feeder.busdict)
    Pinj = np.zeros((num_buses-1)*3)
    Qinj = np.zeros((num_buses-1)*3)
    for key, bus in feeder.busdict.items():
        if str(key) != "1":
            bus_P_inj = np.zeros(3)
            bus_Q_inj = np.zeros(3)
            for load in bus.loads:
                bus_P_inj -= load.Psched.T[0]/feeder.subkVAbase
                bus_Q_inj -= load.Qsched.T[0]/feeder.subkVAbase

            # For some reason this works on A0 but not A3
            # for act in bus.actuators:
            #     # check if these are in p.u.
            #     # print out Pgen to debug
            #     bus_P_inj += act.Pgen.value.T[0]
            #     bus_Q_inj += act.Qgen.value.T[0]

            for cap in bus.cap:
                # caps have a negative sign
                bus_P_inj += -cap.Pvec.T[0]/feeder.subkVAbase
                bus_Q_inj += -cap.Qvec.T[0]/feeder.subkVAbase

            for i in range(3):
                Pinj[(int(key)-2)*3 + i] = bus_P_inj[i]
                Qinj[(int(key)-2)*3 + i] = bus_Q_inj[i]
            
    return Pinj, Qinj


def sanity_check(base_feeder, enable_actuators, opt_vars={"verbose":True,"eps_rel":1e-3,"eps_abs":1e-10, "max_iter":50000}):
    
    prob, result = lin_optimization(base_feeder, enable_actuators,
                                                 verbose=opt_vars["verbose"],
                                                 eps_rel=opt_vars["eps_rel"],
                                                 eps_abs=opt_vars["eps_abs"],
                                                 max_iter=opt_vars["max_iter"])
    if prob.status != "optimal":
        print("CVX cant even solve base feeder. there is a problem")

    # DETERMINING THE INITIAL POWER INJECTIONS
    Pinj_pu, Qinj_pu = initial_power_injections(base_feeder)

    
    # print("Pinj", Pinj_pu)
    # print()
    # print("Qinj", Qinj_pu)
    # print()

    # for key, bus in base_feeder.busdict.items():
    #     print("Bus",key,": Voltage", bus.Vmagsq_linopt.value.T[0])
    #     # print("CVX Bus",key,": Voltage angle", bus.Vang_linopt.value.T[0])
    for key, line in base_feeder.linedict.items():
        print("CVX Line",key,": Flow:", line.P_linopt.value.T[0])

    # run linearized power flow
    (Ypu, Y, busidx) = calc_Ybus(base_feeder)
    # Pinj_keith, Qinj_keith = DSS_netInj(base_feeder, busidx, loud=0)
    # print("Pinj_k", Pinj_keith)
    # print()
    # print("Qinj_k", Qinj_keith)
    # print()

     
    nnodes = int(len(Ypu))
    nbus = int(len(Ypu)/3)
    Yr = Ypu[3:,3:] #calc_Ybus puts the slack bus in the first entry of Y
    Psir = np.concatenate((np.concatenate((np.real(Yr),-np.imag(Yr)),axis=1),np.concatenate((-np.imag(Yr),-np.real(Yr)),axis=1)),axis=0)
    try:
        invPsir = np.linalg.inv(Psir)
    except np.linalg.LinAlgError:
        print("Couldn't invert bc singular matrix. Trying pseudoinverse")
        invPsir = np.linalg.pinv(Psir)


    # This is linearized power flow
    dV = np.matmul(invPsir,np.hstack((Pinj_pu, Qinj_pu))) #gives dV in per unit

    Vmagest = np.ones(nnodes) + np.hstack((np.zeros(3), dV[:(nnodes-3)]))
    Vangest = np.hstack((np.zeros(3), dV[(nnodes-3):])) #this angle elstimate will be in radians not degrees
    # print("Vmagest", Vmagest)
    print("---------------linearized power flow---------------------")
    ############################ Calc Line Flows ################################

    #matrices with line flows in the upper triangle
    #this isnt the most elegant way to do this for 3 phase, a 3D tensor would probably be better. But this does the trick..
    #remember that all of these are approximate values
    #and they are all in pu
    ComplexLineCurrentMat = np.zeros((nnodes,nnodes),dtype=complex) #complex line currents
    ComplexLinePowerMat = np.zeros((nnodes,nnodes),dtype=complex) #complex line power flows
    ComplexLineLossMat = np.zeros((nnodes,nnodes),dtype=complex) #complex line power losses
    MagLineCurrentMat = np.zeros((nnodes,nnodes)) #line current magnitudes
    MagLinePowerMat = np.zeros((nnodes,nnodes)) #line power flow magnitudes
    MagLineLossMat = np.zeros((nnodes,nnodes)) #line power loss magnitudes
    for i in np.arange(nbus):
        for k in np.arange(nbus):
            if k > i:
                Vdiff = Vmagest[i*3:(i+1)*3]*np.exp(1j*Vangest[i*3:(i+1)*3]) - Vmagest[k*3:(k+1)*3]*np.exp(1j*Vangest[k*3:(k+1)*3])
                I = np.matmul(-Ypu[i*3:(i+1)*3,k*3:(k+1)*3],Vdiff)
                ComplexLineCurrentMat[i*3:(i+1)*3,k*3:(k+1)*3] = np.diag(I)
                ComplexLinePowerMat[i*3:(i+1)*3,k*3:(k+1)*3] = np.diag(Vmagest[i*3:(i+1)*3]*np.exp(1j*Vangest[i*3:(i+1)*3])*np.conj(I))
                ComplexLineLossMat[i*3:(i+1)*3,k*3:(k+1)*3] = np.diag(Vdiff*np.conj(I))
                MagLineCurrentMat[i*3:(i+1)*3,k*3:(k+1)*3] = np.diag(np.abs(I))
                MagLinePowerMat[i*3:(i+1)*3,k*3:(k+1)*3] = np.diag(np.abs(Vmagest[i*3:(i+1)*3]*np.exp(1j*Vangest[i*3:(i+1)*3])*np.conj(I)))
                if np.any(np.abs(Vmagest[i*3:(i+1)*3]*np.exp(1j*Vangest[i*3:(i+1)*3])*np.conj(I))):
                    print("LPF Flow from bus", i+1, "to", k+1, ":", np.abs(Vmagest[k*3:(k+1)*3]*np.exp(1j*Vangest[k*3:(k+1)*3])*np.conj(I)))
                    # print("LPF Current from bus", i+1, "to", k+1, ":", I)

                MagLineLossMat[i*3:(i+1)*3,k*3:(k+1)*3] = np.diag(np.abs(Vdiff*np.conj(I)))


    # check line flow capacities
    # print("Maximum Line Flow", np.max(MagLinePowerMat))

    # DSS
    run_dss(base_feeder)

    return


def get_k_component_list(feeder, k=2):
    component_id_list = []
    for key, line in feeder.linedict.items():
        component_id_list.append("line" + key)
    for key, shunt in feeder.shuntdict.items():
        component_id_list.append("shunt" + key)
    for key, switch in feeder.switchdict.items():
        component_id_list.append("switch" + key)
    # DON'T ADD TRANSFORMERS BECUASE THERE WILL BE AN ASSOCIATED LINE AND REMOVING THE LINE WILL ALSO REMOVE THE TRANSFORMER
    # for key, trans in feeder.transdict.items():
    #     component_id_list.append("trans" + key)

    component_combinations  = combinations(component_id_list, k)
    return list(component_combinations)



def nk_contingency_finder(base_feeder, k=2, enable_actuators=False, opt_vars={"verbose":False,"eps_rel":1e-3,"eps_abs":1e-10, "max_iter":50000}, use_recursive_opt=True):


    infractions = {}
    prob, result = lin_optimization(base_feeder, enable_actuators=False,
                                                 verbose=opt_vars["verbose"],
                                                 eps_rel=opt_vars["eps_rel"],
                                                 eps_abs=opt_vars["eps_abs"],
                                                 max_iter=opt_vars["max_iter"])
    if prob.status != "optimal":
        print("CVX cant even solve base feeder. there is a problem")

    # DETERMINING THE INITIAL POWER INJECTIONS
    Pinj_pu, Qinj_pu = initial_power_injections(base_feeder)

    component_combinations = get_k_component_list(base_feeder, k)
    print("Total number of combinations:", len(component_combinations))

    
    # Recursive calling for optimization
    if k > 1 and use_recursive_opt:
        print("Recursively calling n-" + str(k-1) + " contingency finder")
        prev_infractions = nk_contingency_finder(base_feeder, k-1, enable_actuators, opt_vars)
        infractions.update(prev_infractions)

        # remove components where there is already an infraction by removing (at least) one less component
        combinations_copy = component_combinations.copy()
        for current_comb in combinations_copy:
            matched_prev = False
            for prev_comb in prev_infractions:
                temp = True
                for el in prev_comb:
                    if el not in current_comb:
                        temp = False
                if temp:
                    matched_prev = True
                    break
            if matched_prev:
                component_combinations.remove(current_comb)

    
    for comb in component_combinations:
        mod_feeder = base_feeder
        print("Removing components:", comb)
        for comp in range(k):
            mod_feeder = remove_component(mod_feeder, comb[comp])

        # run linearized power flow
        (Ypu, Y, busidx) = calc_Ybus(mod_feeder)
        nnodes = int(len(Ypu))
        nbus = int(len(Ypu)/3)
        Yr = Ypu[3:,3:] #calc_Ybus puts the slack bus in the first entry of Y
        Psir = np.concatenate((np.concatenate((np.real(Yr),-np.imag(Yr)),axis=1),np.concatenate((-np.imag(Yr),-np.real(Yr)),axis=1)),axis=0)
        try:
            invPsir = np.linalg.inv(Psir)
        except np.linalg.LinAlgError:
            print("Couldn't invert bc singular matrix. Trying pseudoinverse")
            invPsir = np.linalg.pinv(Psir)

        dV = np.matmul(invPsir,np.hstack((Pinj_pu, Qinj_pu))) #gives dV in per unit
        Vmagest = np.ones(nnodes) + np.hstack((np.zeros(3), dV[:(nnodes-3)]))
        Vangest = np.hstack((np.zeros(3), dV[(nnodes-3):])) #this angle elstimate will be in radians not degrees
        ############################ Calc Line Flows ################################

        #matrices with line flows in the upper triangle
        #this isnt the most elegant way to do this for 3 phase, a 3D tensor would probably be better. But this does the trick..
        #remember that all of these are approximate values
        #and they are all in pu
        ComplexLineCurrentMat = np.zeros((nnodes,nnodes),dtype=complex) #complex line currents
        ComplexLinePowerMat = np.zeros((nnodes,nnodes),dtype=complex) #complex line power flows
        ComplexLineLossMat = np.zeros((nnodes,nnodes),dtype=complex) #complex line power losses
        MagLineCurrentMat = np.zeros((nnodes,nnodes)) #line current magnitudes
        MagLinePowerMat = np.zeros((nnodes,nnodes)) #line power flow magnitudes
        MagLineLossMat = np.zeros((nnodes,nnodes)) #line power loss magnitudes
        for i in np.arange(nbus):
            for j in np.arange(nbus):
                if j > i:
                    Vdiff = Vmagest[i*3:(i+1)*3]*np.exp(1j*Vangest[i*3:(i+1)*3]) - Vmagest[j*3:(j+1)*3]*np.exp(1j*Vangest[j*3:(j+1)*3])
                    I = np.matmul(-Ypu[i*3:(i+1)*3,j*3:(j+1)*3],Vdiff)
                    ComplexLineCurrentMat[i*3:(i+1)*3,j*3:(j+1)*3] = np.diag(I)
                    ComplexLinePowerMat[i*3:(i+1)*3,j*3:(j+1)*3] = np.diag(Vmagest[i*3:(i+1)*3]*np.exp(1j*Vangest[i*3:(i+1)*3])*np.conj(I))
                    ComplexLineLossMat[i*3:(i+1)*3,j*3:(j+1)*3] = np.diag(Vdiff*np.conj(I))
                    MagLineCurrentMat[i*3:(i+1)*3,j*3:(j+1)*3] = np.diag(np.abs(I))
                    MagLinePowerMat[i*3:(i+1)*3,j*3:(j+1)*3] = np.diag(np.abs(Vmagest[i*3:(i+1)*3]*np.exp(1j*Vangest[i*3:(i+1)*3])*np.conj(I)))
                    MagLineLossMat[i*3:(i+1)*3,j*3:(j+1)*3] = np.diag(np.abs(Vdiff*np.conj(I)))


        # check nodal voltage constraints
        voltage_violation = False
        Vmax = 1.05
        Vmin = .95
        for n in range(nnodes):
            if Vmagest[n] > Vmax or Vmagest[n] < Vmin:
                # print("Voltage constraint violated at node", n, "with magnitude", Vmagest[n])
                voltage_violation = True

        powerflow_violation = False
        # check line flow capacities
        for key, line in mod_feeder.linedict.items():
            i = np.minimum(int(busidx.index(line.from_node.buskey)), int(busidx.index(line.to_node.buskey)))
            j = np.maximum(int(busidx.index(line.from_node.buskey)), int(busidx.index(line.to_node.buskey)))
            line_rating_pu = line.MVArating_3ph / 3 * 1000 / mod_feeder.subkVAbase
            if line_rating_pu < np.abs(MagLinePowerMat[i*3][j*3]):
                # print("Line capacity exceeded between buses", i+1,"and", j+1, "with flow", MagLinePowerMat[i*3][j*3])
                powerflow_violation = True
            if line_rating_pu < np.abs(MagLinePowerMat[i*3+1][j*3+1]):
                # print("Line capacity exceeded between buses", i+1,"and", j+1, "with flow", MagLinePowerMat[i*3+1][j*3+1])
                powerflow_violation = True
            if line_rating_pu < np.abs(MagLinePowerMat[i*3+2][j*3+2]):
                # print("Line capacity exceeded between buses", i+1,"and", j+1, "with flow", MagLinePowerMat[i*3+2][j*3+2])
                powerflow_violation = True

        if voltage_violation and powerflow_violation:
            infractions[comb] = "both violations"
        elif voltage_violation:
            infractions[comb] = "voltage violation"
        elif powerflow_violation:
            infractions[comb] = "powerflow violation"

    print("Done with n-" + str(k) + " contingency analysis")
    print("-------------------------------------------------------------------------------------------------------------")

    return infractions


# 

def run_dss(myfeeder):
    # UNFINISHED
    DSS_alltimesteps(myfeeder,0) # Second argument turns voltage alarms on/off

    # for key, bus in base_feeder.busdict.items():
    #     print("Bus",key,": Voltage", bus.Vmagsq_linopt.value.T[0])
    #     # print("CVX Bus",key,": Voltage angle", bus.Vang_linopt.value.T[0])
    print("---------------------------dss----------------------------")
    for key, line in myfeeder.linedict.items():
        Imag = line.Imag_NL.T[0]
        Iang = line.Iang_NL.T[0]
        from_bus = line.from_node
        Vmag = from_bus.Vmag_NL.T[0]
        Vang = from_bus.Vang_NL.T[0]
        flow = np.abs(Vmag*np.exp(1j*Vang)*np.conj(Imag*np.exp(1j*Iang))) / myfeeder.subkVAbase / 1000
        print("DSS Line",key,": Flow:", flow)

    
    # 'turn export targets on/off here'
    # export_Vtargets(myfeeder)
    # ##[jasper] - fn to get target in vector format for LPBC
    # #nodes_arr,Vmag_targ,Vang_targ,lVbase = get_targets(myfeeder)
    
    # # Vtargdict[key(nodeID)][Vmag/Vang/KVbase]
    # Vtargdict, act_keys = get_targets(myfeeder)
    # #lpbc_keys = [lpbc_node1,lpbc_node2,lpbc_node3,lpbc_node4]
    return

if __name__ == "__main__":

    filepath = "IEEE14_T/1_model/mod_pu3/"
    modelpath = filepath + "IEEE14_T_model_pu3_v1_A0.xlsx"

    loadfolder = "IEEE14_T/2_load/variation_raw_01/"
    loadpath = loadfolder + "IEEE14_ld_raw_gen_v0.xlsx"

    # filepath = "single_line/"
    # modelpath = filepath + "singleLineNetworkCrossZ_reform.xlsx"

    # loadfolder = "single_line/"
    # loadpath = loadfolder + "singleLoad_reform.xlsx"


    'IEEE118 & IEE14'
    subkVbase_phg = 138/np.sqrt(3)
    subkVAbase = 100000.
    timesteps = 2

    # Specify initial timestep
    date = datetime.datetime.now()
    month = date.month
    day = date.day
    hour = date.hour
    minute = date.minute
    #timestepcur = hour*60+minute

    timestepcur = 0 #8*60 # [INPUT HERE] Manual input of start time

    # Input constnats for PV forecasting
    PV_on = False # True for ON
    PVnodes = []

    PVforecast = {}
    #PVforecast['on_off'] = PV_on
    for node in PVnodes: # this sets all nodes the same, would have to manually change fields to have different inputs for different nodes
        PVforecast[node] = {}
        PVforecast[node]['on_off'] = PV_on
        PVforecast[node]['lat'] = 37.87
        PVforecast[node]['lon'] = -122
        PVforecast[node]['maridian'] = -120
        PVforecast[node]['PVfrac'] = 0.3

    phase_size, myfeeder = feeder_init(modelpath, loadfolder, loadpath, timesteps, timestepcur, subkVbase_phg, subkVAbase, PVforecast)
    # sanity_check(myfeeder, False)
    infraction_dict = nk_contingency_finder(myfeeder, k=3)
    print("Removing these components from the base feeder should result in either a voltage or flow violation")
    print("Note that removing a line between two nodes will also remove a transformer between those two nodes if present")
    for comb in infraction_dict:
        print(comb, infraction_dict[comb])
    # prob, result = lin_optimization(myfeeder)
    # status = prob.status

    #Vtargdict, act_keys, subkVAbase, myfeeder = spbc_run(myfeeder, timestepcur)
    #gendict = get_gen(myfeeder)
    #flowdict = get_flow(myfeeder)