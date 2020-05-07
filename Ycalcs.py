

import numpy as np
import sys
import pprint
import pandas as pd
import os


'''
Doesnt take into account shunt admittances yet (bdict or shunt powers)
'''

def export_Ybus(Ybus_pu, Ybus, filepath):
    # current_directory = filepath
    # final_directory = os.path.join(current_directory, 'Ybus')
    # if not os.path.exists(final_directory):
    #     os.makedirs(final_directory)
    # os.chdir(final_directory)
    os.chdir(filepath)
    Ybus_pu_df = pd.DataFrame(Ybus_pu)
    Ybus_df = pd.DataFrame(Ybus)
    #then save results to a csv
    Ybus_pu_df.to_csv('Ybus_pu.csv')
    Ybus_df.to_csv('Ybus.csv')
    # os.chdir(current_directory)
    return

def calc_Ybus(myfeeder):
    #buses in Y are ordered according to busdict() order, which is in order of the rows of the excel file
    #need to put the substation in the first entry
    busidx = []
    for Vsrc, bus in myfeeder.Vsrcdict.items():
        substation = Vsrc #this doesnt support networks with more than one substation
        # print('substation at node ' + str(substation))
    busidx.append(substation)
    for key, bus in myfeeder.busdict.items():
        if key != substation:
            busidx.append(key)

    nbuses = len(busidx)
    Ypu = np.zeros((nbuses*3, nbuses*3), dtype=np.complex_)
    Y = np.zeros((nbuses*3, nbuses*3), dtype=np.complex_)

    for key, line in myfeeder.linedict.items():
        a = busidx.index(line.from_node.buskey)
        b = busidx.index(line.to_node.buskey)

        Ypu[a*3:(a+1)*3, b*3:(b+1)*3] = -np.linalg.pinv(line.Zpu)
        Ypu[b*3:(b+1)*3, a*3:(a+1)*3] = -np.linalg.pinv(line.Zpu)
        Y[a*3:(a+1)*3, b*3:(b+1)*3] = -np.linalg.pinv(line.Z)
        Y[b*3:(b+1)*3, a*3:(a+1)*3] = -np.linalg.pinv(line.Z)

    for key, xfmr in myfeeder.transdict.items():
        a = busidx.index(xfmr.w0_node.buskey)
        b = busidx.index(xfmr.w1_node.buskey)
        Ypu[a*3:(a+1)*3, b*3:(b+1)*3] = -np.reciprocal(xfmr.Zpu, out=np.zeros_like(xfmr.Zpu), where=xfmr.Zpu!=0)
        Ypu[b*3:(b+1)*3, a*3:(a+1)*3] = -np.reciprocal(xfmr.Zpu, out=np.zeros_like(xfmr.Zpu), where=xfmr.Zpu!=0)
        Zbase = myfeeder.busdict[xfmr.w0_node.buskey].Zbase #the non-pu Z of a transformer is not defined, so I jsut used the Zbase at w0
        Y[a*3:(a+1)*3, b*3:(b+1)*3] = -np.reciprocal(xfmr.Zpu*Zbase, out=np.zeros_like(xfmr.Zpu), where=xfmr.Zpu!=0)
        Y[b*3:(b+1)*3, a*3:(a+1)*3] = -np.reciprocal(xfmr.Zpu*Zbase, out=np.zeros_like(xfmr.Zpu), where=xfmr.Zpu!=0)

    for key, switch in myfeeder.switchdict.items():
        a = busidx.index(switch.from_node.buskey)
        b = busidx.index(switch.to_node.buskey)
        Ypu[a*3:(a+1)*3, b*3:(b+1)*3] = -np.reciprocal(switch.Zpu, out=np.zeros_like(switch.Zpu), where=switch.Zpu!=0)
        Ypu[b*3:(b+1)*3, a*3:(a+1)*3] = -np.reciprocal(switch.Zpu, out=np.zeros_like(switch.Zpu), where=switch.Zpu!=0)
        Y[a*3:(a+1)*3, b*3:(b+1)*3] = -np.reciprocal(switch.Z, out=np.zeros_like(switch.Z), where=switch.Z!=0)
        Y[b*3:(b+1)*3, a*3:(a+1)*3] = -np.reciprocal(switch.Z, out=np.zeros_like(switch.Z), where=switch.Z!=0)

    for i in np.arange(nbuses):
        rowsumYpu = np.zeros((3,3), dtype=np.complex_)
        rowsumY = np.zeros((3,3), dtype=np.complex_)
        for k in np.arange(nbuses):
            rowsumYpu = rowsumYpu + Ypu[i*3:(i+1)*3,k*3:(k+1)*3]
            rowsumY = rowsumY + Y[i*3:(i+1)*3,k*3:(k+1)*3]
        Ypu[i*3:(i+1)*3,i*3:(i+1)*3] = -rowsumYpu
        Y[i*3:(i+1)*3,i*3:(i+1)*3] = -rowsumY

    for key, shunt in myfeeder.shuntdict.items():
        pass
        #this will require making the shunt elements constant impedance, and putting them in pu
    return(Ypu, Y, busidx)


#Uncomment this code to test the above functions

'''
timesteps = 1
testcase = '13balFlatLoad' #actually balanced (13bal has some imbalances)
# testcase = '13unb'
testcase = '1'
testcase = '2'
# testcase = '13bal'

#Test Case
if testcase == '37':
    loadfolder = os.getcwd() + "/IEEE37/"
    modelpath = loadfolder + "003_GB_IEEE37_OPAL_reform.xls"
    loadpath = loadfolder + "003_GB_IEEE37_time_sigBuilder_1300-1400_norm05.xlsx"
    # Specify substation kV and kVA bases of the load data
    subkVbase_phg = 4.8/np.sqrt(3)
    subkVAbase = 2500
    #shouldnt this come from the load file?
elif testcase == '13unb':
    loadfolder = os.getcwd() + "/IEEE13unb/"
    modelpath = loadfolder + "001_phasor08_IEEE13.xls"
    loadpath = loadfolder + "001_phasor08_IEEE13_norm03_HIL_7_1.xlsx"
    # Specify substation kV and kVA bases of the load data
    subkVbase_phg = 4.16/np.sqrt(3)
    subkVAbase = 5000
elif testcase == '13bal':
    loadfolder = os.getcwd() + "/IEEE13bal/"
    modelpath = loadfolder + "016_GB_IEEE13_balance_all_ver2.xls"
    loadpath = loadfolder + "016_GB_IEEE13_balance_sigBuilder_Q_12_13_norm03_3_1.xlsx"
    # Specify substation kV and kVA bases of the load data
    subkVbase_phg = 4.16/np.sqrt(3)
    subkVAbase = 5000
elif testcase == '13balFlatLoad':
    loadfolder = os.getcwd() + "/IEEE13balFlatLoad/"
    modelpath = loadfolder + "016_GB_IEEE13_balance_all_ver2.xls"
    loadpath = loadfolder + "016_GB_IEEE13_balance_sigBuilder_Q_12_13_norm03_3_1.xlsx"
    # Specify substation kV and kVA bases of the load data
    subkVbase_phg = 4.16/np.sqrt(3)
    subkVAbase = 5000
elif testcase == '2':
    loadfolder = os.getcwd() + "/splitLine3Load/"
    # modelpath = loadfolder + "splitLine3LoadNetwork.xls"
    modelpath = loadfolder + "splitLine3LoadNetworkCrossZ.xls"
    loadpath = loadfolder + "threeLoad.xlsx"
    # Specify substation kV and kVA bases of the load data
    subkVbase_phg = 1
    subkVAbase = 3000
elif testcase == '1':
    loadfolder = os.getcwd() + "/singleLine/"
    # modelpath = loadfolder + "singleSwitchNetwork.xls" #switch impedance set to a default in switchbuilder
    # modelpath = loadfolder + "singleTransformerNetwork.xls"
    # modelpath = loadfolder + "singleLineNetwork.xls"
    modelpath = loadfolder + "singleLineNetworkCrossZ.xls"
    loadpath = loadfolder + "singleLoad.xlsx"
    # Specify substation kV and kVA bases of the load data
    subkVbase_phg = 1
    subkVAbase = 3000 #keeps the lines from being adjusted by iconn.Zbase = iconn.kVbase_phg*iconn.kVbase_phg*1000/iconn.kVAbase in set_per_unit
else:
    error('error laoding model')

subkVAbase = subkVAbase/3
modeldata = pd.ExcelFile(modelpath)
actpath = loadpath

# Create feeder object
myfeeder = feeder(modelpath,loadfolder,loadpath,actpath,timesteps,subkVbase_phg,subkVAbase)
#feeder doesnt actually put the shunt portion of the line impedances on the network (bdict doesnt go anywhere)

# Ypu = getYpu(myfeeder,timestep = 0) #pu convention would run into trouble w a mesh network
# key = '675'
# Zeffk = calcZeffk(Ypu, key)

print(vars(myfeeder))

(Ypu, Y, Manualbusidx) = calc_Ybus(myfeeder)

(Ydss, Ycsc, DSSbusidx, slackbusidx) = DSS_getY_fromFeeder(myfeeder)

print('Ydss')
print(Ydss)

print('Ycsc')
print(Ycsc)
print(Ycsc.toarray())

print('Y')
print(Y)

print('Ypu')
print(Ypu)

sys.exit()

# ISSUE HERE: calc_Ybus appears to not be matching Ydss for 13 node feeder?
#think theres something going on with the indexing in DSS_getY_fromFeeder (dss may do some weird bus reordering)

print('Manualbusidx : ' + str(Manualbusidx))
print('DSSbusidx : ' + str(DSSbusidx))
print('DSSslackbusidx : ', slackbusidx)
print('Y[0:3,0:3]: ', Y[0:3,0:3])
print('Ypu[0:3,0:3]: ', Ypu[0:3,0:3])
print('Ydss[slackbusidx*3:slackbusidx*3+3,slackbusidx*3:slackbusidx*3+3]: ', Ydss[slackbusidx*3:slackbusidx*3+3,slackbusidx*3:slackbusidx*3+3])
for i in np.arange(14):
    print('i :', i)
    print('Ydss[i*3:i*3+3,i*3:i*3+3]: ', Ydss[i*3:i*3+3,i*3:i*3+3])


sys.exit()
'''
