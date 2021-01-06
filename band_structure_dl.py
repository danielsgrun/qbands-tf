########################################
#### 2+2-WELL ENERGY BAND STRUCTURE ####
####          & DEEP LEARNING ALG's.####
#### By: Daniel S Grun              ####
########################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
#from numpy.linalg import matrix_power
#from scipy import signal
from tqdm import tqdm
import time

########################
#### DELTA FUNCTION ####
########################

def d(x,y):
    if x == y:
        ans = 1
    else:
        ans = 0
    return ans

###########################
#### SYSTEM PARAMETERS ####
###########################

start = time.time()

M = 4
P = 11

def lambda1(U):
    lll = 0
    if U > 0:
        lll = J*J/(16*U*(M-P+1))
    else:
        lll = 0
    return lll

def lambda2(U):
    lll = 0
    if U > 0:
        lll = J*J/(16*U*(M-P-1))
    else:
        lll = 0
    return lll

def lambda12(U):
    lll = 0
    if U > 0:
        lll = J*J/(16*U)*(1./(M-P+1)-1./(M-P-1))
    else:
        lll = 0
    return lll

n = M+P

#U0 = 76.519
#J = 73.219
#mu = 15.168

U0 = 75.876
J = 24.886
mu = 0.

dim = int((n+3)*(n+2)*(n+1)/6)

psi0 = np.zeros(dim)
Hint = np.zeros((dim,dim))
Htun = np.zeros((dim,dim))
#Hbre = np.zeros((dim,dim))
Heff1 = np.zeros((dim,dim))
Heff2 = np.zeros((dim,dim))
Heff12 = np.zeros((dim,dim))


# 2+2 system
s = 0
for i1 in range(0,n+1):
  for j1 in range(0,n+1-i1):
    for k1 in range(0,n+1-i1-j1):
      l1 = n-k1-j1-i1
      ss = 0
      for i2 in range(0,n+1):
        for j2 in range(0,n+1-i2):
          for k2 in range(0,n+1-i2-j2):
            l2 = n-k2-j2-i2

            Hint[ss,s] = (i1+k1-j1-l1)**2 * d(i2,i1)*d(j2,j1)*d(k2,k1)*d(l2,l1)

            Htun[ss,s] = ( (np.sqrt(i1*(j1+1.))*d(i2,i1-1)*d(j2,j1+1) 
                          + np.sqrt(j1*(i1+1.))*d(i2,i1+1)*d(j2,j1-1))*d(k2,k1)*d(l2,l1)
                          +(np.sqrt(j1*(k1+1.))*d(j2,j1-1)*d(k2,k1+1) 
                          + np.sqrt(k1*(j1+1.))*d(j2,j1+1)*d(k2,k1-1))*d(i2,i1)*d(l2,l1)
                          +(np.sqrt(k1*(l1+1.))*d(k2,k1-1)*d(l2,l1+1) 
                          + np.sqrt(l1*(k1+1.))*d(k2,k1+1)*d(l2,l1-1))*d(i2,i1)*d(j2,j1)
                          +(np.sqrt(i1*(l1+1.))*d(i2,i1-1)*d(l2,l1+1) 
                          + np.sqrt(l1*(i1+1.))*d(i2,i1+1)*d(l2,l1-1))*d(j2,j1)*d(k2,k1) ) 
            
            Heff1[ss,s] = ( (np.sqrt(k1*(i1+1.))*d(i2,i1+1)*d(k2,k1-1) 
                           +np.sqrt(i1*(k1+1.))*d(i2,i1-1)*d(k2,k1+1))*d(j2,j1)*d(l2,l1) ) 
            
            Heff2[ss,s] = ( (np.sqrt(j1*(l1+1.))*d(j2,j1-1)*d(l2,l1+1) 
                           +np.sqrt(l1*(j1+1.))*d(j2,j1+1)*d(l2,l1-1))*d(i2,i1)*d(k2,k1) )
            
            Heff12[ss,s] = ( (np.sqrt((i1+1.)*(l1+1.)*j1*k1)*d(i2,i1+1)*d(j2,j1-1)*d(k2,k1-1)*d(l2,l1+1) 
                  +np.sqrt((i1+1.)*(j1+1.)*k1*l1)*d(i2,i1+1)*d(j2,j1+1)*d(k2,k1-1)*d(l2,l1-1) 
                  +np.sqrt((j1+1.)*(k1+1.)*i1*l1)*d(i2,i1-1)*d(j2,j1+1)*d(k2,k1+1)*d(l2,l1-1) 
                  +np.sqrt((k1+1.)*(l1+1.)*i1*j1)*d(i2,i1-1)*d(j2,j1-1)*d(k2,k1+1)*d(l2,l1+1)) )
            
            ss += 1
      s += 1


plt.close('all')

U = np.linspace(0,0.25*J,num=1000)
eigvals = np.zeros((len(U),dim))
#eigvals_eff = np.zeros((len(U),dim))

#mu = np.array(np.arange(0,150,3))

contador = 0
cont = 0

for i in tqdm(range(0,len(U))):

  eigvals[i] = la.eigvals(-U[i]*Hint - 0.5*J*Htun)
  eigvals[i].sort()
  
#  eigvals_eff[i] = la.eigvals(lambda1(U[i])*Heff1 - 
#                              lambda2(U[i])*Heff2 + 
#                              lambda12(U[i])*Heff12)
#  eigvals_eff[i].sort()



#xy_dupla = []

#for i in range(0,dim):
#    for j in range(0,len(U)):
#        xy_dupla.append([U[j]/J, eigvals[j,i]/J])

#xy_dupla = np.array(xy_dupla)


#f1 = plt.figure(figsize=(11,8))
#for i in range(0,dim):
    
#    plt.plot(U/J, eigvals[:,i]/J,alpha=0.4,lw=1.0, color='cyan')
#    plt.plot(U/J, eigvals_eff[:,i]/J, alpha=0.4, lw=1.0, ls='dotted', color='red')

#plt.xlabel("$U/J$", fontsize=21)
#plt.xticks(fontsize=21)
#plt.yticks(fontsize=21)
#plt.xlim(0,U[-1]/J)
#plt.xlim(0,1.5)
#plt.ylim(-30, 10)
#plt.ylabel("E/J", fontsize=21)
#plt.axvline(x=U0/J, color='k', ls='--', alpha=0.7)
#plt.show()




from sklearn.model_selection import train_test_split

from tensorflow.keras import Sequential # Sequential is the neural-network class
from tensorflow.keras.layers import Dense # Dense is the standard network layer

from tensorflow.keras.optimizers import Adam

opt = Adam(learning_rate=0.0002)

Net=Sequential() # creating a neural network!

Net.add(Dense(40,input_shape=(1,),activation="relu")) # first hidden layer: 40 neurons (and 1 input neuron!)

Net.add(Dense(2000,activation="relu"))

#Net.add(Dense(10000,activation="relu")) 

Net.add(Dense(dim,activation="linear")) # output layer: 1 neuron "relu"

# Compile network: (randomly initialize weights, choose advanced optimizer, set up everything!)
Net.compile(loss='mean_squared_error',
              optimizer=opt) # adam is adaptive and works better than normal gradient descent

Xdata = (U-np.average(U))/np.std(U)
ydata = eigvals/J
for i in range(0,len(U)):
    ydata[:][i] = (ydata[:][i] - np.average(ydata[:][i]))/np.std(ydata[:][i])

x_train, x_test = train_test_split(Xdata, random_state = 23, train_size=0.7)
y_train, y_test = train_test_split(ydata, random_state = 23, train_size=0.7)

x_train, x_val = train_test_split(x_train, random_state = 50, train_size=0.5)
y_train, y_val = train_test_split(y_train, random_state = 50, train_size=0.5)
    
history=Net.fit(x_train,y_train,epochs=5000, batch_size=700, validation_data=(x_val,y_val), verbose=1)

y_train_predict = Net.predict(x_train)
y_test_predict = Net.predict(x_test)
y_val_predict = Net.predict(x_val)

f2 = plt.figure(figsize=(11,8))
for i in range(0,dim-1):
    
    plt.scatter(x_train, y_train[:,i] - y_train_predict[:,i], alpha=0.4,s=40, color='royalblue')
    plt.scatter(x_test, y_test[:,i] - y_test_predict[:,i], alpha=0.4, s=40, color='cyan')
    plt.scatter(x_val, y_val[:,i] - y_val_predict[:,i], alpha=0.4, s=40, color='purple')
    
plt.scatter(x_train, y_train[:,-1] - y_train_predict[:,-1], alpha=0.4,s=40, color='royalblue', label='Train')
plt.scatter(x_test, y_test[:,-1] - y_test_predict[:,-1], alpha=0.4, s=40, color='cyan', label='Test')
plt.scatter(x_val, y_val[:,-1] - y_val_predict[:,-1], alpha=0.4, s=40, color='purple', label='Validation')

plt.legend(loc=0, fontsize=21)
plt.xlabel("$\\tilde{U}/J$", fontsize=21)
plt.xticks(fontsize=21)
plt.yticks(fontsize=21)
#plt.xlim(0,U[-1]/J)
#plt.xlim(0,1.5)
#plt.ylim(-30, 10)
plt.ylabel("$\Delta \\tilde{E}/J$", fontsize=21)
#plt.axvline(x=U0/J, color='k', ls='--', alpha=0.7)
plt.show()

f3 = plt.figure(figsize=(11,8))
plt.plot(history.history['loss'], lw=3.0, alpha=0.8, color='gray', label="Training loss")
plt.plot(history.history['val_loss'], lw=3.0, alpha=0.8, color='royalblue', label="Testing loss")
plt.legend(loc=0, fontsize=21)
plt.yscale('log')
plt.xlabel("Epoch", fontsize=21)
plt.xticks(fontsize=21)
plt.yticks(fontsize=21)
#plt.axvline(x=U0/J, color='k', ls='--', alpha=0.7)
plt.show()


xvar = Xdata
yvar = Net.predict(xvar)

f4 = plt.figure(figsize=(11,8))
for i in range(0,dim-1):
    plt.plot(xvar, yvar[:,i], lw=1.5, alpha=0.2, color='royalblue')
plt.plot(xvar, yvar[:,-1], lw=1.5, alpha=0.2, color='royalblue', label="Neural network")

plt.legend(loc=0, fontsize=21)
plt.xlabel("$\\tilde{U}/J$", fontsize=21)
plt.xticks(fontsize=21)
plt.ylabel("$\\tilde{E}/J$", fontsize=21)
plt.yticks(fontsize=21)


f5 = plt.figure(figsize=(11,8))
for i in range(0,dim-1):
    plt.plot(Xdata, ydata[:,i], lw=1.5, alpha=0.2, color='royalblue')
plt.plot(Xdata, ydata[:,-1], lw=1.5, alpha=0.2, color='royalblue', label="Exact diagonalization")


plt.legend(loc=0, fontsize=21)
plt.xlabel("$\\tilde{U}/J$", fontsize=21)
plt.xticks(fontsize=21)
plt.ylabel("$\\tilde{E}/J$", fontsize=21)
plt.yticks(fontsize=21)