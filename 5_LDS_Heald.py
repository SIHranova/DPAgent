#%%  Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


#%% Basic stochastic linear dynamical system simulation
### system params
a = 0.3                            # self retention rate
d = 0.5                            # drift rate
sigma_q = 0.1                      # dynamics noise

x_0 = 1                            # initial position
T = 1000                           # how many time points to simulate 
N  = 1000                          # hom many simulations to run
X = np.zeros([N, T+1])             # simulation log
X[:,0] = x_0                       # initial position


### simulate system
for t in range(T):
    X[:,t+1] = a*X[:,t] + d + np.random.normal(loc=0, scale=np.sqrt(sigma_q),size=N)


###  plot simulation trajectories
# plt.figure()
# plt.plot(np.arange(T+1),X.T);


### plot distribution of x at T compared to analytic result (Heald et. al, 2021 Eq.4)

steady_state = X[:,-1]                        # x after it has converged

x_lim = [-0.5, 2]                             # plot limits

mu = d/(1-a)                                  # analytic mean as t -> infinity 
sigma = sigma_q/(1-a**2)                      # analytic variance as t -> infinity

x = np.arange(-0.5,2, 0.01)                   
y = norm.pdf(x,loc=mu, scale=np.sqrt(sigma))  # pdf with analytic mean and variance 

# plot empricial density vs analytics result
plt.figure()
plt.hist(steady_state,bins=30,density=True)
plt.plot(x,y)
plt.xlim(x_lim)
plt.title("Analytic distibution of x as t-> infty vs empirical")

#%% Infer p(x_t|y_1,...,y_t) if all parameters known (E step)
np.random.seed(42)
# a = 0.3                            # self retention rate
# d = 0.5                            # drift rate
# sigma_q = np.sqrt(0.1)                      # dynamics noise
# sigma_r = np.sqrt(0.05)                     # observations noise

a = 0.9 # 0.3                            # self retention rate
d = 0.5 # 0.5                            # drift rate
sigma_q = 0.3# 0.1                      # dynamics noise
sigma_r = 0.5

x_0 = 1                            # initial position
T = 100                            # how many time points to simulate 
N  = 1                             # hom many simulations to run
X = np.zeros(T+1)                  # simulation log
Y = np.zeros(T+1)
X[0] = x_0                         # initial position


### simulate system

for t in range(T):
    X[t+1] = a*X[t] + d + np.random.normal(loc=0, scale=sigma_q)
    Y[t+1] = X[t+1] + np.random.normal(loc=0,scale=sigma_r)


### Kalman filter

x_filt = np.zeros(T+1)
P_filt = np.zeros(T+1)
x_pred = np.zeros(T+1)
P_pred = np.zeros(T+1)
Kalman_gain = np.zeros(T+1)

x_filt[0] = 0
P_filt[0] = 1

for t in range(T+1):

    if t==0:
        x_pred[t] = x_filt[t]
        P_pred[t] = P_filt[t]
    else:
        x_pred[t] = a*x_filt[t-1] + d
        P_pred[t] = a**2*P_filt[t-1] + sigma_q**2


    Kalman_gain[t] = P_pred[t]/(P_pred[t]+sigma_r**2)
    x_filt[t] = x_pred[t] + Kalman_gain[t]*(Y[t] - x_pred[t])
    P_filt[t] = (1-Kalman_gain[t])*P_pred[t]

    pass

plt.figure()
# plt.gca().set_prop_cycle(None)
plt.plot(np.arange(T+1), X.T, '--k', label="true X")
plt.scatter(np.arange(T+1), Y.T, label="observation Y",s=10, alpha=0.6)
plt.plot(np.arange(T+1), x_filt  , label="filtered X")
# plt.title("True trajectory (solid) vs noisy observation (dashed)")
plt.legend()
plt.ylim([-0.3,7])
plt.xlim([-1,101])



# %%
