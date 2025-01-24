#%%
### SOME VERSION OF W IMPLEMENTED
import numpy as np
from scipy.special import digamma, softmax
import matplotlib.pyplot as plt
from environment import data, W, plot_heatmap


class HDP():
    
    def __init__(self):
        pass
    
    
    def initialize_HDP(self, lambda_H=np.ones(10), TAU=10, gamma=2, alpha=1, kappa=3, K=0, max_context=6):
        
        self.max_context = max_context   # max number of contexts
        self.K = K                       # current number of contexts
        self.TAU = TAU
        self.gamma = gamma
        self.alpha = alpha
        self.kappa = kappa
        self.rho = kappa/(alpha+kappa)
        self.lambda_H = lambda_H                                                 # parameters of base Measure H = Dir(lambda)
        self.generative_model_counts = np.zeros([lambda_H.size,self.max_context])
        self.generative_model_counts[:,0] = lambda_H                             # int_phi Cat(x|phi)Dir(phi|lambda_H) = Cat(x|lambda_H)  
        self.generative_model_obs = np.nan_to_num(self.generative_model_counts/self.generative_model_counts.sum(axis=0))
        self.observations  = np.zeros([self.TAU,W],dtype=int)
        self.context = np.zeros(TAU, dtype=int)


    def initialize_beliefs(self):

        self.prior_context = np.zeros(self.max_context)
        self.prior_context[0] = 1                                                         # context prior p(c1)

        self.beta_prime = np.zeros(self.max_context)                                      # Expectation of stick break beta'_k = gamma_1/(gamma_1+gamma_2) 
        self.global_prior_counts = np.zeros([self.max_context,2])                         # parameters gamma_1, gamma_2 of beta_k: p(beta'_k|gamma_k1, gamma_k2)
        self.global_prior = np.zeros(self.max_context)                                    # p(z|gamma_1, gamma_2)  
        self.global_prior[0] = 1

        self.transition_matrix_counts = np.zeros([self.max_context, self.max_context,2])  # parameters alpha_1, alpha_2 of pi_jk: p(pi_jk|alpha_1, alpha_2)
        self.transition_matrix = np.zeros([self.max_context, self.max_context])           # transition matrix Pi: p(c_t|c_t-1, alpha_1, alpha_2) with stick breaks integrated out
        self.transition_matrix[0,0] = 1 

        self.posterior_context = np.zeros([self.TAU, self.max_context])

    def update_beliefs_context(self,obs,tau):
        
        self.observations[tau] = obs

        # if tau == 0:
        #     obs_messages = np.array([self.generative_model_obs[self.observations[tau]], np.ones(self.max_context)])
        #     q_z = np.array([self.global_prior, np.ones(self.max_context)])
        # else:
        #     obs_messages = self.generative_model_obs[self.observations[tau-1:tau+1]]
        #     q_z = np.array([self.global_prior, self.global_prior])

        if tau == 0:

            obs_messages = np.array([
                           self.generative_model_obs[self.observations[tau]].prod(axis=0),\
                           np.ones(self.max_context)\
                           ])

            q_z = np.array([self.global_prior, np.ones(self.max_context)])

        else:

            obs_messages = np.array([
                           self.generative_model_obs[self.observations[tau-1]].prod(axis=0),\
                           self.generative_model_obs[self.observations[tau]].prod(axis=0)\
                           ])
            
            # q_z = np.array([self.global_prior, self.global_prior])

            q_w = np.array([self.rho, 1-self.rho])
            z = np.array([np.eye(self.max_context)[self.context[tau]-1], self.global_prior]).T.dot(q_w)

            q_z = np.array([z,z])

        obs_messages = obs_messages*q_z
        
        if tau < 2:
            prior_context = self.prior_context
        else:
            prior_context = self.transition_matrix.dot(self.posterior_context[tau-2])
        
        # joint q(c_t,c_t-1)
        q_c = self.transition_matrix*prior_context[None,:]*obs_messages[0,:][:,None]*obs_messages[1,:][None,:]
        
        # joint q(c_t)
        q_c = (q_c/q_c.sum()).sum(axis=1)
        self.posterior_context[tau] = q_c

        assert np.isclose(q_c.sum(),1)
        
        return q_c


    def update_beliefs(self, observation,tau):

        # print(f"---------\ntau={tau}")

        if tau == 0:
            self.initialize_beliefs()
        
        q_c = self.update_beliefs_context(observation,tau)
        
        # w_t = np.random.binomial(1,self.rho)
        
        # if w_t == True and tau > 0:
        #     current_context = self.context[tau-1]
        #     # print(self.alpha, self.kappa, self.rho)
        #     # print("overrode context transition")
        # else:  
        #     current_context = np.argmax(q_c)

        current_context = np.argmax(q_c)        

        self.context[tau] = current_context
        
        if current_context + 1 > self.K:    # count from 1, not zero
            # print("\nopening new context\n")
            self.K += 1
            self.global_prior_counts[self.K-1] = [1,self.gamma]                                                       # initialize prior over new beta'_k
            self.generative_model_counts[:,self.K] = self.lambda_H
            self.transition_matrix_counts[np.arange(self.K), [self.K-1]*(self.K), :] = [1,self.alpha+self.kappa]
            self.transition_matrix_counts[[self.K-1]*(self.K), np.arange(self.K), :] = [1,self.alpha+self.kappa]
        else:
            pass

        # update gamma of q(beta'|gamma)
        counts = np.zeros([self.max_context,2])
        counts[current_context,0] += 1
        counts[:current_context,1] += 1
        self.global_prior_counts += counts
        self.global_prior = self.construct_G_0(approx=False)

        # update lambda of q(phi|lambda)
        for obs in observation:
            self.generative_model_counts[obs, current_context] += 1

        # print(tau,observation)
        # print(self.generative_model_counts)    
        self.generative_model_obs = np.nan_to_num(self.generative_model_counts/self.generative_model_counts.sum(axis=0))

        # print(f"\ndata likelihood:\n{self.generative_model_counts}")

        # update phi of q(c_t|c_t-1,phi)
        if tau > 0:

            # print(f"tau: {tau}, contexts: {self.context[tau-1]},{self.context[tau]}")
            counts = np.zeros([self.max_context, self.max_context, 2])
            counts[self.context[tau], self.context[tau-1], 0] = 1
            counts[:self.context[tau], self.context[tau-1], 1] = 1 

            self.transition_matrix_counts = self.transition_matrix_counts + counts

        self.transition_matrix = self.construct_G_j(approx=False)

        # self.prior_context = np.eye(self.max_context)[current_context]


    def construct_G_0(self, approx=False):

        global_prior = np.zeros(self.max_context)
        
        if not approx:
            beta_prime = np.nan_to_num(self.global_prior_counts/self.global_prior_counts.sum(axis=1)[:,None])  # expected b_k'            
            beta_prime_l = np.insert(np.cumprod(beta_prime[:,1]),0,1)                                   #  prod_l=1^k-1 (1-beta'_l)
            beta_prime_k = np.insert(beta_prime[:,0], self.K, 1)

            for k in range(self.K+1):
                global_prior[k] = beta_prime_k[k]*beta_prime_l[k]

            # print(self.K, global_prior, global_prior.sum())

        else:
        
            beta_prime = digamma(self.global_prior_counts)
            beta_prime_l = np.insert(np.cumsum(beta_prime[:,1]),0,0)
            beta_prime_k = np.insert(beta_prime[:,0],self.K,0)
            
            # print(digamma(self.global_prior_counts.sum(axis=1))) 
            norm = np.cumsum(digamma(self.global_prior_counts.sum(axis=1)))
            norm = np.insert(norm, self.K, norm[self.K-1])
            
            for k in range(self.K+1):
                global_prior[k] = beta_prime_k[k] + beta_prime_l[k] - norm[k]
            
            global_prior[:self.K+1] = softmax(global_prior[:self.K+1])
            
        # print(f"global_prior:\n {self.global_prior_counts}, {global_prior}")

        return global_prior


    def construct_G_j(self, approx=False):
        
        transition_matrix = np.zeros([self.max_context, self.max_context])
        # initialize 2K+1 pi_jk with prior probability (1,alpha)

        pi_prime = np.nan_to_num(self.transition_matrix_counts / self.transition_matrix_counts.sum(axis=-1)[:,:,None]) # expected pi_jk'
        # construct q(c_t|c_t-1,alpha) = pi'_jk * prod_l=1^k-1 (1-pi'_jl), where pi'_jk = alpha_jk1/(alpha_jk2)
        pi_prime_l = np.insert(np.cumprod(pi_prime[:,:,1], axis=0), 0, 1, axis=0)
        pi_prime_k = np.insert(pi_prime[:,:,0], self.K, 1, axis=0)

        for k in range(self.K+1):
            transition_matrix[k] = pi_prime_k[k,:]*pi_prime_l[k,:]

        assert np.all(np.isclose(transition_matrix.sum(axis=0)[:self.K],1))

        return transition_matrix



i = 0
for gamma in np.arange(3,9,0.5):#,2,0.1):
    for alpha in np.arange(5,9,0.5):#,3,0.1):
        for kappa in np.arange(3,9,0.5):#,3,0.1):
            # print(gamma,alpha,kappa)
            # gamma = 2.1
            # alpha = 5.1
            # kappa = 2.1
            agent = HDP()
            agent.initialize_HDP(TAU=data.shape[0],alpha=alpha, gamma=gamma,kappa=kappa)

            for tau in range(int(data[:,0].size / 5)): #]):
                agent.update_beliefs(data[tau*W:(tau+1)*W,0],tau)

            # for tau, obs in enumerate(data[:,0]): #]):
            #     agent.update_beliefs(obs,tau)
            
            if agent.K == 3:
                print(agent.K)
                plt.plot(agent.generative_model_obs[:,:agent.K])
                plt.savefig(str(i)+"_0.png")
                plt.close()
                # plot_heatmap(agent.generative_model_obs, ind= str(i) + "_1", title= f"phi - gamma: {gamma.round(3)}, alpha: {alpha.round(3)}, kappa: {kappa.round(3)}")
                plot_heatmap(agent.transition_matrix, ind = str(i) + "_2", title=f"trans matrix - gamma: {gamma.round(3)}, alpha: {alpha.round(3)}, kappa: {kappa.round(3)}")
            # plot_heatmap(agent.generative_model_obs, title= f"phi - gamma: {gamma}, alpha: {alpha}, kappa: {kappa}")
            # plot_heatmap(agent.transition_matrix, title=f"trans matrix - gamma: {gamma}, alpha: {alpha}, kappa: {kappa}")

            i += 1
            # plot_heatmap(agent.transition_matrix_counts[:,:,0])
            # plot_heatmap(agent.transition_matrix_counts[:,:,1])


# %%
