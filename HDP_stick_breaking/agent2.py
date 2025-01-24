#%%
import numpy as np
from scipy.special import digamma, softmax
import matplotlib.pyplot as plt
from environment import data, plot_heatmap


class HDP():
    
    def __init__(self):
        pass
    
    
    def initialize_HDP(self, lambda_H=np.ones(10), TAU=10, gamma=3, alpha=3.5, K=0, max_context=10):
        
        self.max_context = max_context   # max number of contexts
        self.K = K                       # current number of contexts
        self.TAU = TAU
        self.gamma = gamma
        self.alpha = alpha
        self.lambda_H = lambda_H                                                 # parameters of base Measure H = Dir(lambda)
        self.generative_model_counts = np.zeros([lambda_H.size,self.max_context])
        self.generative_model_counts[:,0] = lambda_H                             # int_phi Cat(x|phi)Dir(phi|lambda_H) = Cat(x|lambda_H)  
        self.generative_model_obs = np.nan_to_num(self.generative_model_counts/self.generative_model_counts.sum(axis=0))
        self.observations  = np.zeros([self.TAU],dtype=int)
        self.context = np.zeros(TAU, dtype=int)
        self.posterior_context = np.zeros([TAU,max_context], dtype=int)

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


    def update_beliefs_context(self,obs,tau):
        
        self.observations[tau] = obs

        if tau == 0:
            obs_messages = np.array([self.generative_model_obs[self.observations[tau]], np.ones(self.max_context)])
            q_z = np.array([self.global_prior, np.ones(self.max_context)])
        else:
            obs_messages = self.generative_model_obs[self.observations[tau-1:tau+1]]
            q_z = np.array([self.global_prior, self.global_prior])

        obs_messages = obs_messages*q_z

        obs_message_0 = obs_messages[0,:][:,None]
        obs_message_1 = obs_messages[1,:][None,:]
        

        q_c1_c2 = self.transition_matrix*self.prior_context[None,:]*obs_message_0*obs_message_1
        
        posterior_context = (q_c1_c2/q_c1_c2.sum()).sum(axis=1)
        # print(posterior_context.round(3))
        
        assert np.isclose(posterior_context.sum(),1)
        
        return posterior_context, q_z[0]


    def update_beliefs(self, observation,tau):

        # print(f"---------\ntau={tau}")

        if tau == 0:
            self.initialize_beliefs()
        
        q_c, q_z = self.update_beliefs_context(observation,tau)

        current_context = np.argmax(q_c)
        self.context[tau] = current_context
        self.posterior_context[tau] = q_c

        if current_context + 1 > self.K:    # count from 1, not zero
            # print("\nopening new context\n")
            self.K += 1
            self.global_prior_counts[self.K-1] = [1,self.gamma]                                                       # initialize prior over new beta'_k
            self.generative_model_counts[:,self.K] = self.lambda_H
            self.transition_matrix_counts[np.arange(self.K), [self.K-1]*(self.K), :] = [1,self.alpha]
            self.transition_matrix_counts[[self.K-1]*(self.K), np.arange(self.K), :] = [1,self.alpha]

            # update gamma of q(beta'|gamma)
            counts = np.zeros([self.max_context,2])
            counts[current_context,0] += 1
            counts[:current_context,1] += 1
            self.global_prior_counts += counts
            self.global_prior = self.construct_G_0(approx=False)

            # update lambda of q(phi|lambda)
            self.generative_model_counts[observation, current_context] += 1         
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

            self.prior_context = np.eye(self.max_context)[current_context]

        else:
            # update lambda of q(phi|lambda)
            print(tau,self.K)
            print(self.generative_model_counts[observation])
            print(q_c)
            print(q_z)
            print(q_c[:self.K]*q_z[:self.K])
            print(self.generative_model_counts[observation, :self.K])
            self.generative_model_counts[observation, :self.K] += q_c[:self.K]*q_z[:self.K]
            self.generative_model_obs = np.nan_to_num(self.generative_model_counts/self.generative_model_counts.sum(axis=0))

            #update q(z)
            print("HERE")
            q_z = self.construct_G_0(approx=True) + \
                  q_c*(digamma(self.generative_model_counts[obs,:self.K]) - digamma(self.generative_model_counts.sum(axis=0)[:self.K]))
            q_z[:self.K] = softmax(q_z[:self.K])
            # update gamma of q(beta'|gamma)
            counts = np.zeros([self.max_context,2])
            counts[current_context,0] += 1
            counts[:current_context,1] += 1
            self.global_prior_counts += counts
            self.global_prior = self.construct_G_0(approx=False)


            # print(f"\ndata likelihood:\n{self.generative_model_counts}")

            # update phi of q(c_t|c_t-1,phi)
            if tau > 0:

                # print(f"tau: {tau}, contexts: {self.context[tau-1]},{self.context[tau]}")
                counts = np.zeros([self.max_context, self.max_context, 2])
                counts[self.context[tau], self.context[tau-1], 0] = 1
                counts[:self.context[tau], self.context[tau-1], 1] = 1 

                self.transition_matrix_counts = self.transition_matrix_counts + counts

            self.transition_matrix = self.construct_G_j(approx=False)

            self.prior_context = np.eye(self.max_context)[current_context]


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
            
            # global_prior[:self.K+1] = softmax(global_prior[:self.K+1])
            
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
    

# data = [1,1,1,1,2,1,6,7,7,1,3,4,1,2,6,7,8,9,5,7,6,8,1,3,2,4,2,3,1,8,7,6,7,5,6]

for gamma in np.arange(1,2,0.1):
    for alpha in np.arange(1,3,0.1):
        print(gamma,alpha)
        agent = HDP()
        agent.initialize_HDP(TAU=data.shape[0],alpha=alpha, gamma=gamma)
        for tau, obs in enumerate(data[:,0]): #]):
            agent.update_beliefs(obs,tau)

        plot_heatmap(agent.generative_model_obs,title= f"phi; {gamma.round(3)}, {alpha.round(3)}")
        plot_heatmap(agent.transition_matrix,title=f"trans matrix; {gamma.round(3)}, {alpha.round(3)}")
        # plot_heatmap(agent.transition_matrix_counts[:,:,0])
        # plot_heatmap(agent.transition_matrix_counts[:,:,1])

