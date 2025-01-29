#%%
import numpy as np
from scipy.special import digamma, softmax
# from environment import data, W, component_params_true, transition_matrix_true, plot_heatmap # data - the words spoken, the speaker and the time point; W -  how many observations are passed at a time
import warnings


warnings.filterwarnings("ignore")

class HDP():
    
    def __init__(self,
                 lambda_H=np.ones(10),
                 TAU=10,
                 T=1,
                 gamma=2,
                 alpha=1,
                 kappa=0,
                 K=0,
                 max_context=6,
                 debug=False,

                 state_transition_matrix = None,
                 observation_generation_matrix = None,
                 utility = None,
                 policies = None,
                #  prior_rewards = None,
                #  counts_prior_rewards = None,
                 prior_policies = None,
                 counts_prior_policies = None,
                 prior_states = None,
                 na = None,
                 env = None,
                 approx_pred_pol = None,
                 approx_pred_rew = None,
                ):
        
        self.max_context = max_context                                     # max number of contexts
        self.T = T                                                         # number of observations per episode
        self.K = K                                                         # current number of contexts
        self.TAU = TAU                                                     # number of episodes
        self.gamma = gamma                                                 # cluster opening tendency
        self.alpha = alpha                                                 # transitioning into a new cluster tendency
        self.kappa = kappa                                                 # self transition bias
        self.rho = kappa/(alpha+kappa)
        self.lambda_H = lambda_H                                           # parameters of base Measure H = Dir(lambda)
        self.observations  = np.zeros([self.TAU,self.T],dtype=int)         # array storing observations
        self.context = np.zeros(TAU, dtype=int)                            # array storing inferred context
        self.posterior_context = np.zeros([self.TAU, T, self.max_context]) # array storing posterior over contextss
        self.debug=debug

        self.na = na
        self.nc = K+1
        self.nr = utility.size
        self.ns = prior_states.size
        self.npi = prior_policies.size
        self.env = env
        self.state_transition_matrix = state_transition_matrix
        self.observation_generation_matrix = observation_generation_matrix
        self.policies = policies
        self.prior_policies = prior_policies
        self.counts_prior_policies = counts_prior_policies
        self.utility = utility
        
        self.prior_states = prior_states
        self.approx_pred_pol = approx_pred_pol
        self.approx_pred_rew = approx_pred_rew


    def initialize_beliefs(self):

        self.prior_context = np.zeros(self.max_context)
        self.prior_context[0] = 1                                                         # context prior p(c1)

        self.beta_prime = np.zeros(self.max_context)                                      # Expectation of stick break beta'_k = gamma_k1/(gamma_k1+gamma_k2) 
        self.global_prior_counts = np.zeros([self.max_context,2])                         # parameters gamma_1, gamma_2 of beta_k: p(beta'_k|gamma_k1, gamma_k2)
        self.global_prior = np.zeros(self.max_context)                                    # p(z|gamma_1, gamma_2)  = int_b p(z|b)p(b|gamma_1, gamma_2)
        self.global_prior[0] = 1

        self.transition_matrix_counts = np.zeros([self.max_context, self.max_context,2])  # parameters alpha_1, alpha_2 of stick break pi'_jk: p(pi_jk|alpha_jk1, alpha_jk2)
        self.transition_matrix = np.zeros([self.max_context, self.max_context])           # transition matrix Pi: p(c_t|c_t-1, alpha_1, alpha_2) with stick breaks integrated out
        self.transition_matrix[0,0] = 1

        self.prior_rewards_counts = np.zeros([self.nr, self.ns, self.max_context])
        self.prior_rewards_counts[:,:,0] = self.lambda_H                                # int_phi Cat(x|phi)Dir(phi|lambda_H) = Cat(x|lambda_H)  
        self.prior_rewards = np.nan_to_num(self.prior_rewards_counts/self.prior_rewards_counts.sum(axis=0))

        self.opened_new_context = np.zeros(self.TAU)


    def ln(self, array):
        array[array==0] = 1e-20
        return np.log(array)


    def linear_ind(self, array):
        array = array[:,None].T if array.shape[-1] == 1 else array.T
        return np.ravel_multi_index(array, [self.na]*(self.T-1))


    def initialize_states_messages(self,t,tau):

        # initialize messages for Bethe Approximation Belief Propagation
        self.fwd_messages = np.zeros([self.ns, self.T, self.npi, self.nc]) + 1/self.ns
        self.fwd_messages[:,0,:,:] = self.prior_states[:,None,None]

        self.fwd_norms = np.zeros([self.T+1, self.npi, self.nc])
        self.fwd_norms[0,:,:] = 1                               # accounts for the normalizing constant of the prior

        self.bwd_messages = np.zeros([self.ns, self.T, self.npi, self.nc]) + 1/self.ns
        self.bwd_norms = np.zeros([self.T, self.npi, self.nc])

        self.obs_messages = np.zeros((self.ns, self.T, self.npi, self.nc)) + 1/self.ns

        self.reward_messages = np.zeros([self.ns, self.T, self.npi, self.nc])
        print("HERE")
        rew_mess = np.einsum('r,rsc -> sc', self.utility, self.prior_rewards[tau,t])
        rew_mess /= rew_mess.sum(axis=0)
        self.reward_messages[:] = rew_mess[:,None,None,:]

        # backward message intialization
        for c in range(self.nc):
            for pi, policy in enumerate(self.policies):
                for t, u in zip(np.flip(np.arange(self.T-1)), np.flip(policy)):
                    self.bwd_messages[:,t,pi,c] = (self.bwd_messages[:,t+1,pi,c]*self.obs_messages[:,t+1,pi,c]*self.reward_messages[:,t+1,pi,c])\
                                                   .dot(self.state_transition_matrix[:,:,u])
                                                  
                    self.bwd_norms[t,pi,c] = self.bwd_messages[:,t,pi,c].sum()
                    self.bwd_messages[:,t,pi,c] /= self.bwd_norms[t,pi,c] 


    def update_states_messages(self,t,tau,pi,policy,c,reward,observation):
        
        # update rewards messages based on what was observed
        self.reward_messages[:,t,:,:] = self.prior_rewards[tau,t, reward,:,None,:]
        self.obs_messages[:,t,:,:] = self.observation_generation_matrix[observation,:,None,None]
        
        # perform forward pass
        if (t < self.T-1):
            for tp, u in enumerate(policy):
                self.fwd_messages[:,tp+1,pi,c] = self.state_transition_matrix[:,:,u]\
                                                 .dot(self.fwd_messages[:,tp,pi,c]*self.obs_messages[:,tp,pi,c]*self.reward_messages[:,tp,pi,c])
                self.fwd_norms[tp+1,pi,c] = self.fwd_messages[:,tp+1,pi,c].sum() 
                self.fwd_messages[:,tp+1,pi,c] /= self.fwd_norms[tp+1,pi,c]

        # update backward pass based on observed information
        if(t>0):
            for tp, u in zip(np.flip(np.arange(self.T-1)), np.flip(policy)):
                self.bwd_messages[:,tp,pi,c] = (self.bwd_messages[:,tp+1,pi,c]*self.obs_messages[:,tp+1,pi,c]*self.reward_messages[:,tp+1,pi,c])\
                                                .dot(self.state_transition_matrix[:,:,u])
                self.bwd_messages[:,tp,pi,c] /= self.bwd_messages[:,tp,pi,c].sum() 


    def update_beliefs_states(self, t, tau, reward, action, observation):

        if t==0:
            self.possible_policies = self.policies.copy()
            self.possible_policies_ind = np.arange(self.policies.shape[0])
            self.initialize_states_messages(t,tau)
        
        # check which policies are still possible
        if action is not None:
            self.possible_policies = self.possible_policies[self.possible_policies[:,t-1] == action]
            self.possible_policies_ind = self.linear_ind(self.possible_policies)
            
        for c in range(self.nc):
            for pi, policy in enumerate(self.policies):
                if pi in self.possible_policies_ind:
                    for tp, u in enumerate(policy):
                        self.update_states_messages(t,tau,pi,policy,c,reward,observation)
                else:
                    self.fwd_messages[:,:,pi,:] = 0
                    self.fwd_norms[:,pi,:] = 0

        post  = self.fwd_messages*self.bwd_messages*self.obs_messages*self.reward_messages
        post_norm = post.sum(axis=0)
        post = np.nan_to_num(post/post_norm)

        self.fwd_norms[-1,:,:] = post_norm[-1,:,:]
        self.forward_norms[tau,t] = self.fwd_norms
        self.posterior_states[tau,t,:,:,:,:] = post


    def update_beliefs_context(self,tau,t):

        # Infers q(c_t,c_{t-1}) via Bethe Approximation

        if tau == 0:
            
            obs_messages = np.array([
                             self.prior_rewards[self.observations[tau]].prod(axis=0),\
                             np.ones(self.max_context)\
                           ])

            q_z = np.array([self.global_prior, np.ones(self.max_context)])

        else:

            #  [p'(o_{t,1:W}  | c_{t}  ,z_{t}  )]]
            # [[p'(o_{t-1,1:W}| c_{t-1},z_{t-1})],
            obs_messages = np.array([
                             self.prior_rewards[self.observations[tau-1]].prod(axis=0),\
                             self.prior_rewards[self.observations[tau]].prod(axis=0)\
                           ])
            
            # q_z = \int_{beta} q(z|beta)q(beta)
            q_z = np.array([self.global_prior, self.global_prior])


        # sum_{z_{t}} q(z_t) log \prod_w p'(o_tw|c_t=k,z_t=k) = log \prod_w \phi'_{wk}^{q(z_t=k)}
        obs_messages = obs_messages**q_z
        
        if self.debug:
            print("\ncalculating q(c_t)")
            print('observations and observation messages:')
            print(self.observations[tau-1], self.observations[tau])
            print(obs_messages)
        

        if tau < 2:
            prior_context = self.prior_context
        else:
            prior_context = self.transition_matrix.dot(self.posterior_context[tau-2])
        
        #  q(c_t,c_t-1)
        q_c = self.transition_matrix*prior_context[None,:]*obs_messages[0,:][None,:]*obs_messages[1,:][:,None]
        
        #  q(c_t) = sum_{c_{t-1}} q(c_t,c_{t-1})
        q_c = (q_c/q_c.sum()).sum(axis=1)

        self.posterior_context[tau,t] = q_c

        assert np.isclose(q_c.sum(),1)
        
        return q_c


    def update_beliefs(self, t, tau, state, reward, action, observation):

        # print(f"---------\ntau={tau}")
        self.observations[tau,t] = observation

        if tau == 0:
            self.initialize_beliefs()
        
        self.update_beliefs_states(t, tau, reward, action, observation)


        print("HERE")
        q_c = self.update_beliefs_context(tau,t)
        
        current_context = np.argmax(q_c)
        self.context[tau] = current_context

        if current_context + 1 > self.K:         # if inferred presence of new context
            # print(f"opened new context at tau: {tau}")
            self.K += 1
            self.opened_new_context[tau] = 1

            # add prior over new weight beta'_k
            self.global_prior_counts[self.K-1] = [1,self.gamma]
            
            # add prior over new atom \phi_k
            self.prior_rewards_counts[:,self.K] = self.lambda_H

            # add prior over new transition weights
            self.transition_matrix_counts[np.arange(self.K), [self.K-1]*(self.K), :] = [1,self.alpha]
            self.transition_matrix_counts[[self.K-1]*(self.K), np.arange(self.K), :] = [1,self.alpha]
            
            # add self-transition bias kappa      
            counts = np.zeros([self.max_context, self.max_context, 2])  
            counts[self.K-1, self.K-1, 0] = self.kappa
            counts[:self.K-1, self.K-1, 1] = self.kappa
            
            self.transition_matrix_counts = self.transition_matrix_counts + counts


        # update posterior over global context  q(beta'|gamma)
        counts = np.zeros([self.max_context,2])
        counts[current_context,0] += 1
        counts[:current_context,1] += 1
        self.global_prior_counts += counts
        self.global_prior = self.construct_G_0(approx=False)

        # update lambda of q(phi|lambda)
        for obs in observation:
            self.prior_rewards_counts[obs, current_context] += 1

        # print(tau,observation)
        # print(self.prior_rewards_counts)    
        self.prior_rewards = np.nan_to_num(self.prior_rewards_counts/self.prior_rewards_counts.sum(axis=0))

        # print(f"\ndata likelihood:\n{self.generative_model_counts}")

        # update phi of q(c_t|c_t-1,phi)
        if tau > 0:
            # print(f"tau: {tau}, contexts: {self.context[tau-1]},{self.context[tau]}")
            counts = np.zeros([self.max_context, self.max_context, 2])
            counts[self.context[tau], self.context[tau-1], 0] = 1
            counts[:self.context[tau], self.context[tau-1], 1] = 1 

            self.transition_matrix_counts = self.transition_matrix_counts + counts

        self.transition_matrix = self.construct_G_j(approx=False)
        
        # self.posterior_context[tau] = np.eye(self.max_context)[current_context]


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



# class HDP_speaker_discretization():
    
#     def __init__(self):
#         pass
    

#     def initialize_HDP(self, lambda_H=np.ones(10), TAU=10, gamma=2, alpha=1, kappa=0, K=0, max_context=6,debug=False, obs_batch_size=1):
        
#         self.max_context = max_context                                  # max number of contexts
#         self.W = obs_batch_size                                         # how many observations passed in a single go
#         self.K = K                                                      # current number of contexts
#         self.TAU = TAU                                                  # number of episodes
#         self.gamma = gamma                                              # cluster opening tendency
#         self.alpha = alpha                                              # transitioning into a new cluster tendency
#         self.kappa = kappa                                              # self transition bias
#         self.rho = kappa/(alpha+kappa)
#         self.lambda_H = lambda_H                                        # parameters of base Measure H = Dir(lambda)
#         self.observations  = np.zeros([self.TAU,self.W],dtype=int)           # array storing observations
#         self.context = np.zeros(TAU, dtype=int)                         # array storing inferred context
#         self.posterior_context = np.zeros([self.TAU, self.max_context]) # array storing posterior over contextss
#         self.debug=debug


#     def initialize_beliefs(self):

#         self.prior_context = np.zeros(self.max_context)
#         self.prior_context[0] = 1                                                         # context prior p(c1)

#         self.beta_prime = np.zeros(self.max_context)                                      # Expectation of stick break beta'_k = gamma_k1/(gamma_k1+gamma_k2) 
#         self.global_prior_counts = np.zeros([self.max_context,2])                         # parameters gamma_1, gamma_2 of beta_k: p(beta'_k|gamma_k1, gamma_k2)
#         self.global_prior = np.zeros(self.max_context)                                    # p(z|gamma_1, gamma_2)  = int_b p(z|b)p(b|gamma_1, gamma_2)
#         self.global_prior[0] = 1

#         self.transition_matrix_counts = np.zeros([self.max_context, self.max_context,2])  # parameters alpha_1, alpha_2 of stick break pi'_jk: p(pi_jk|alpha_jk1, alpha_jk2)
#         self.transition_matrix = np.zeros([self.max_context, self.max_context])           # transition matrix Pi: p(c_t|c_t-1, alpha_1, alpha_2) with stick breaks integrated out
#         self.transition_matrix[0,0] = 1 

#         self.generative_model_counts = np.zeros([self.lambda_H.size,self.max_context])
#         self.generative_model_counts[:,0] = self.lambda_H                                # int_phi Cat(x|phi)Dir(phi|lambda_H) = Cat(x|lambda_H)  
#         self.generative_model_obs = np.nan_to_num(self.generative_model_counts/self.generative_model_counts.sum(axis=0))

#         self.opened_new_context = np.zeros(self.TAU)

#     def update_beliefs_context(self,tau):

#         # Infers q(c_t,c_{t-1}) via Bethe Approximation

#         if tau == 0:
            
#             obs_messages = np.array([
#                              self.generative_model_obs[self.observations[tau]].prod(axis=0),\
#                              np.ones(self.max_context)\
#                            ])

#             q_z = np.array([self.global_prior, np.ones(self.max_context)])

#         else:

#             #  [p'(o_{t,1:W}  | c_{t}  ,z_{t}  )]]
#             # [[p'(o_{t-1,1:W}| c_{t-1},z_{t-1})],
#             obs_messages = np.array([
#                              self.generative_model_obs[self.observations[tau-1]].prod(axis=0),\
#                              self.generative_model_obs[self.observations[tau]].prod(axis=0)\
#                            ])
            
#             # q_z = \int_{beta} q(z|beta)q(beta)
#             q_z = np.array([self.global_prior, self.global_prior])


#         # sum_{z_{t}} q(z_t) log \prod_w p'(o_tw|c_t=k,z_t=k) = log \prod_w \phi'_{wk}^{q(z_t=k)}
#         obs_messages = obs_messages**q_z
        
#         if self.debug:
#             print("\ncalculating q(c_t)")
#             print('observations and observation messages:')
#             print(self.observations[tau-1], self.observations[tau])
#             print(obs_messages)
        

#         if tau < 2:
#             prior_context = self.prior_context
#         else:
#             prior_context = self.transition_matrix.dot(self.posterior_context[tau-2])
        
#         #  q(c_t,c_t-1)
#         q_c = self.transition_matrix*prior_context[None,:]*obs_messages[0,:][None,:]*obs_messages[1,:][:,None]
        
#         #  q(c_t) = sum_{c_{t-1}} q(c_t,c_{t-1})
#         q_c = (q_c/q_c.sum()).sum(axis=1)

#         self.posterior_context[tau] = q_c

#         assert np.isclose(q_c.sum(),1)
        
#         return q_c


#     def update_beliefs(self, observation,tau):

#         # print(f"---------\ntau={tau}")
#         self.observations[tau] = observation

#         if tau == 0:
#             self.initialize_beliefs()
        
#         q_c = self.update_beliefs_context(tau)
        
#         current_context = np.argmax(q_c)
#         self.context[tau] = current_context

#         if current_context + 1 > self.K:         # if inferred presence of new context
#             # print(f"opened new context at tau: {tau}")
#             self.K += 1
#             self.opened_new_context[tau] = 1

#             # add prior over new weight beta'_k
#             self.global_prior_counts[self.K-1] = [1,self.gamma]
            
#             # add prior over new atom \phi_k
#             self.generative_model_counts[:,self.K] = self.lambda_H

#             # add prior over new transition weights
#             self.transition_matrix_counts[np.arange(self.K), [self.K-1]*(self.K), :] = [1,self.alpha]
#             self.transition_matrix_counts[[self.K-1]*(self.K), np.arange(self.K), :] = [1,self.alpha]
            
#             # add self-transition bias kappa      
#             counts = np.zeros([self.max_context, self.max_context, 2])  
#             counts[self.K-1, self.K-1, 0] = self.kappa
#             counts[:self.K-1, self.K-1, 1] = self.kappa
            
#             self.transition_matrix_counts = self.transition_matrix_counts + counts


#         # update posterior over global context  q(beta'|gamma)
#         counts = np.zeros([self.max_context,2])
#         counts[current_context,0] += 1
#         counts[:current_context,1] += 1
#         self.global_prior_counts += counts
#         self.global_prior = self.construct_G_0(approx=False)

#         # update lambda of q(phi|lambda)
#         for obs in observation:
#             self.generative_model_counts[obs, current_context] += 1

#         # print(tau,observation)
#         # print(self.generative_model_counts)    
#         self.generative_model_obs = np.nan_to_num(self.generative_model_counts/self.generative_model_counts.sum(axis=0))

#         # print(f"\ndata likelihood:\n{self.generative_model_counts}")

#         # update phi of q(c_t|c_t-1,phi)
#         if tau > 0:
#             # print(f"tau: {tau}, contexts: {self.context[tau-1]},{self.context[tau]}")
#             counts = np.zeros([self.max_context, self.max_context, 2])
#             counts[self.context[tau], self.context[tau-1], 0] = 1
#             counts[:self.context[tau], self.context[tau-1], 1] = 1 

#             self.transition_matrix_counts = self.transition_matrix_counts + counts

#         self.transition_matrix = self.construct_G_j(approx=False)
        
#         # self.posterior_context[tau] = np.eye(self.max_context)[current_context]


#     def construct_G_0(self, approx=False):

#         global_prior = np.zeros(self.max_context)
        
#         if not approx:
#             beta_prime = np.nan_to_num(self.global_prior_counts/self.global_prior_counts.sum(axis=1)[:,None])  # expected b_k'            
#             beta_prime_l = np.insert(np.cumprod(beta_prime[:,1]),0,1)                                   #  prod_l=1^k-1 (1-beta'_l)
#             beta_prime_k = np.insert(beta_prime[:,0], self.K, 1)

#             for k in range(self.K+1):
#                 global_prior[k] = beta_prime_k[k]*beta_prime_l[k]

#             # print(self.K, global_prior, global_prior.sum())

#         else:
        
#             beta_prime = digamma(self.global_prior_counts)
#             beta_prime_l = np.insert(np.cumsum(beta_prime[:,1]),0,0)
#             beta_prime_k = np.insert(beta_prime[:,0],self.K,0)
            
#             # print(digamma(self.global_prior_counts.sum(axis=1))) 
#             norm = np.cumsum(digamma(self.global_prior_counts.sum(axis=1)))
#             norm = np.insert(norm, self.K, norm[self.K-1])
            
#             for k in range(self.K+1):
#                 global_prior[k] = beta_prime_k[k] + beta_prime_l[k] - norm[k]
            
#             global_prior[:self.K+1] = softmax(global_prior[:self.K+1])
            
#         # print(f"global_prior:\n {self.global_prior_counts}, {global_prior}")

#         return global_prior


#     def construct_G_j(self, approx=False):
        
#         transition_matrix = np.zeros([self.max_context, self.max_context])
#         # initialize 2K+1 pi_jk with prior probability (1,alpha)

#         pi_prime = np.nan_to_num(self.transition_matrix_counts / self.transition_matrix_counts.sum(axis=-1)[:,:,None]) # expected pi_jk'
#         # construct q(c_t|c_t-1,alpha) = pi'_jk * prod_l=1^k-1 (1-pi'_jl), where pi'_jk = alpha_jk1/(alpha_jk2)
#         pi_prime_l = np.insert(np.cumprod(pi_prime[:,:,1], axis=0), 0, 1, axis=0)
#         pi_prime_k = np.insert(pi_prime[:,:,0], self.K, 1, axis=0)

#         for k in range(self.K+1):
#             transition_matrix[k] = pi_prime_k[k,:]*pi_prime_l[k,:]

#         assert np.all(np.isclose(transition_matrix.sum(axis=0)[:self.K],1))

#         return transition_matrix
    