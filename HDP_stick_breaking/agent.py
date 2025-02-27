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
                 h=1000,
                 debug = False,
                 dec_temp = 1,
                 rho_g = 1,
                 rho_l = 1,     # global prior counts forgetting rate
                ):
        
        self.debug = debug
        self.max_context = max_context                                     # max number of contexts
        self.T = T                                                         # number of observations per episode
        self.K = K                                                         # current number of contexts
        self.TAU = TAU                                                     # number of episodes
        self.gamma = gamma                                                 # cluster opening tendency
        self.alpha = alpha                                                 # transitioning into a new cluster tendency
        self.kappa = kappa                                                 # self transition bias
        self.lambda_H = lambda_H                                           # parameters of base Measure H = Dir(lambda)

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

        self.h = h
        self.dec_temp = dec_temp
        self.rho_g = rho_g
        self.rho_l = rho_l


    def initialize_beliefs(self):

        self.prior_context = np.zeros(self.max_context)
        self.prior_context[0] = 1                                                         # context prior p(c1)

        self.beta_prime = np.zeros(self.max_context)                                      # Expectation of stick break beta'_k = gamma_k1/(gamma_k1+gamma_k2) 
        self.global_prior_counts = np.zeros([self.TAU+1, self.max_context,2])             # parameters gamma_1, gamma_2 of beta_k: p(beta'_k|gamma_k1, gamma_k2)
        self.global_prior = np.zeros(self.max_context)                                    # p(z|gamma_1, gamma_2)  = int_b p(z|b)p(b|gamma_1, gamma_2)
        self.global_prior[0] = 1

        self.transition_matrix_counts = np.zeros([self.max_context, self.max_context,2])  # parameters alpha_1, alpha_2 of stick break pi'_jk: p(pi_jk|alpha_jk1, alpha_jk2)
        self.transition_matrix = np.zeros([self.max_context, self.max_context])           # transition matrix Pi: p(c_t|c_t-1, alpha_1, alpha_2) with stick breaks integrated out
        self.transition_matrix[0,0] = 1

        self.prior_rewards_counts = np.zeros([self.TAU+1, self.nr, self.ns, self.max_context])
        self.prior_rewards_counts[0,:,:,0] = self.lambda_H                                  # int_phi Cat(x|phi)Dir(phi|lambda_H) = Cat(x|lambda_H)  
        self.prior_rewards = np.nan_to_num(self.prior_rewards_counts/self.prior_rewards_counts.sum(axis=1,keepdims=True))

        # self.forward_norms = np.zeros([self.TAU, self.T, self.T+1, self.npi, self.max_context])
        self.likelihood_policies = np.zeros([self.TAU, self.T, self.npi, self.max_context])
        self.posterior_policies = np.zeros([self.TAU, self.T, self.npi, self.max_context])

        self.prior_policies_counts = np.zeros([self.TAU+1, self.npi, self.max_context])
        self.prior_policies_counts[0,:,:self.K+1] = self.counts_prior_policies
        
        self.prior_policies = np.zeros([self.TAU+1, self.npi, self.max_context])

        self.context_likelihood = np.zeros([self.TAU, self.max_context])

        if self.approx_pred_pol:
            self.prior_policies[0] = self.digamma_approximation(self.prior_policies_counts[0])
        else:
            self.prior_policies[0] = self.prior_policies_counts[0] / self.prior_policies_counts[0].sum(axis=0)

        self.posterior_states = np.zeros([self.TAU,self.T, self.ns, self.T, self.npi, self.max_context])
        self.observations = np.zeros([self.TAU,self.T], dtype=int)              # array storing observations
        self.rewards = np.zeros([self.TAU,self.T], dtype=int)              # array storing observations

        self.context = np.zeros(self.TAU, dtype=int)                            # array storing inferred context
        self.posterior_context = np.zeros([self.TAU, self.T, self.max_context]) # array storing posterior over contextss
        self.actions = np.zeros([self.TAU, self.T])
        self.opened_new_context = np.zeros(self.TAU,dtype=bool)


    def ln(self, array):
        array[array==0] = 1e-20
        return np.log(array)


    def linear_ind(self, array):
        array = array[:,None].T if array.shape[-1] == 1 else array.T
        return np.ravel_multi_index(array, [self.na]*(self.T-1))


    def digamma_approximation(self, counts):
        return np.nan_to_num(softmax(digamma(counts) - digamma(counts.sum(axis=0)),axis=0))
          
 
    def initialize_states_messages(self,t,tau):

        # initialize messages for Bethe Approximation Belief Propagation
        self.fwd_messages = np.zeros([self.ns, self.T, self.npi, self.K+1]) + 1/self.ns
        self.fwd_messages[:,0,:,:] = self.prior_states[:,None,None]

        self.fwd_norms = np.zeros([self.T+1, self.npi, self.K+1])
        self.fwd_norms[0,:,:] = 1                               # accounts for the normalizing constant of the prior

        self.bwd_messages = np.zeros([self.ns, self.T, self.npi, self.K+1]) + 1/self.ns
        self.bwd_norms = np.zeros([self.T, self.npi, self.K+1])

        self.obs_messages = np.zeros((self.ns, self.T, self.npi, self.K+1)) + 1/self.ns

        self.reward_messages = np.zeros([self.ns, self.T, self.npi, self.K+1])
        rew_mess = np.einsum('r,rsc -> sc', self.utility, self.prior_rewards[tau,:,:,:self.K+1])
        rew_mess /= rew_mess.sum(axis=0)
        self.reward_messages[:] = rew_mess[:,None,None,:]

        # backward message intialization
        for c in range(self.K+1):
            for pi, policy in enumerate(self.policies):
                for t, u in zip(np.flip(np.arange(self.T-1)), np.flip(policy)):
                    self.bwd_messages[:,t,pi,c] = (self.bwd_messages[:,t+1,pi,c]*self.obs_messages[:,t+1,pi,c]*self.reward_messages[:,t+1,pi,c])\
                                                   .dot(self.state_transition_matrix[:,:,u])
                                                  
                    self.bwd_norms[t,pi,c] = self.bwd_messages[:,t,pi,c].sum()
                    self.bwd_messages[:,t,pi,c] /= self.bwd_norms[t,pi,c] 


    def update_states_messages(self,t,tau,pi,policy,c,reward,observation):
        
        # update rewards messages based on what was observed
        self.reward_messages[:,t,:,:] = self.prior_rewards[tau,reward,:,None,:self.K+1]
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
            
        for c in range(self.K+1):
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
        # self.forward_norms[tau,t] = self.fwd_norms
        self.posterior_states[tau,t,:,:,:,:self.K+1] = post

        return post


    def update_beliefs_policies(self,t,tau):
        
        likelihood = np.zeros([self.npi, self.max_context])
        likelihood[:,:self.K+1] = self.fwd_norms.prod(axis=0)
        posterior_policies  = np.power(likelihood,self.dec_temp)*self.prior_policies[tau]
        posterior_policies /= posterior_policies.sum(axis=0)
        posterior_policies = np.nan_to_num(posterior_policies)
        self.likelihood_policies[tau,t] = np.nan_to_num(likelihood/likelihood.sum(axis=0))
        self.posterior_policies[tau,t] = posterior_policies


        return likelihood, posterior_policies


    def update_beliefs_context_new(self, tau, t, posterior_policies, likelihood_policies):


        if tau == 0:
            prior_context = self.prior_context
        else:
            prior_context = self.transition_matrix.dot(self.posterior_context[tau-1,self.T-1])
        

        if t>0:

            alphas = self.prior_policies_counts[tau]
                       
            outcome_surprise =  (posterior_policies * self.ln(likelihood_policies)).sum(axis=0)
            policy_entropy   = -(posterior_policies * self.ln(posterior_policies)).sum(axis=0)
            policy_surprise  =  (posterior_policies * (digamma(alphas) - digamma(alphas.sum(axis=0)))).sum(axis=0)
            obs_messages = np.nan_to_num(outcome_surprise + policy_entropy + policy_surprise)
            obs_messages[:self.K+1] = self.ln(softmax(obs_messages[:self.K+1]))

        else:
            obs_messages = np.zeros(self.max_context)

        q_z = self.global_prior

        q_c = self.ln(prior_context) + q_z*obs_messages

        print("infering MARGINAL q(c_t) in LOG space") if tau % 100 == 0 else 0

        # if (np.argmax(q_z) == np.argmax(obs_messages) and t==1):
        #     if (np.argmax(q_z) != np.argmax(q_z*obs_messages)):
        # if t==self.T-1:   
        #     print(f"tau: {tau},t: {t}, observation: {self.observations[tau,t]}, reward: {self.rewards[tau,t]}")
        #     print(f"q_z       : {q_z.round(3)}")
        #     print(f"q_o       : {obs_messages.round(3)}")
        #     print(f"q_z*ln q_o: {q_z*obs_messages.round(3)}")
        #     print(f"prior     : {self.ln(prior_context).round(3)}")
        
        q_c[:self.K+1] = softmax(q_c[:self.K+1])
        q_c[self.K+1:] = np.exp(q_c[self.K+1:]).round(1)
        assert np.isclose(q_c.sum(),1)
        self.posterior_context[tau,t] = q_c
        
        return q_c, None

        # if t == 1:
        #     print(f"tau  : {tau}, observation: {self.observations[tau,t]}, reward: {self.rewards[tau,t]}")
        #     print(f"obs  :\n {obs_messages}")
        #     print(f"obs*z:\n {q_z*obs_messages}")
        #     test = self.ln(self.transition_matrix*prior_context[None,:])
        #     test[test < -30] = 0
        #     print(f"prior:\n{test}")

    def update_beliefs_context(self, tau, t, posterior_policies, likelihood_policies):

        # Infers q(c_t,c_{t-1}) via Bethe Approximation

        prior_context = self.prior_context if tau < 2 else self.transition_matrix.dot(self.posterior_context[tau-2,self.T-1])

        if t>0:

            alphas = self.prior_policies_counts[tau]
                       
            outcome_surprise =  (posterior_policies * self.ln(likelihood_policies)).sum(axis=0)
            policy_entropy   = -(posterior_policies * self.ln(posterior_policies)).sum(axis=0)
            policy_surprise  =  (posterior_policies * (digamma(alphas) - digamma(alphas.sum(axis=0)))).sum(axis=0)

            context_likelihood = np.nan_to_num(outcome_surprise + policy_entropy + policy_surprise)
            context_likelihood[:self.K+1] = self.ln(softmax(context_likelihood[:self.K+1]))
        else:
            context_likelihood = np.zeros(self.max_context)

        if t==self.T-1:
            self.context_likelihood[tau] = context_likelihood

        if tau == 0:
            obs_messages = np.array([context_likelihood, np.zeros(self.max_context)])
            q_z = np.array([self.global_prior, np.zeros(self.max_context)])
        else:
            obs_messages = np.array([self.context_likelihood[tau-1], context_likelihood])
            q_z = np.array([self.construct_G_0(self.global_prior_counts[tau-1]), self.global_prior])


        # print("infering JOINT q(c_t,c_{t-1} in LOG space") if tau % 100 == 0 else 0

        obs_messages = q_z*obs_messages

        q_c_joint = self.ln(self.transition_matrix*prior_context[None,:]) + obs_messages[0,:][None,:] + obs_messages[1,:][:,None]
        
        ind = self.K+1 if tau == 0 else self.K
        q_c_joint[:self.K+1, :ind] = softmax(q_c_joint[:self.K+1, :ind])
        q_c_joint[self.K+1:, :] = 0
        q_c_joint[:, ind:] = 0

        q_c = q_c_joint.sum(axis=1)
        assert np.isclose(q_c.sum(),1)
        
        self.posterior_context[tau,t] = q_c

        return q_c, q_c_joint


    def update_beliefs_context_old(self, tau, t, posterior_policies, likelihood_policies):


        if tau < 2:
            prior_context = self.prior_context
        else:
            prior_context = self.transition_matrix.dot(self.posterior_context[tau-2,self.T-1])
        

        if t>0:

            alphas = self.prior_policies_counts[tau]
                       
            outcome_surprise =  (posterior_policies * self.ln(likelihood_policies)).sum(axis=0)
            policy_entropy   = -(posterior_policies * self.ln(posterior_policies)).sum(axis=0)
            policy_surprise  =  (posterior_policies * (digamma(alphas) - digamma(alphas.sum(axis=0)))).sum(axis=0)
            context_likelihood = np.nan_to_num(outcome_surprise + policy_entropy + policy_surprise)
            context_likelihood[:self.K+1] = softmax(context_likelihood[:self.K+1])
        else:
            context_likelihood = np.zeros(self.max_context)
            context_likelihood[:self.K+1] = 1


        if t==self.T-1:
            self.context_likelihood[tau] = context_likelihood

        # Infers q(c_t,c_{t-1}) via Bethe Approximation

        if tau == 0:
            
            obs_messages = np.array([context_likelihood, np.ones(self.max_context)])
            q_z = np.array([self.global_prior, np.ones(self.max_context)])

        else:

            # [[p'(o_{t-1,1:W}| c_{t-1},z_{t-1})],
            #  [p'(o_{t,1:W}  | c_{t}  ,z_{t}  )]]
            obs_messages = np.array([self.context_likelihood[tau-1], context_likelihood])
            
            # q_z = \int_{beta} q(z|beta)q(beta)
            q_z = np.array([self.construct_G_0(self.global_prior_counts[tau-1]), self.global_prior])


        # sum_{z_{t}} q(z_t) log \prod_w p'(o_tw|c_t=k,z_t=k) = log \prod_w \phi'_{wk}^{q(z_t=k)}
        print("infering JOINT q(c_t,c_{t-1} in NORMAL space") if tau % 100 == 0 else 0
        obs_messages = obs_messages**q_z

        # q(c_t=j,c_t-1=k)
        q_c_joint = self.transition_matrix*prior_context[None,:]*obs_messages[0,:][None,:]*obs_messages[1,:][:,None]
        
        #  q(c_t) = sum_{c_{t-1}} q(c_t,c_{t-1})
        q_c_joint = (q_c_joint/q_c_joint.sum())
        q_c = q_c_joint.sum(axis=1)
        self.posterior_context[tau,t] = q_c

        assert np.isclose(q_c.sum(),1)
        
        return q_c, q_c_joint


    def update_beliefs(self, t, tau, state, reward, action, observation):
        
        ########## 1. Infer state q(s,r|\pi,c), policy q(\pi|c) and context q(c) posteriors (E-Step)?
        self.observations[tau,t] = observation
        self.rewards[tau,t] = reward

        q_s = self.update_beliefs_states(t, tau, reward, action, observation)
        likelihood_policies, posterior_policies = self.update_beliefs_policies(t,tau)
        q_c, q_c_joint = self.update_beliefs_context(tau, t, posterior_policies, likelihood_policies)


        if t == self.T-1:
            
            ########## 2. sample context and create new stick breaks and atoms if necessary 
            current_context = np.argmax(q_c)
        
            # argmax updates
            # q_c = np.eye(self.max_context)[current_context]
            # q_c_joint = np.zeros([self.max_context,self.max_context])
            # q_c_joint[current_context,self.context[tau-1]] = 1

            self.context[tau] = current_context

            if current_context + 1 > self.K:
                
                # print(f"\n\nopened new context at tau: {tau}")
                self.K += 1
                self.opened_new_context[tau] = True

                # add prior over new weight beta'_k
                self.global_prior_counts[tau, self.K-1] = [1,self.gamma]
                
                # add prior over new atom \phi_k
                self.prior_rewards_counts[tau,:,:,self.K] = self.lambda_H # + np.random.uniform(size = self.lambda_H.shape)*0.3

                # add prior over new atom \theta_k
                self.prior_policies_counts[tau,:,self.K] = self.h

                # add prior over new transition weights
                self.transition_matrix_counts[np.arange(self.K), [self.K-1]*(self.K), :] = [1,self.alpha]
                self.transition_matrix_counts[[self.K-1]*(self.K), np.arange(self.K), :] = [1,self.alpha]
                
                # add self-transition bias kappa      
                counts = np.zeros([self.max_context, self.max_context, 2])  
                counts[self.K-1, self.K-1, 0] = self.kappa
                counts[:self.K-1, self.K-1, 1] = self.kappa
                
                self.transition_matrix_counts = self.transition_matrix_counts + counts

            ########## 3. update parameter estimates (M-step?)
            
            # renormalizes probability after excluding new context possibility
            q_c[:self.K] /= q_c[:self.K].sum()
            q_c_joint[self.K,:] = 0
            q_c_joint /= q_c_joint.sum() 

            chosen_pol = np.argmax(posterior_policies[:,current_context])
            states = np.argmax(q_s[:,:,chosen_pol, current_context],axis=0)
            
            ### 3.1 update global context prior params q(beta'|gamma) and construct new q(z_t)
            counts = np.zeros([self.max_context,2])
            counts[:self.K,0] = q_c[:self.K]
            for k in range(self.K-1):
                counts[k,1] = q_c[k+1:self.K].sum()

            
            gamma_init = np.zeros([self.max_context,2])
            gamma_init[:self.K,:] = np.array([[1,self.gamma]])
            
            self.global_prior_counts[tau+1] = self.rho_g*self.global_prior_counts[tau] + counts + (1-self.rho_g)*gamma_init
            self.global_prior = self.construct_G_0(self.global_prior_counts[tau+1], approx=False)

            # DEBUG PRINTING
            # print(f"\ntau: {tau}, K: {self.K}")
            # print(f"q_c:\n{q_c}")
            # print(f"counts:\n{counts.T}")
            # if tau >0:
            #     print(f"global prior old:\n{self.global_prior_counts[tau].T}")
            # print(f"global prior:\n{self.global_prior_counts[tau+1].T}")

            ### 3.2 update reward probability params phi q(phi|lambda)
            self.prior_rewards_counts[tau+1] = self.prior_rewards_counts[tau].copy()
            for obs, state in zip(self.observations[tau,1:], states[1:]):
                self.prior_rewards_counts[tau+1, reward, state, :self.K] += q_c[:self.K]
            self.prior_rewards[tau+1] = np.nan_to_num(self.prior_rewards_counts[tau+1]/self.prior_rewards_counts[tau+1].sum(axis=0)[None,:,:])


            ### 3.3 update context transition probability params q(eta'|alpha) and construct p'(c_t, c_t-1)

            counts = np.zeros([self.max_context, self.max_context, 2])

            counts[:self.K,:self.K,0] = q_c_joint[:self.K, :self.K]

            for j in range(self.K):
                for k in range(self.K-1):
                    counts[k,j,1] = q_c_joint[k+1:self.K,j].sum()

            alpha_init = np.zeros([self.max_context,self.max_context,2])
            alpha_init[:self.K, :self.K,0] = 1
            alpha_init[np.arange(self.max_context), np.arange(self.max_context),0] *= self.kappa
            alpha_init[:self.K, :self.K,1] = self.alpha
            
            # print(f"\ntau: {tau}, K: {self.K}")
            # print(f"q_c_joint:\n{q_c_joint}")
            # print(f"counts:\n{counts[:,:,0]}\n{counts[:,:,1]}")
            # if tau >0:
            #     print(f"trans mat old:\n{self.transition_matrix_counts[:,:,0]}\n{self.transition_matrix_counts[:,:,1]}")
            
            self.transition_matrix_counts = self.rho_l*self.transition_matrix_counts + counts + (1-self.rho_l)*alpha_init
            # print(f"trans mat:\n{self.transition_matrix_counts[:,:,0]}\n{self.transition_matrix_counts[:,:,1]}")

            self.transition_matrix = self.construct_G_j(approx=False)
            
            ### 3.4 update context specific policy prior params q(\theta|epsilon)

            counts = self.prior_policies_counts[tau,:,:].copy()
            counts[chosen_pol,:self.K] += q_c[:self.K]
            self.prior_policies_counts[tau+1] = counts
            self.prior_policies[tau+1] = np.nan_to_num(counts / counts.sum(axis=0))

        ######### Print inferred beliefs
        if self.debug:
            if tau < 40:
                if self.opened_new_context[tau]:
                    self.K -= 1
                print(f"--------------------\ntau,t: {tau,t}")
                print(f"action: {action}, observation: {observation}, reward: {reward}")
                # print(f"\nq_s for policy:")
                
                # for k in range(self.K+1):
                #     print(q_s[:,:,0,k])
                #     print(q_s[:,:,1,k])

                print(f"\nq(r|pi,c); policy likelihood:")
                print(likelihood_policies.round(4))

                print(f"\nq(pi|c) policy posterior:")
                print(posterior_policies.round(4))
                
                print(f"\nq(c):")
                print(q_c)

                print(f"\nq_c_joint prior")
                print(q_c_joint.round(5))

                if t == self.T-1:
                    print(f"\nchosen context:")
                    print(current_context)
                    
                    if self.opened_new_context[tau]:
                        print("opened new context!")

                    print(f"\nglobal prior counts")
                    print(self.global_prior_counts[tau+1].T)

                    print(f"\nglobal prior")
                    print(self.global_prior.round(3))



                    print(f"\nrewards counts:")
                    print(f"obs, reward: {observation, reward}")
                    for k in range(self.K+1):
                        print(f"\n{self.prior_rewards_counts[tau+1,:,:,k]}")
                    
                    print(f"prior_rewards")
                    for k in range(self.K+1):
                        print(self.prior_rewards[tau+1,:,:,k].round(3))

                    print(f"\ntransition matrix counts")
                    print(f"contexts:{self.context[tau-1], self.context[tau]}")

                    print("\n")
                    print(self.transition_matrix_counts[:,:,0])
                    print("\n")
                    print(self.transition_matrix_counts[:,:,1])
                    
                    print(f"\ntransition matrix")
                    print(self.transition_matrix)
                    

                    print(f"\npolicy counts")
                    print(f"chosen policy:{chosen_pol}")
                    print(self.prior_policies_counts[tau+1])
                    print(self.prior_policies[tau+1].round(4))
                
                if self.opened_new_context[tau]:
                    self.K = self.K+1


    def construct_G_0(self, global_prior_counts, approx=False):

        global_prior = np.zeros(self.max_context)
        
        if not approx:
            beta_prime = np.nan_to_num(global_prior_counts/global_prior_counts.sum(axis=1)[:,None])  # expected b_k'            
            beta_prime_l = np.insert(np.cumprod(beta_prime[:,1]),0,1)                                   #  prod_l=1^k-1 (1-beta'_l)
            beta_prime_k = np.insert(beta_prime[:,0], self.K, 1)

            for k in range(self.K+1):
                global_prior[k] = beta_prime_k[k]*beta_prime_l[k]

            # print(self.K, global_prior, global_prior.sum())

        else:
        
            beta_prime = digamma(global_prior_counts)
            beta_prime_l = np.insert(np.cumsum(beta_prime[:,1]),0,0)
            beta_prime_k = np.insert(beta_prime[:,0],self.K,0)
            
            # print(digamma(global_prior_counts.sum(axis=1))) 
            norm = np.cumsum(digamma(global_prior_counts.sum(axis=1)))
            norm = np.insert(norm, self.K, norm[self.K-1])
            
            for k in range(self.K+1):
                global_prior[k] = beta_prime_k[k] + beta_prime_l[k] - norm[k]
            
            global_prior[:self.K+1] = softmax(global_prior[:self.K+1])
            
        # print(f"global_prior:\n {global_prior_counts}, {global_prior}")

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


    def sample_action(self,t,tau):

        post_policies = self.posterior_policies[tau,t]
        post_policies = post_policies.dot(self.posterior_context[tau,t])
        # chosen_action = self.policies[np.argmax(post_policies)][t]
        
        post_actions = np.zeros(self.na)
        for a in range(self.na):
            post_actions[a] = post_policies[self.policies[:,t] == a].sum()

        chosen_action = np.random.choice(np.arange(self.na), p=post_actions)
        self.actions[tau,t] = chosen_action
        
        return chosen_action

class HDP_speaker_discretization():
    
    def __init__(self):
        pass
    

    def initialize_HDP(self, lambda_H=np.ones(10), TAU=10, gamma=2, alpha=1, kappa=0, K=0, max_context=6,debug=False, obs_batch_size=1):
        
        self.max_context = max_context                                  # max number of contexts
        self.W = obs_batch_size                                         # how many observations passed in a single go
        self.K = K                                                      # current number of contexts
        self.TAU = TAU                                                  # number of episodes
        self.gamma = gamma                                              # cluster opening tendency
        self.alpha = alpha                                              # transitioning into a new cluster tendency
        self.kappa = kappa                                              # self transition bias
        self.rho = kappa/(alpha+kappa)
        self.lambda_H = lambda_H                                        # parameters of base Measure H = Dir(lambda)
        self.observations  = np.zeros([self.TAU,self.W], dtype=int)           # array storing observations
        self.context = np.zeros(TAU//obs_batch_size, dtype=int)                         # array storing inferred context
        self.posterior_context = np.zeros([self.TAU//obs_batch_size, self.max_context]) # array storing posterior over contextss
        self.debug=debug


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

        self.generative_model_counts = np.zeros([self.lambda_H.size,self.max_context])
        self.generative_model_counts[:,0] = self.lambda_H                                # int_phi Cat(x|phi)Dir(phi|lambda_H) = Cat(x|lambda_H)  
        self.generative_model_obs = np.nan_to_num(self.generative_model_counts/self.generative_model_counts.sum(axis=0))

        self.opened_new_context = np.zeros(self.TAU//self.W)

    def update_beliefs_context(self,tau):

        # Infers q(c_t,c_{t-1}) via Bethe Approximation

        if tau == 0:
            
            obs_messages = np.array([
                             self.generative_model_obs[self.observations[tau]].prod(axis=0),\
                             np.ones(self.max_context)\
                           ])

            q_z = np.array([self.global_prior, np.ones(self.max_context)])

        else:

            # [p'(o_{t-1,1:W}| c_{t-1},z_{t-1})],
            # [p'(o_{t,1:W}  | c_{t}  ,z_{t}  )]
            obs_messages = np.array([
                             self.generative_model_obs[self.observations[tau-1]].prod(axis=0),\
                             self.generative_model_obs[self.observations[tau]].prod(axis=0)\
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

        self.posterior_context[tau] = q_c

        assert np.isclose(q_c.sum(),1)
        
        return q_c


    def update_beliefs(self, observation,tau):

        # print(f"---------\ntau={tau}")
        self.observations[tau] = observation

        if tau == 0:
            self.initialize_beliefs()
        
        q_c = self.update_beliefs_context(tau)
        
        current_context = np.argmax(q_c)
        self.context[tau] = current_context

        if current_context + 1 > self.K:         # if inferred presence of new context
            # print(f"opened new context at tau: {tau}")
            self.K += 1
            self.opened_new_context[tau] = 1

            # add prior over new weight beta'_k
            self.global_prior_counts[self.K-1] = [1,self.gamma]
            
            # add prior over new atom \phi_k
            self.generative_model_counts[:,self.K] = self.lambda_H

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
