#%%
import numpy as np
from scipy.special import digamma, softmax
import seaborn as sns
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

import warnings
warnings.filterwarnings("ignore")


class HDP_IMM(): 


    def __init__(self,
                 lambda_H=np.ones(10),                   # parameters of base Measure H = Dir(lambda) for novel context
                 TAU=10,                                 # number of episodes
                 T=1,                                    # length of episode
                 gamma=2,                                # global prior cluster opening tendency
                 alpha=1,                                # local prior cluster opening tendency
                 kappa=0,                                # self transition bias  
                 K=1,                                    # initial number of contexts
                 max_context=7,                          # max number of contexts
                 state_transition_matrix = None,         # state transition model p(s'|s,a)
                 observation_generation_matrix = None,   # observation model p(o|s)
                 utility = None,                         # reward utility p(R=1|o)
                 policies = None,                        # set of possible policies
                #  prior_rewards = None,
                 counts_prior_rewards = None,            # Dirichlet counts lambda of p(r|s,c, lambda)
                 prior_policies = None,                  # p(r|s,c, lambda)p(lambda) = int_lambda p(r|s,c,lambda)p(lambda)
                 counts_prior_policies = None,           # Dirichlet counts h of p(pi|c,h)
                 prior_states = None,                    # prior over initial states p(s_1)
                 na = None,                              # number of actions   
                 env = None,                             # task environment class
                 approx_pred_pol = None,                 # whether to use digamma approximation for policies
                 approx_pred_rew = None,                 # whether to use digamma approximation for rewards
                 h=1000,                                 # habitual/automatization tendency initial counts; h_0 in paper
                 debug = False,                          # when true prints inferred beliefs
                 dec_temp = 1,                           # policy selection decision temperature
                 rho_g = 1,                              # forgetting rate global prior counts
                 rho_l = 1,                              # forgetting rate local prior counts
                 gamma_init = 1000,                      # initial value of prior parameter gamma_k1 for variable beta_k
                 template_context_contingencies=None,    # possible template context contingencies
                 context_observation_counts = None,      # Dirichlet counts rho of p(d|c,rho), where d is a context observation and c a possible context.
                 use_context_obs = False,                # adds prediction error from context observation to free energy when context observation d present
                 cap = 10000,                            # maximum value of Dir counts gamma_k1 and gamma_k2
                 use_template=False                      # whether to use template context contingencies for new contexts
                ):
        
        self.debug = debug                                                      # print beliefs when true
        self.max_context = max_context                                          # max number of contexts
        self.TAU = TAU                                                          # number of episodes
        self.T = T                                                              # episode length
        self.K = K                                                              # initial number of contexts
        self.gamma = gamma                                                      # global prior cluster opening tendency
        self.alpha = alpha                                                      # local prior cluster opening tendency
        self.kappa = kappa                                                      # self transition bias
        self.lambda_H = lambda_H                                                # parameters of base Measure H = Dir(lambda) for novel context
        self.init_reward_counts = counts_prior_rewards                          # initial counts for p(r|s,c, lambda) if simulation initialized with K!=0
        self.na = na                                                            # number of actions
        self.nr = utility.size                                                  # number of rewards
        self.ns = prior_states.size                                             # number of states
        self.npi = prior_policies.shape[0]                                      # number of policies
        self.env = env                                                          # task environment class
        self.state_transition_matrix = state_transition_matrix                  # state transition model p(s'|s,a)
        self.observation_generation_matrix = observation_generation_matrix      # observation model p(o|s)
        self.policies = policies                                                # set of possible policies
        self.prior_policies = prior_policies                                    # prior over policies p(pi|c) = int_pi p(pi|theta)p(theta|h)
        self.counts_prior_policies = counts_prior_policies                      # Dirichlet counts h of p(pi|c,h)
        self.utility = utility                                                  # reward utility p(R=1|o)
        self.use_context_obs = use_context_obs                                  # adds prediction error from context observation to free energy when context observation d present
        self.prior_states = prior_states                                        # prior over initial states p(s_1)
        self.approx_pred_pol = approx_pred_pol                                  # whether to use digamma approximation for policies
        self.approx_pred_rew = approx_pred_rew                                  # whether to use digamma approximation for rewards
        self.h = h                                                              # habitual/automatization tendency initial counts; h_0 in paper
        self.dec_temp = dec_temp                                                # policy selection decision temperature
        self.rho_g = rho_g                                                      # forgetting rate global prior counts
        self.rho_l = rho_l                                                      # forgetting rate local prior counts
        self.gamma_init = gamma_init                                            # initial value of prior parameter gamma_k1 for variable beta_k
        self.template_context_contintengcies = template_context_contingencies   # possible template context contingencies
        self.context_observation_counts = context_observation_counts            # Dirichlet counts rho of p(d|c,rho), where d is a context observation and c a possible context.
        self.nco = context_observation_counts.shape[0]                          # number of context observations
        self.cap = cap                                                          # maximum value of Dir counts gamma_k1 and gamma_k2
        self.use_template = use_template                                        # whether to use template context contingencies for new contexts
        # self.duplicates = []
        if not template_context_contingencies is None:
            self.template_copy = template_context_contingencies.copy()


    def initialize_beliefs(self):

        self.prior_context = np.zeros(self.max_context)
        self.prior_context[0] = 1                                                           # context prior p(c1)

        self.global_prior_counts = np.zeros([self.TAU+1, self.max_context])                 # parameters gamma_1, gamma_2 of beta_k: p(beta'_k|gamma_k1, gamma_k2)
        self.global_prior_counts[0,:self.K+1] = [self.gamma_init]*self.K + [self.gamma]
        self.global_prior = np.zeros(self.max_context)                                      # p(z|gamma_1, gamma_2)  = int_b p(z|b)p(b|gamma_1, gamma_2)
        self.global_prior[0] = 1

        self.transition_matrix_counts = np.zeros([self.max_context, self.max_context])      # parameters alpha_1, alpha_2 of stick break pi'_jk: p(pi_jk|alpha_jk1, alpha_jk2)
        self.transition_matrix_counts[:self.K+1,:self.K+1] = 1
        for c in range(self.K):
            self.transition_matrix_counts[c,c] = self.kappa + 1
            self.transition_matrix_counts[self.K,c] = self.alpha
            
        self.transition_matrix_counts[:self.K,self.K] = 1
        self.transition_matrix_counts[self.K,self.K] = 11
        
        self.transition_matrix = self.digamma_approximation(self.transition_matrix_counts)
        
        self.transition_matrix_log = np.zeros([self.TAU+1, self.max_context, self.max_context])
        self.transition_matrix_log[0] = self.transition_matrix_counts.copy()
        

        self.prior_context_observation_counts = np.zeros([self.TAU+1, self.nco, self.max_context])
        self.prior_context_observation_counts[0,:,:self.K+1] = self.context_observation_counts

        self.prior_context_observation = np.zeros([self.TAU+1, self.nco, self.max_context])
        self.prior_context_observation[0,:,:self.K+1] = self.digamma_approximation(self.context_observation_counts)


        self.prior_rewards_counts = np.zeros([self.TAU+1, self.nr, self.ns, self.max_context])
        self.prior_rewards_counts[0,:,:,:self.K] = self.init_reward_counts
        
        self.prior_rewards_counts[0,:,:,self.K] = self.lambda_H

        self.prior_rewards = np.nan_to_num(self.prior_rewards_counts/self.prior_rewards_counts.sum(axis=1,keepdims=True))

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
        self.observations = np.zeros([self.TAU,self.T], dtype=int)
        self.rewards = np.zeros([self.TAU,self.T], dtype=int)
        self.context_obs = np.zeros([self.TAU,self.T],dtype=int)
        self.context = np.zeros(self.TAU, dtype=int)
        self.posterior_context = np.zeros([self.TAU, self.T, self.max_context])
        self.posterior_context_joint = np.zeros([self.TAU, self.T, self.max_context, self.max_context])
        
        self.actions = np.zeros([self.TAU, self.T])
        self.opened_new_context = np.zeros(self.TAU+1,dtype=bool)

        if self.debug:
            print("---------   INITIAL BELIEFS -------------")
            print(f"\n{self.K} contexts")
            print(f"\nglobal prior: {self.global_prior_counts[0].round(3)}")
            
            print(f"\ntransition matrix")
            print(self.transition_matrix_counts.round(3))

            print(f"\nprior rewards")
            for k in range(self.K+1):
                print(self.prior_rewards_counts[0,:,:,k].round(3))


    def ln(self, array):
        return np.log(array+1e-20)


    def linear_ind(self, array):
        array = array[:,None].T if array.shape[-1] == 1 else array.T
        return np.ravel_multi_index(array, [self.na]*(self.T-1))


    def digamma_approximation(self, counts, normalize=True):
        if normalize:
            matrix = np.nan_to_num(softmax(digamma(counts) - digamma(counts.sum(axis=0)),axis=0))
        else:
            matrix = np.nan_to_num(digamma(counts) - digamma(counts.sum(axis=0)))

        return matrix 


    def initialize_states_messages(self,t,tau):

        # initialize messages for Bethe Approximation Belief Propagation
        self.fwd_messages = np.zeros([self.ns, self.T, self.npi, self.K+1]) + 1/self.ns
        self.fwd_messages[:,0,:,:] = self.prior_states[:,None,None]

        self.fwd_norms = np.zeros([self.T+1, self.npi, self.K+1])
        self.fwd_norms[0,:,:] = 1

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


    # compute q(s_{t+1:T},r_{t+1:T}|pi, c_{tau}, z_{tau}); refer to (Schwoebel et al., 2021) for details
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

    # compute q(pi|c_{tau}, z_{tau}) as per Eq. 8
    def update_beliefs_policies(self,t,tau):
        
        likelihood = np.zeros([self.npi, self.max_context])
        likelihood[:,:self.K+1] = self.fwd_norms.prod(axis=0)
        posterior_policies  = np.power(likelihood,self.dec_temp)*self.prior_policies[tau]
        posterior_policies /= posterior_policies.sum(axis=0)
        posterior_policies = np.nan_to_num(posterior_policies)
        self.likelihood_policies[tau,t] = np.nan_to_num(likelihood/likelihood.sum(axis=0))
        self.posterior_policies[tau,t] = posterior_policies


        return likelihood, posterior_policies

    # compute q(c_{tau+1}, c_{tau}) as per Eq. 10
    def update_beliefs_context(self, tau, t, posterior_policies, likelihood_policies, context_obs):
      

        prior_context = self.prior_context if tau < 2 else self.transition_matrix.dot(self.posterior_context[tau-2,self.T-1])

        if t>0:
            alphas = self.prior_policies_counts[tau]
            outcome_surprise =  (posterior_policies * self.ln(likelihood_policies)).sum(axis=0)
            policy_entropy   = -(posterior_policies * self.ln(posterior_policies)).sum(axis=0)
            policy_surprise  =  (posterior_policies * (digamma(alphas) - digamma(alphas.sum(axis=0)))).sum(axis=0)
            obs_surprise     = self.ln(self.prior_context_observation[tau,context_obs])

            context_likelihood = np.nan_to_num(outcome_surprise + policy_entropy + policy_surprise + self.use_context_obs*obs_surprise)
            context_likelihood[:self.K+1] = self.ln(softmax(context_likelihood[:self.K+1]))

        
        else:
            context_likelihood = np.zeros(self.max_context)

        if t==self.T-1:
            self.context_likelihood[tau] = context_likelihood

        if tau == 0:
            obs_messages = np.array([context_likelihood, np.zeros(self.max_context)])

            q_z = np.array([self.global_prior, np.zeros(self. max_context)])
        else:
            obs_messages = np.array([self.context_likelihood[tau-1], context_likelihood])
            q_z = np.array([self.digamma_approximation(self.global_prior_counts[tau-1]), self.global_prior])


        obs_messages = q_z*obs_messages
  
        q_c_joint = self.ln(self.transition_matrix*prior_context[None,:]) + obs_messages[0,:][None,:] + obs_messages[1,:][:,None]
        q_c_joint[:self.K+1, :self.K+1] = softmax(q_c_joint[:self.K+1, :self.K+1])
        q_c_joint[self.K+1:, :] = 0
        q_c_joint[:, self.K+1:] = 0

        q_c = q_c_joint.sum(axis=1)
        assert np.isclose(q_c.sum(),1)

        self.posterior_context[tau,t] = q_c
        self.posterior_context_joint[tau,t,:,:] = q_c_joint

        return q_c, q_c_joint


    ###### Implementation of Algorithm 1. in paper
    def update_beliefs(self, t, tau, state, reward, action, observation, context_obs):
        
        ########## Infer state/rewards q(s_{t+1:T},r_{t+1:T}|pi, c_{tau}, z_{tau}), policy q(pi|c_{tau}, z_{tau}) and context q(c_{tau}) posteriors
        self.observations[tau,t] = observation
        self.rewards[tau,t] = reward
        self.context_obs[tau,t] = context_obs

        q_s = self.update_beliefs_states(t, tau, reward, action, observation)
        likelihood_policies, posterior_policies = self.update_beliefs_policies(t,tau)
        q_c, q_c_joint = self.update_beliefs_context(tau, t, posterior_policies, likelihood_policies, context_obs)


        if t == self.T-1:
            
            ########## 2. At end of an episode tau, sample a context and open a new one if necessary; Algorithm 1. Line 5-8
            if q_c[self.K] >= 0.5:
                current_context = self.K
            else: 
                current_context = np.argmax(q_c[:self.K])

            self.context[tau] = current_context

            
            q_c = np.eye(self.max_context)[current_context]
            q_c_joint = np.zeros([self.max_context,self.max_context])
            q_c_joint[current_context,self.context[tau-1]] = 1

            
            # open new context Algorithm 1. Line 12-17
            if current_context + 1 > self.K:
                    
                self.K += 1
                self.opened_new_context[tau+1] = True

                # initialize prior over new global context probability beta_k
                self.global_prior_counts[tau, self.K-1:self.K+1] = [self.gamma_init, self.gamma]
            
                if self.use_template:
                    # initializa prior over reward probabilities lambda
                    self.prior_rewards_counts[tau,:,:,self.K] = self.lambda_H
                    chosen_template = np.argmax(self.template_context_contintengcies[reward,observation])
                    self.prior_rewards_counts[tau,:,:,self.K-1] = self.template_context_contintengcies[:,:,chosen_template]
                    self.template_context_contintengcies = np.delete(self.template_context_contintengcies,chosen_template,axis=-1)
                    if self.template_context_contintengcies.size == 0:
                        self.template_context_contintengcies = self.template_copy.copy()

                else:
                    self.prior_rewards_counts[tau,:,:,self.K] = self.lambda_H 
                    self.prior_rewards_counts[tau,:,:,self.K-1] = self.lambda_H + np.random.uniform(size = self.lambda_H.shape)


                # initialize prior over context-specific policy repetition tendency 
                self.prior_policies_counts[tau,:,self.K] = self.h

                # add prior over new context transition weights
                self.transition_matrix_counts[self.K-1, :self.K] = 1
                self.transition_matrix_counts[:self.K, self.K-1] = 1
                self.transition_matrix_counts[self.K-1, self.K-1] += self.kappa
                self.transition_matrix_counts[self.K,:self.K] = self.alpha
    
                
                self.transition_matrix_counts[np.arange(self.K+1),self.K] = 1
                self.transition_matrix_counts[self.K, self.K] = 11

                self.prior_context_observation_counts[tau,:,self.K] = 1

            ########## 3. Update parameter distributions - Algorithm. 1 Line 19-20 
            
            # renormalizes probability after excluding new context possibility
            q_c[self.K:] = 0
            q_c /= q_c.sum()
            assert (np.isclose(q_c.sum(),1))

            q_c_joint[self.K,:] = 0
            q_c_joint /= q_c_joint.sum() 

            chosen_pol = np.argmax(posterior_policies[:,current_context])
            states = np.argmax(q_s[:,:,chosen_pol, current_context],axis=0)
            
            ### 3.1 update global context prior params q(beta|gamma) and construct new q(z_t)

            beta_init = self.global_prior_counts[tau].copy()
            beta_init[:self.K] = self.gamma_init
            counts = self.rho_g*(self.global_prior_counts[tau]) + q_c + (1-self.rho_g)*beta_init
            counts[counts>self.cap] = self.cap
            self.global_prior_counts[tau+1] = counts
            self.global_prior = self.digamma_approximation(self.global_prior_counts[tau+1])

            ### 3.2 update reward probability params phi q(phi|lambda)

            self.prior_rewards_counts[tau+1] = self.prior_rewards_counts[tau].copy()
            for reward, state in zip(self.rewards[tau,1:], states[1:]):
                self.prior_rewards_counts[tau+1, reward, state, :self.K] += q_c[:self.K]
            
            if self.approx_pred_rew:
                self.prior_rewards[tau+1] = self.digamma_approximation(self.prior_rewards_counts[tau+1])
            else:
                self.prior_rewards[tau+1] = np.nan_to_num(self.prior_rewards_counts[tau+1]/self.prior_rewards_counts[tau+1].sum(axis=0)[None,:,:])

            ### 3.3 update context transition probability params q(eta|alpha) and construct p'(c_t,c_t-1)
            
            alpha_init = np.ones([self.max_context, self.max_context]) + np.eye(self.max_context)*self.kappa
            alpha_init[self.K,:] = self.alpha
            alpha_init[self.K+1:,:] = 0
            alpha_init[:,self.K:] = 0
            self.transition_matrix_counts = self.rho_l*self.transition_matrix_counts + q_c_joint + (1-self.rho_l)*alpha_init
            self.transition_matrix = self.digamma_approximation(self.transition_matrix_counts)
            self.transition_matrix_log[tau+1] = self.transition_matrix_counts.copy()
            
            
            ### 3.4 update context specific policy prior params q(theta|omega)

            counts = self.prior_policies_counts[tau,:,:].copy()
            counts[chosen_pol,:self.K] += q_c[:self.K]
            self.prior_policies_counts[tau+1] = counts

            if self.approx_pred_pol:
                self.prior_policies[tau+1] = self.digamma_approximation(counts)
            else:
                self.prior_policies[tau+1] = np.nan_to_num(counts / counts.sum(axis=0))


            ### 3.4 update context observation counts - not used in simulations

            self.prior_context_observation_counts[tau+1] = self.prior_context_observation_counts[tau].copy() 
            self.prior_context_observation_counts[tau+1,context_obs,:] += q_c
            self.prior_context_observation[tau+1] = self.digamma_approximation(self.prior_context_observation_counts[tau+1])
                
        ######### Print inferred beliefs
        
        if self.debug:
            if tau > 1: #400 and tau < 420 and t == 4:
                if self.opened_new_context[tau+1]:
                    self.K -= 1
                print(f"--------------------\ntau,t: {tau,t}")
                print(f"action: {action}, observation: {observation}, reward: {reward}")


                # print(f"\nq(R|pi,c); policy likelihood:")
                # print(likelihood_policies.round(3))

                # print(f"q(pi|c) policy posterior:")
                # print(posterior_policies.round(3))
                
                print(f"\nq_c (renormalised):")
                print(q_c.round(3))

                print(f"\ninferred q_c_joint (renormalized?)")
                print(q_c_joint.round(3))

                if t == self.T-1:
                    print(f"\nchosen context: {current_context}, opened new: {self.opened_new_context[tau+1]}")

                    print(f"\nrewards counts:")
                    print(f"obs, reward: {observation, reward}")
                    for k in range(self.K+1):
                        print(f"\n{self.prior_rewards_counts[tau+1,:,:,k]}")
                    
                    print(f"prior_rewards")
                    for k in range(self.K+1):
                        print(self.prior_rewards[tau+1,:,:,k].round(3))

                    print(f"\nglobal prior counts")
                    print(self.global_prior_counts[tau+1].T)

                    print(f"\nglobal prior")
                    print(self.global_prior.round(5))


                    print(f"\ntransition matrix")
                    print(f"contexts:{self.context[tau-1], self.context[tau]}")
                    print(self.transition_matrix_counts.round(3))
                    print(self.transition_matrix.round(3))
                
                if self.opened_new_context[tau+1]:
                    self.K = self.K+1

                a = 0
                
    
    def sample_action(self,t,tau):

        post_policies = self.posterior_policies[tau,t]
        post_cont = self.posterior_context[tau,t].copy()

        if tau >= 1:
            post_cont[:self.K] /= post_cont[:self.K].sum()
            post_cont[self.K] = 0

        post_policies = post_policies.dot(post_cont)
        # print(tau,post_policies)
        
        post_actions = np.zeros(self.na)
        for a in range(self.na):
            post_actions[a] = post_policies[self.policies[:,t] == a].sum()
        
        assert (np.isclose(post_actions.sum(), 1))
        post_actions /= post_actions.sum()
        chosen_action = np.random.choice(np.arange(self.na), p=post_actions)
        self.actions[tau,t] = chosen_action
        
        return chosen_action

