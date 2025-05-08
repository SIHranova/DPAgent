import numpy as np
import scipy.special as scp

class HibachiGrillPerception():

    def __init__(self,
                 state_transition_matrix,              # p(s_t|s_t-1)
                 context_transition_matrix,            # p(c_t|c_t-1)
                 observation_generation_matrix,        # p(o_t|s_t)
                 utility,                              # pre-given desirability of observations
                 policies,                             # all possible policies given environment setup
                 prior_rewards,                        # p(s_t|s_t-1)
                 counts_prior_rewards,                 # hyperparameters beta of p(phi;beta)
                 prior_policies,                       # p(pi|theta)
                 counts_prior_policies,                # parameters alpha of p(theta;alpha)
                 prior_states,                         # initial p(s|c)
                 counts_prior_context,                 # hyperparameters gamma p(eta;gamma); symmetric for known context and gamma_init for trailing context dimension
                 prior_context,                        # initial p(c)
                 counts_prior_bundle,                  # initial hyperparameters for kappa p(w_t; kappa)
                 na,                                   #
                 nc,                                   #
                 env,                                  #
                 approx_pred_pol = True,               # use digamma approx when updating policy prior p(pi|c)
                 approx_pred_rew = True,               # use digamma approx when updating reward posterir p(r|s,c)
                 gamma_init = 0.2,                     # concentration parameter for Dirichlet Process
                 rho = 1                               # context forgetting rate
                ):
        
 
        self.state_transition_matrix = state_transition_matrix
        self.observation_generation_matrix = observation_generation_matrix
        self.context_transition_matrix = context_transition_matrix
        self.utility = utility                                        
        self.policies = policies
        self.prior_states = prior_states                           
        self.approx_pred_pol = approx_pred_pol
        self.approx_pred_rew = approx_pred_rew
        self.na = na
        self.nc = nc
        self.k = nc - 1                                # number of currently inferred context; should be 1 unless we initialize agent with knowledge of more context 
        self.gamma_init = gamma_init                   # gamma_init is concentration parameter for Dirichlet process
        
        #inherited from other classes
        self.environment = env
        self.TAU = env.TAU
        self.T = env.T
        self.nr = env.nr
        self.nc = nc
        self.ns = env.ns
        self.rewards = env.rewards

        # derived assignments
        self.npi = policies.shape[0]
        self.possible_policies = self.policies.copy()
        self.possible_policies_ind = np.arange(self.policies.shape[0])                                 

        # belief update logs
        self.inferred_new_context = np.full(self.TAU,False)
        self.posterior_states = np.zeros([self.TAU,self.T, self.ns, self.T, self.npi, self.nc])

        self.prior_policies = np.zeros([self.TAU, self.T, self.npi, self.nc])
        self.prior_policies[0,:] = prior_policies

        self.alpha_policy_counts = np.zeros([self.TAU, self.T, self.npi, self.nc])
        self.alpha_policy_counts[0,:] = counts_prior_policies[None,:,:]
        self.h =  1/np.unique(counts_prior_policies)[0]
        assert(np.unique(counts_prior_policies).size == 1)                                      # assumes all policies initialized the same!
        
        self.prior_rewards = np.zeros([self.TAU, self.T, self.nr, self.ns, self.nc])
        self.prior_rewards[0] = prior_rewards[None,:,:,:] 

        self.beta_reward_counts = np.zeros([self.TAU, self.T, self.nr, self.ns, self.nc])
        self.beta_reward_counts[0] = counts_prior_rewards[None,:,:,:]

        self.forward_norms = np.zeros([self.TAU, self.T, self.T+1, self.npi, self.nc])
        self.likelihood_policies = np.zeros([self.TAU, self.T, self.npi, self.nc])
        self.posterior_policies = np.zeros([self.TAU, self.T, self.npi, self.nc])
        
        self.prior_context = np.zeros([self.TAU, self.T, self.nc])
        self.prior_context[0,:] = prior_context[None,:]

        self.gamma_context_counts = np.zeros([self.TAU, self.T, self.nc])
        self.gamma_context_counts[0,:] = counts_prior_context[None,:]

        self.posterior_context = np.zeros([self.TAU, self.T, self.nc])
        # self.posterior_context[0] = 1

        self.prior_bundle = np.zeros([self.TAU,self.T,2])
        self.prior_bundle[0,:] = (counts_prior_bundle/counts_prior_bundle.sum())[None,:]
        
        self.epsilon_bundle_counts = np.zeros([self.TAU, self.T,2])
        self.epsilon_bundle_counts[0,:] = counts_prior_bundle[None,:]

        self.posterior_bundle = np.zeros([self.TAU, self.T, 2])
        # self.posterior_bundle[0] = 1


    def ln(self, array):
        array[array==0] = 1e-20
        return np.log(array)


    def linear_ind(self, array):
        array = array[:,None].T if array.shape[-1] == 1 else array.T # bad coding :D
        return np.ravel_multi_index(array, [self.na]*(self.T-1))


    def expand_dimension(self, array):
        empty_dimension = np.empty(array.shape[:-1])
        empty_dimension[:] = np.nan
        array = np.append(array, empty_dimension[...,None], axis=-1)
        return array


    def open_new_context(self):
        
        self.forward_norms = self.expand_dimension(self.forward_norms)
        self.posterior_states = self.expand_dimension(self.posterior_states)

        self.likelihood_policies = self.expand_dimension(self.likelihood_policies)
        self.posterior_policies = self.expand_dimension(self.posterior_policies)

        self.posterior_context = self.expand_dimension(self.posterior_context)

        self.alpha_policy_counts = self.expand_dimension(self.alpha_policy_counts)
        self.prior_policies = self.expand_dimension(self.prior_policies)
        
        self.gamma_context_counts = self.expand_dimension(self.gamma_context_counts)
        self.prior_context = self.expand_dimension(self.prior_context)

        self.beta_reward_counts = self.expand_dimension(self.beta_reward_counts)
        self.prior_rewards = self.expand_dimension(self.prior_rewards)


    def digamma_approximation(self, counts):
        return scp.softmax(scp.digamma(counts) - scp.digamma(counts.sum(axis=0)),axis=0)
    
                                
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
    

    def update_beliefs_policies(self,t,tau):
        
        likelihood = self.fwd_norms.prod(axis=0)                      # exp(log(norms)) = -F(pi,c)
        posterior_policies  = likelihood*self.prior_policies[tau,t]   # exp(digamma(alpha_ij) - digamma(alpha_j)) when you integrate theta out
        posterior_policies /= posterior_policies.sum(axis=0)
        
        # store in global log
        self.likelihood_policies[tau,t] = likelihood/likelihood.sum(axis=0)
        self.posterior_policies[tau,t] = posterior_policies

        return likelihood, posterior_policies
    

    def update_beliefs_context(self,t,tau, likelihood_policies, posterior_policies, posterior_bundle):
        

        # construct \hat{p}(c_t) = int_{\eta} sum_{w_t} q(w_t)q(\eta) ln p(c_2|w_2,\eta')
        # DEBUG MAYBE NEED TO EXTEND BY ONE?
        if tau == 0:
            prior_context = self.prior_context[tau,0]
        else:
            prior_context = np.vstack([np.nan_to_num(self.posterior_context[tau-1,0]),self.prior_context[tau,0]]).T
            prior_context = prior_context.dot(posterior_bundle)

        if t>0:
            alphas = self.alpha_policy_counts[tau,t]

            # posterior_context =   self.ln(likelihood_policies) \
            #                      - self.ln(posterior_policies)\
            #                      + scp.digamma(alphas) - scp.digamma(alphas.sum(axis=0))
            # posterior_context = (posterior_policies*posterior_context).sum(axis=0) + self.ln(prior_context)                            
            outcome_surprise =  (posterior_policies * self.ln(likelihood_policies)).sum(axis=0)
            policy_entropy   = -(posterior_policies * self.ln(posterior_policies)).sum(axis=0)
            policy_surprise  =  (posterior_policies * (scp.digamma(alphas) - scp.digamma(alphas.sum(axis=0)))).sum(axis=0)

            if False and tau <30:
                print(f"prior_context   :{self.ln(prior_context).round(3)}")
                print(f"outcome_surprise:{outcome_surprise.round(3)}")
                print(f"policy_entropy  :{policy_entropy.round(3)}")
                print(f"policy_surprise :{policy_surprise.round(3)}")
            posterior_context = outcome_surprise + policy_entropy + policy_surprise + self.ln(prior_context)

        else:
            # DEBUG
            posterior_context = self.ln(prior_context)

        posterior_context = np.nan_to_num(scp.softmax(posterior_context))
        self.posterior_context[tau] = posterior_context[None,...]
        
        return posterior_context


    def update_beliefs_bundle(self,t,tau,posterior_context):
        
        # DEBUG INDEXES ARE SWITCHED?
        q_w0 = (np.nan_to_num(self.posterior_context[tau-1,0])**posterior_context).prod()*self.prior_bundle[tau,0,0]
        q_w1 = (self.prior_context[tau,0]**posterior_context).prod()*self.prior_bundle[tau,0,1]
        posterior_bundle = np.array([q_w0,q_w1])/np.array([q_w0,q_w1]).sum()
        
        if tau > 0:
            self.posterior_bundle[tau,:] = posterior_bundle[None,:]
        else:
            self.posterior_bundle[tau,:] = np.nan
            
        return posterior_bundle


    def update_beliefs_prior_rewards(self,tau, posterior_context):
        
        beta = self.beta_reward_counts[tau,0]
        beta_prime = beta.copy()

        for t in range(1,self.T):
        
            posterior_states = self.posterior_states[tau,t]
            posterior_policies = self.posterior_policies[tau,t]
            # posterior_context = self.posterior_context[tau,t]

            if self.k < self.nc:
                posterior_context = posterior_context[:-1]/posterior_context[:-1].sum() 
            elif not beta_prime[0,0,-1] is np.nan:
                beta_prime[:,:,-1] = self.beta_reward_counts[0,0][:,:,0]
            reward = self.rewards[tau,t]

            post_state = np.einsum('spc,pc->sc', posterior_states[:,t,:,:], posterior_policies)
            state = np.argmax(post_state[:,:self.k],axis=0)                                                  #deterministic state update

            beta_prime[reward,state,:self.k] += posterior_context[:self.k]
            
        self.beta_reward_counts[tau+1] = beta_prime[None,...]

        # normalize reward counts
        if self.approx_pred_rew:
            posterior_predictive_rewards = self.digamma_approximation(beta_prime)
        else:
            posterior_predictive_rewards = beta_prime / beta_prime.sum(axis=0) 

        self.prior_rewards[tau+1] = posterior_predictive_rewards[None,...]
        
        return posterior_predictive_rewards


    def update_beliefs_prior_policies(self,t,tau, posterior_context):

        pol_ind = self.linear_ind(self.actions[tau])[0]
        alphas = self.alpha_policy_counts[tau,t].copy()
        alphas_prime = alphas.copy()
        
        if self.k < self.nc:
            posterior_context = posterior_context[:-1] / posterior_context[:-1].sum()
        else:
            alphas_prime[:,-1] = 1/self.h
        
        alphas_prime[pol_ind,:self.k] += posterior_context

        self.alpha_policy_counts[tau+1] = alphas_prime[None,:,:]
        
        assert np.all(self.alpha_policy_counts[tau+1,:,:,-1] == 1/self.h)  # confirm empty context has no habit bias; perhaps talk about how to initialize psychologically? 

        if self.approx_pred_pol:
            # integral_{theta} q(theta) ln p(pi|c,theta;alpha') 
            posterior_predictive_policies = self.digamma_approximation(alphas_prime)
        else:
            # integral_{theta} q(theta) p(pi|c,theta;alpha')
            posterior_predictive_policies = alphas_prime / alphas_prime.sum(axis=0)
        
        self.prior_policies[tau+1] = posterior_predictive_policies[None,:,:]
        

    def update_beliefs_prior_context(self,t,tau, posterior_context):

        gamma = self.gamma_context_counts[tau,t].copy()
        gamma_prime = gamma.copy()

        if self.k < self.nc:
            posterior_context = posterior_context[:-1]/posterior_context[:-1].sum() 
        else:
            gamma_prime[-2:] = np.array([1,self.gamma_init])
            self.nc += 1

        gamma_prime[:self.k] += posterior_context
        self.gamma_context_counts[tau+1] = gamma_prime[None,:]

        posterior_predictive_context = self.digamma_approximation(gamma_prime)
        self.prior_context[tau+1] = posterior_predictive_context[None,:]


    def update_beliefs_prior_bundle(self,t,tau,posterior_bundle):
        
        epsilon = self.epsilon_bundle_counts[tau,0].copy()
        if tau > 0:
            epsilon += posterior_bundle
        self.epsilon_bundle_counts[tau+1,:,:] = epsilon[None,:]
        self.prior_bundle[tau+1,:,:] = self.digamma_approximation(epsilon)[None,:] 


class NonParamHierarchicalPerception():

    def __init__(self,
                 state_transition_matrix,              # p(s_t|s_t-1)
                 context_transition_matrix,            # p(c_t|c_t-1)
                 observation_generation_matrix,        # p(o_t|s_t)
                 utility,                              # pre-given desirability of observations
                 policies,                             # all possible policies given environment setup
                 prior_rewards,                        # p(s_t|s_t-1)
                 counts_prior_rewards,                 # hyperparameters beta of p(phi;beta)
                 prior_policies,                       # p(pi|theta)
                 counts_prior_policies,                # parameters alpha of p(theta;alpha)
                 prior_states,                         # initial p(s|c)
                 counts_prior_context,                 # hyperparameters gamma p(eta;gamma); symmetric for known context and kappa for trailing context dimension
                 prior_context,                        # initial p(c)
                 na,                                   #
                 nc,                                   #
                 env,                                  #
                 approx_pred_pol = True,               # use digamma approx when updating policy prior p(pi|c)
                 approx_pred_rew = True,               # use digamma approx when updating reward posterir p(r|s,c)
                 gamma_init = 0.2,                     # concentration parameter for Dirichlet Process
                 rho = 1.                              # context counts forgetting rate
                ):
        
 
        self.state_transition_matrix = state_transition_matrix
        self.observation_generation_matrix = observation_generation_matrix
        self.context_transition_matrix = context_transition_matrix
        self.utility = utility                                        
        self.policies = policies
        self.prior_states = prior_states                           
        self.approx_pred_pol = approx_pred_pol
        self.approx_pred_rew = approx_pred_rew
        self.na = na
        self.nc = nc
        self.k = nc - 1                                # number of currently inferred context; should be 1 unless we initialize agent with knowledge of more context 
        self.gamma_init = gamma_init                   # gamma_init is concentration parameter for Dirichlet process
        self.rho = rho
        
        #inherited from other classes
        self.environment = env
        self.TAU = env.TAU
        self.T = env.T
        self.nr = env.nr
        self.nc = nc
        self.ns = env.ns
        self.rewards = env.rewards

        # derived assignments
        self.npi = policies.shape[0]
        self.possible_policies = self.policies.copy()
        self.possible_policies_ind = np.arange(self.policies.shape[0])                                 

        # belief update logs
        self.inferred_new_context = np.full(self.TAU,False)
        self.posterior_states = np.zeros([self.TAU,self.T, self.ns, self.T, self.npi, self.nc])

        self.prior_policies = np.zeros([self.TAU, self.T, self.npi, self.nc])
        self.prior_policies[0,:] = prior_policies

        self.prior_policies_counts = np.zeros([self.TAU, self.T, self.npi, self.nc])
        self.prior_policies_counts[0,:] = counts_prior_policies[None,:,:]
        self.h =  1/np.unique(counts_prior_policies)[0]
        assert(np.unique(counts_prior_policies).size == 1)                                      # assumes all policies initialized the same!

        
        self.prior_rewards = np.zeros([self.TAU, self.T, self.nr, self.ns, self.nc])
        self.prior_rewards[0] = prior_rewards[None,:,:,:] 

        self.prior_rewards_counts = np.zeros([self.TAU, self.T, self.nr, self.ns, self.nc])
        self.prior_rewards_counts[0] = counts_prior_rewards[None,:,:,:]

        self.forward_norms = np.zeros([self.TAU, self.T, self.T+1, self.npi, self.nc])
        self.likelihood_policies = np.zeros([self.TAU, self.T, self.npi, self.nc])
        self.posterior_policies = np.zeros([self.TAU, self.T, self.npi, self.nc])
        
        self.prior_context = np.zeros([self.TAU, self.T, self.nc])
        self.prior_context[0,:] = prior_context[None,:]

        self.prior_context_counts = np.zeros([self.TAU, self.T, self.nc])
        self.prior_context_counts[0,:] = counts_prior_context[None,:]

        self.posterior_context = np.zeros([self.TAU, self.T, self.nc])
        

    def ln(self, array):
        array[array==0] = 1e-20
        return np.log(array)


    def linear_ind(self, array):
        array = array[:,None].T if array.shape[-1] == 1 else array.T
        return np.ravel_multi_index(array, [self.na]*(self.T-1))


    def expand_dimension(self, array):
        empty_dimension = np.empty(array.shape[:-1])
        empty_dimension[:] = np.nan
        array = np.append(array, empty_dimension[...,None], axis=-1)
        return array


    def open_new_context(self):
        
        self.forward_norms = self.expand_dimension(self.forward_norms)
        self.posterior_states = self.expand_dimension(self.posterior_states)

        self.likelihood_policies = self.expand_dimension(self.likelihood_policies)
        self.posterior_policies = self.expand_dimension(self.posterior_policies)

        self.posterior_context = self.expand_dimension(self.posterior_context)

        self.prior_policies_counts = self.expand_dimension(self.prior_policies_counts)
        self.prior_policies = self.expand_dimension(self.prior_policies)
        
        self.prior_context_counts = self.expand_dimension(self.prior_context_counts)
        self.prior_context = self.expand_dimension(self.prior_context)

        self.prior_rewards_counts = self.expand_dimension(self.prior_rewards_counts)
        self.prior_rewards = self.expand_dimension(self.prior_rewards)


    def digamma_approximation(self, counts):
        return scp.softmax(scp.digamma(counts) - scp.digamma(counts.sum(axis=0)),axis=0)
    
                                
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
    

    def update_beliefs_policies(self,t,tau):
        
        likelihood = self.fwd_norms.prod(axis=0)                      # exp(log(norms)) = -F(pi,c)
        posterior_policies  = likelihood*self.prior_policies[tau,t]   # exp(digamma(alpha_ij) - digamma(alpha_j)) when you integrate theta out
        posterior_policies /= posterior_policies.sum(axis=0)
        
        # store in global log
        self.likelihood_policies[tau,t] = likelihood/likelihood.sum(axis=0)
        self.posterior_policies[tau,t] = posterior_policies

        return likelihood, posterior_policies
    

    def update_beliefs_context(self,t,tau, likelihood_policies, posterior_policies):
        

        prior_context = self.prior_context[tau,0]

        if t>0:
            alphas = self.prior_policies_counts[tau,t]

            # posterior_context =   self.ln(likelihood_policies) \
            #                      - self.ln(posterior_policies)\
            #                      + scp.digamma(alphas) - scp.digamma(alphas.sum(axis=0))
            # posterior_context = (posterior_policies*posterior_context).sum(axis=0) + self.ln(prior_context)                            
            outcome_surprise =  (posterior_policies * self.ln(likelihood_policies)).sum(axis=0)
            policy_entropy   = -(posterior_policies * self.ln(posterior_policies)).sum(axis=0)
            policy_surprise  =  (posterior_policies * (scp.digamma(alphas) - scp.digamma(alphas.sum(axis=0)))).sum(axis=0)

            posterior_context = outcome_surprise + policy_entropy + policy_surprise + self.ln(prior_context)

            if True and tau <30:
                print(f"prior_context   :{self.ln(prior_context).round(3)}")
                print(f"outcome_surprise:{outcome_surprise.round(3)}")
                print(f"policy_entropy  :{policy_entropy.round(3)}")
                print(f"policy_surprise :{policy_surprise.round(3)}") 
        else:
            posterior_context = self.ln(prior_context)

        posterior_context = np.nan_to_num(scp.softmax(posterior_context))
        self.posterior_context[tau] = posterior_context[None,...] 
        return posterior_context
    

    def update_beliefs_prior_rewards(self,tau, posterior_context):
        
        beta = self.prior_rewards_counts[tau,0]
        beta_prime = beta.copy()

        for t in range(1,self.T):
        
            posterior_states = self.posterior_states[tau,t]
            posterior_policies = self.posterior_policies[tau,t]
            # posterior_context = self.posterior_context[tau,t]

            if self.k < self.nc:
                posterior_context = posterior_context[:-1]/posterior_context[:-1].sum() 
            elif not beta_prime[0,0,-1] is np.nan:
                beta_prime[:,:,-1] = self.prior_rewards_counts[0,0][:,:,0]

            reward = self.rewards[tau,t]

            post_state = np.einsum('spc,pc->sc', posterior_states[:,t,:,:], posterior_policies)
            state = np.argmax(post_state[:,:self.k],axis=0)                                                  #deterministic state update

            beta_prime[reward,state,:self.k] += posterior_context[:self.k]
            
        self.prior_rewards_counts[tau+1] = beta_prime[None,...]

        # normalize reward counts
        if self.approx_pred_rew:
            posterior_predictive_rewards = self.digamma_approximation(beta_prime)
        else:
            posterior_predictive_rewards = beta_prime / beta_prime.sum(axis=0) 

        self.prior_rewards[tau+1] = posterior_predictive_rewards[None,...]
        
        return posterior_predictive_rewards


    def update_beliefs_prior_policies(self,t,tau, posterior_context):

        pol_ind = self.linear_ind(self.actions[tau])[0]
        alphas = self.prior_policies_counts[tau,t].copy()
        alphas_prime = alphas.copy()
        
        if self.k < self.nc:
            posterior_context = posterior_context[:-1] / posterior_context[:-1].sum()
        else:
            alphas_prime[:,-1] = 1/self.h
        
        alphas_prime[pol_ind,:self.k] += posterior_context

        self.prior_policies_counts[tau+1] = alphas_prime[None,:,:]
        
        assert np.all(self.prior_policies_counts[tau+1,:,:,-1] == 1/self.h)  # confirm empty context has no habit bias; perhaps talk about how to initialize psychologically? 

        if self.approx_pred_pol:
            # integral_{theta} q(theta) ln p(pi|c,theta;alpha') 
            posterior_predictive_policies = self.digamma_approximation(alphas_prime)
        else:
            # integral_{theta} q(theta) p(pi|c,theta;alpha')
            posterior_predictive_policies = alphas_prime / alphas_prime.sum(axis=0)
        
        self.prior_policies[tau+1] = posterior_predictive_policies[None,:,:]
        

    def update_beliefs_prior_context(self,t,tau, posterior_context):

        gamma = self.prior_context_counts[tau,t].copy()
        gamma_prime = gamma.copy()

        if self.k < self.nc:
            posterior_context = posterior_context[:-1]/posterior_context[:-1].sum() 
        else:
            gamma_prime[-2:] = np.array([1,self.gamma_init])
            self.nc += 1


        # gamma_prime[:self.k] += posterior_context
        gamma_prime[:self.k] = self.rho*gamma_prime[:self.k] + posterior_context + (1-self.rho)*np.ones(self.k) 
        self.prior_context_counts[tau+1] = gamma_prime[None,:]

        posterior_predictive_context = self.digamma_approximation(gamma_prime)
        self.prior_context[tau+1] = posterior_predictive_context[None,:]
        

class HierarchicalPerception():

    def __init__(self,
                 state_transition_matrix,            # p(s_t|s_t-1)
                 utility,                            # pre-given desirability of observations
                 policies,                           # all possible policies given environment setup
                 prior_rewards,                      # p(s_t|s_t-1)
                 counts_prior_rewards,               # parameters beta of p(phi|beta)
                 prior_policies,                     # p(pi|theta)
                 counts_prior_policies,              # parameters alpha of p(theta|alpha)
                 prior_states,                       # initial p(s|c)
                 prior_context,                      # initial p(c)
                 na,
                 nc,
                 env,
                 approx_pred_pol = True,             # use digamma approx when updating policy prior p(pi|c)
                 approx_pred_rew = True,             # use digamma approx when updating reward posterir p(r|s,c)
                 observation_generation_matrix=None,  # p(o_t|s_t)
                 dec_temp = 1,
                ):
        
 
        self.state_transition_matrix = state_transition_matrix
        self.observation_generation_matrix = observation_generation_matrix   
        self.utility = utility                                        
        self.policies = policies
        self.prior_states = prior_states                           
        self.approx_pred_pol = approx_pred_pol
        self.approx_pred_rew = approx_pred_rew
        self.na = na
        self.nc = nc
        self.dec_temp = dec_temp
        
        #inherited from other classes
        self.environment = env
        self.TAU = env.TAU
        self.T = env.T
        self.nr = env.nr
        self.nc = nc
        self.ns = env.ns

        # derived assignments
        self.npi = policies.shape[0]
        self.possible_policies = self.policies.copy()
        self.possible_policies_ind = np.arange(self.policies.shape[0])                                 


        # belief update logs
        self.posterior_states = np.zeros([self.TAU,self.T, self.ns, self.T, self.npi, self.nc])

        self.prior_policies = np.zeros([self.TAU, self.T, self.npi, self.nc])
        self.prior_policies[0,:] = prior_policies

        self.prior_policies_counts = np.zeros([self.TAU, self.T, self.npi, self.nc])
        self.prior_policies_counts[0,:] = counts_prior_policies[None,:,:]
        
        self.prior_rewards = np.zeros([self.TAU, self.T, self.nr, self.ns, self.nc])
        self.prior_rewards[0,:] = prior_rewards[None,:,:,:]

        self.prior_rewards_counts = np.zeros([self.TAU, self.T, self.nr, self.ns, self.nc])
        self.prior_rewards_counts[0,0] = counts_prior_rewards

        self.forward_norms = np.zeros([self.TAU, self.T, self.T+1, self.npi, self.nc])
        self.likelihood_policies = np.zeros([self.TAU, self.T, self.npi, self.nc])
        self.posterior_policies = np.zeros([self.TAU, self.T, self.npi, self.nc])
        
        self.prior_context = np.zeros([self.TAU, self.T, self.nc])
        self.prior_context[0,:] = prior_context[None,:]
        self.posterior_context = np.zeros([self.TAU, self.T, self.nc])


    def ln(self, array):
        array[array==0] = 1e-20
        return np.log(array)


    def linear_ind(self, array):
        
        array = array[:,None].T if array.shape[-1] == 1 else array.T
        return np.ravel_multi_index(array, [self.na]*(self.T-1))


    def digamma_approximation(self, counts):
        return scp.softmax(scp.digamma(counts) - scp.digamma(counts.sum(axis=0)),axis=0)
    
                                
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

        if tau > 0:
            self.prior_rewards[tau,t] = self.prior_rewards[tau-1,self.T-1,].copy()

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
                    # for tp, u in enumerate(policy):
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

        return post
    

    def update_beliefs_policies(self,t,tau):
        
        likelihood = self.fwd_norms.prod(axis=0)                      # exp(log(norms)) = -F(pi,c) 
        posterior_policies  = np.power(likelihood, self.dec_temp) * self.prior_policies[tau,t]   # exp(digamma(alpha_ij) - digamma(alpha_j)) when you integrate theta out
        posterior_policies /= posterior_policies.sum(axis=0)
        
        # store in global log
        self.likelihood_policies[tau,t] = likelihood/likelihood.sum(axis=0)
        self.posterior_policies[tau,t] = posterior_policies


        return likelihood, posterior_policies
    

    def update_beliefs_context_sarah(self,t,tau, likelihood_policies, posterior_policies, prior_context):
        
        if t>0:
            alphas = self.prior_policies_counts[tau,t]

            posterior_context = (posterior_policies * self.ln(likelihood_policies)).sum(axis=0) \
                                - (posterior_policies * self.ln(posterior_policies)).sum(axis=0)\
                                + (posterior_policies * scp.digamma(alphas)).sum(axis=0) - scp.digamma(alphas.sum(axis=0))\
                                + self.ln(prior_context)                            
        else:
            posterior_context = self.ln(prior_context)

        posterior_context = np.nan_to_num(scp.softmax(posterior_context))
        self.posterior_context[tau,t] = posterior_context


        return posterior_context
    

    def update_beliefs_context(self,t,tau, likelihood_policies, posterior_policies, prior_context):
    
        # context-specific policy likelihood
        if t>0:
            alphas = self.prior_policies_counts[tau,t]

            # posterior_context =   self.ln(likelihood_policies) \
            #                      - self.ln(posterior_policies)\
            #                      + scp.digamma(alphas) - scp.digamma(alphas.sum(axis=0))
            # posterior_context = (posterior_policies*posterior_context).sum(axis=0) + self.ln(prior_context)                            
            outcome_surprise =   (posterior_policies * self.ln(likelihood_policies)).sum(axis=0)
            policy_entropy   =  -(posterior_policies * self.ln(posterior_policies)).sum(axis=0)
            policy_surprise  =   (posterior_policies * (scp.digamma(alphas) - scp.digamma(alphas.sum(axis=0)))).sum(axis=0)

            posterior_context = outcome_surprise + policy_entropy + policy_surprise + self.ln(prior_context)


            # print('\n',tau, self.rewards[tau,t], self.actions[tau][0])
            # print('outcome_surprise')
            # print(outcome_surprise.round(3))
            # print('policy_entropy')
            # print(policy_entropy.round(3))
            # print('policy_surprise')
            # print(policy_surprise.round(3))
            # print('prior_context')
            # print(self.ln(prior_context).round(3))
            # print('posterior context')
            # print(np.nan_to_num(scp.softmax(posterior_context)))


        else:
            posterior_context = self.ln(prior_context)

        posterior_context = np.nan_to_num(scp.softmax(posterior_context))
        self.posterior_context[tau,t] = posterior_context
        
        # if t == 0:
        #     print('\n',tau,t, self.rewards[tau,t], None)
        # else:
        #     print('\n',tau,t, self.rewards[tau,t], self.actions[tau])

        # print(prior_context)
        # print(posterior_context)

        return posterior_context
    

    def update_beliefs_prior_rewards(self,t,tau,reward,posterior_states, posterior_policies, posterior_context):
        
        post_state = np.einsum('spc,pc->sc', posterior_states[:,t,:,:], posterior_policies)
        state = np.argmax(post_state,axis=0)

        beta = self.prior_rewards_counts[tau,t-1]
        beta_prime = beta.copy()
        beta_prime[reward,state,:] += posterior_context

        self.prior_rewards_counts[tau,t] = beta_prime

        if self.approx_pred_rew:
            posterior_predictive_rewards = self.digamma_approximation(beta_prime)
        else:
            posterior_predictive_rewards = beta_prime / beta_prime.sum(axis=0) 

        if tau != self.TAU-1:
            if t == self.T-1:
                self.prior_rewards[tau+1,:2] = posterior_predictive_rewards
                self.prior_rewards_counts[tau+1,0] =  beta_prime
            else:
                self.prior_rewards[tau,t+1] = posterior_predictive_rewards

        return posterior_predictive_rewards


    def update_beliefs_prior_policies(self,t,tau, posterior_context):
        
        pol_ind = self.linear_ind(self.actions[tau])[0]
        alphas = self.prior_policies_counts[tau,t].copy()
        alphas_prime = alphas.copy()
        alphas_prime[pol_ind,:] += posterior_context
        self.prior_policies_counts[tau+1] = alphas_prime[None,:,:]

        if self.approx_pred_pol:
            posterior_predictive_policies = self.digamma_approximation(alphas_prime)
        else:
            posterior_predictive_policies = alphas_prime / alphas_prime.sum(axis=0)
        

        self.prior_policies[tau+1] = posterior_predictive_policies[None,:,:]
        
        # print('\n',tau, ', action: ', self.actions[tau][0])
        # print(posterior_context)
        # print(alphas_prime)
        # print(posterior_predictive_policies)

        if tau == 130:
            a=0


