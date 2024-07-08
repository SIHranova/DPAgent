import numpy as np
import scipy.special as scp



class NonParamHierarchicalPerception():

    def __init__(self,
                 state_transition_matrix,            # p(s_t|s_t-1)
                 context_transition_matrix,          # p(c_t|c_t-1)
                 utility,                            # pre-given desirability of observations
                 policies,                           # all possible policies given environment setup
                 prior_rewards,                      # p(s_t|s_t-1)
                 counts_prior_rewards,               # parameters beta of p(phi|beta)
                 prior_policies,                     # p(pi|theta)
                 counts_prior_policies,              # parameters alpha of p(theta|alpha)
                 prior_states,                       # initial p(s|c)
                 prior_context,                      # initial p(c)
                 na,                                 #
                 nc,                                 #
                 env,                                #
                 approx_pred_pol = True,             # use digamma approx when updating policy prior p(pi|c)
                 approx_pred_rew = True,             # use digamma approx when updating reward posterir p(r|s,c)
                 observation_generation_matrix=None, # p(o_t|s_t)
                 kappa = 0.2                         # concentration parameter for Dirichlet Process
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
        self.k = nc - 1                                # number of currently inferred contexts; should be 1 unless we initialize agent with knowledge of more contexts 
        self.kappa = kappa                             # kappa is concentration parameter for Dirichlet process
        
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
        self.prior_rewards[0,:2] = prior_rewards[None,:,:,:]

        self.prior_rewards_counts = np.zeros([self.TAU, self.T, self.nr, self.ns, self.nc])
        self.prior_rewards_counts[0,0] = counts_prior_rewards

        self.forward_norms = np.zeros([self.TAU, self.T, self.T+1, self.npi, self.nc])
        self.likelihood_policies = np.zeros([self.TAU, self.T, self.npi, self.nc])
        self.posterior_policies = np.zeros([self.TAU, self.T, self.npi, self.nc])
        
        self.prior_contexts = np.zeros([self.TAU, self.T, self.nc])
        self.prior_contexts[0,:] = prior_context[None,:]

        self.posterior_contexts = np.zeros([self.TAU, self.T, self.nc])


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

        return post
    

    def update_beliefs_policies(self,t,tau):
        
        likelihood = self.fwd_norms.prod(axis=0)                      # exp(log(norms)) = -F(pi,c)
        posterior_policies  = likelihood*self.prior_policies[tau,t]   # exp(digamma(alpha_ij) - digamma(alpha_j)) when you integrate theta out
        posterior_policies /= posterior_policies.sum(axis=0)
        
        # store in global log
        self.likelihood_policies[tau,t] = likelihood/likelihood.sum(axis=0)
        self.posterior_policies[tau,t] = posterior_policies

        return likelihood, posterior_policies
    
    def update_beliefs_context(self,t,tau, likelihood_policies, posterior_policies, prior_context):
    
        # context-specific policy likelihood
        if t> 0:
            alphas = self.prior_policies_counts[tau,t]

            # posterior_context =   self.ln(likelihood_policies) \
            #                      - self.ln(posterior_policies)\
            #                      + scp.digamma(alphas) - scp.digamma(alphas.sum(axis=0))
            # posterior_context = (posterior_policies*posterior_context).sum(axis=0) + self.ln(prior_context)                            
            outcome_surprise = (posterior_policies * self.ln(likelihood_policies)).sum(axis=0)
            policy_entropy =  -(posterior_policies * self.ln(posterior_policies)).sum(axis=0)
            policy_surprise = (posterior_policies * (scp.digamma(alphas) - scp.digamma(alphas.sum(axis=0)))).sum(axis=0)

            posterior_context = outcome_surprise + policy_entropy + policy_surprise + self.ln(prior_context)


            print('\n',tau, self.rewards[tau,t], self.actions[tau][0])
            print('outcome_surprise')
            print(outcome_surprise.round(3))
            print('policy_entropy')
            print(policy_entropy.round(3))
            print('policy_surprise')
            print(policy_surprise.round(3))
            print('prior_context')
            print(self.ln(prior_context).round(3))
            print('posterior context')
            print(np.nan_to_num(scp.softmax(posterior_context)))

            if tau == 130:
                a=0

        else:
            posterior_context = self.ln(prior_context)

        posterior_context = np.nan_to_num(scp.softmax(posterior_context))
        self.posterior_contexts[tau,t] = posterior_context
        
        # if t == 0:
        #     print('\n',tau,t, self.rewards[tau,t], None)
        # else:
        #     print('\n',tau,t, self.rewards[tau,t], self.actions[tau])

        # print(prior_context)
        # print(posterior_context)
        
        if tau == 100:
            a=0

        return posterior_context
    

    def update_beliefs_prior_rewards(self,t,tau,reward,posterior_states, posterior_policies, posterior_context):
        
        # update reward counts beta
        post_state = np.einsum('spc,pc->sc', posterior_states[:,t,:,:], posterior_policies)
        state = np.argmax(post_state,axis=0)

        beta = self.prior_rewards_counts[tau,t-1]
        beta_prime = beta.copy()
        beta_prime[reward,state,:-1] += posterior_context[:-1]
        self.prior_rewards_counts[tau,t] = beta_prime

        assert np.all(beta[:,:,-1] == beta_prime[:,:,-1])

        # normalize reward counts
        if self.approx_pred_rew:
            posterior_predictive_rewards = self.digamma_approximation(beta_prime)
        else:
            posterior_predictive_rewards = beta_prime / beta_prime.sum(axis=0) 

        # carry over information for next trial
        if tau != self.TAU-1:
            if t == self.T-1:
                #check if still works without index?
                self.prior_rewards[tau+1] = posterior_predictive_rewards
                self.prior_rewards_counts[tau+1,0] =  beta_prime
            else:
                self.prior_rewards[tau,t+1] = posterior_predictive_rewards

        return posterior_predictive_rewards


    def update_beliefs_prior_policies(self,t,tau, posterior_context):
        
        pol_ind = self.linear_ind(self.actions[tau])[0]
        alphas = self.prior_policies_counts[tau,t].copy()
        alphas_prime = alphas.copy()
        alphas_prime[pol_ind,:-1] += posterior_context[:-1]
        self.prior_policies_counts[tau+1] = alphas_prime[None,:,:]
        
        assert np.all(self.prior_policies_counts[tau,:,-1] == self.prior_policies_counts[tau+1,:,-1])

        if self.approx_pred_pol:
            posterior_predictive_policies = self.digamma_approximation(alphas_prime)
        else:
            posterior_predictive_policies = alphas_prime / alphas_prime.sum(axis=0)
        
        self.prior_policies[tau+1] = posterior_predictive_policies[None,:,:]
        




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
                 observation_generation_matrix=None  # p(o_t|s_t)
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
        self.prior_rewards[0,:2] = prior_rewards[None,:,:,:]

        self.prior_rewards_counts = np.zeros([self.TAU, self.T, self.nr, self.ns, self.nc])
        self.prior_rewards_counts[0,0] = counts_prior_rewards

        self.forward_norms = np.zeros([self.TAU, self.T, self.T+1, self.npi, self.nc])
        self.likelihood_policies = np.zeros([self.TAU, self.T, self.npi, self.nc])
        self.posterior_policies = np.zeros([self.TAU, self.T, self.npi, self.nc])
        
        self.prior_contexts = np.zeros([self.TAU, self.T, self.nc])
        self.prior_contexts[0,:] = prior_context[None,:]
        self.posterior_contexts = np.zeros([self.TAU, self.T, self.nc])


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

        return post
    

    def update_beliefs_policies(self,t,tau):
        
        likelihood = self.fwd_norms.prod(axis=0)                      # exp(log(norms)) = -F(pi,c)
        posterior_policies  = likelihood*self.prior_policies[tau,t]   # exp(digamma(alpha_ij) - digamma(alpha_j)) when you integrate theta out
        posterior_policies /= posterior_policies.sum(axis=0)
        
        # store in global log
        self.likelihood_policies[tau,t] = likelihood/likelihood.sum(axis=0)
        self.posterior_policies[tau,t] = posterior_policies

        # print('\n',tau,t)
        # print(likelihood/likelihood.sum(axis=0))

        # if tau==190:
        #     a = 0
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
        self.posterior_contexts[tau,t] = posterior_context

        # print('\n',tau,t,self.rewards[tau,t],self.actions[tau,t-1])
        # print((posterior_policies * self.ln(likelihood_policies))[0])
        # print((- (posterior_policies * self.ln(posterior_policies)))[0])
        # print((+ posterior_policies * (scp.digamma(alphas) - scp.digamma(alphas.sum(axis=0))))[0])
        # (posterior_policies * scp.digamma(alphas)).sum(axis=0) - scp.digamma(alphas.sum(axis=0))
        # print(self.ln(prior_context))
        # print(+ (posterior_policies * scp.digamma(alphas)).sum(axis=0) - scp.digamma(alphas.sum(axis=0)))

        return posterior_context
    

    def update_beliefs_context(self,t,tau, likelihood_policies, posterior_policies, prior_context):
    
        # context-specific policy likelihood
        if t> 0:
            alphas = self.prior_policies_counts[tau,t]

            # posterior_context =   self.ln(likelihood_policies) \
            #                      - self.ln(posterior_policies)\
            #                      + scp.digamma(alphas) - scp.digamma(alphas.sum(axis=0))
            # posterior_context = (posterior_policies*posterior_context).sum(axis=0) + self.ln(prior_context)                            
            outcome_surprise = (posterior_policies * self.ln(likelihood_policies)).sum(axis=0)
            policy_entropy =  -(posterior_policies * self.ln(posterior_policies)).sum(axis=0)
            policy_surprise = (posterior_policies * (scp.digamma(alphas) - scp.digamma(alphas.sum(axis=0)))).sum(axis=0)

            posterior_context = outcome_surprise + policy_entropy + policy_surprise + self.ln(prior_context)


            print('\n',tau, self.rewards[tau,t], self.actions[tau][0])
            print('outcome_surprise')
            print(outcome_surprise.round(3))
            print('policy_entropy')
            print(policy_entropy.round(3))
            print('policy_surprise')
            print(policy_surprise.round(3))
            print('prior_context')
            print(self.ln(prior_context).round(3))
            print('posterior context')
            print(np.nan_to_num(scp.softmax(posterior_context)))

            if tau == 130:
                a=0

        else:
            posterior_context = self.ln(prior_context)

        posterior_context = np.nan_to_num(scp.softmax(posterior_context))
        self.posterior_contexts[tau,t] = posterior_context
        
        # if t == 0:
        #     print('\n',tau,t, self.rewards[tau,t], None)
        # else:
        #     print('\n',tau,t, self.rewards[tau,t], self.actions[tau])

        # print(prior_context)
        # print(posterior_context)
        
        if tau == 100:
            a=0

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


