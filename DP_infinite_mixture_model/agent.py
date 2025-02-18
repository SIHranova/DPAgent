import numpy as np


class NonParamAgent():

    def __init__(self,
                 state_transition_matrix,
                 na,
                 env,
                 perception,
                 approx_pred_pol = True,
                 approx_pred_rew = True,
                ):
        
        # direct assignments
        self.state_transition_matrix = state_transition_matrix
        self.na = na

        #inherited from env class
        self.environment = env
        self.TAU = env.TAU
        self.T = env.T
        self.nr = env.nr
        self.ns = env.ns
        self.actions = env.actions
        self.rewards = env.rewards
  

        #inherited from perception class
        self.perc = perception
        self.kappa = perception.kappa

        # self.prior_rewards = perception.prior_rewards
        # self.prior_rewards_counts = perception.prior_rewards_counts
        
        # self.prior_policies = perception.prior_policies
        # self.prior_policies_counts = perception.prior_policies_counts

        # self.posterior_states = perception.posterior_states
        # self.posterior_policies = perception.prior_policies

        # self.forward_norms = perception.forward_norms
        # self.likelihood_policies = perception.likelihood_policies
        # self.posterior_policies = perception.posterior_policies

        # self.prior_context = perception.prior_context
        # self.posterior_context = perception.posterior_context
        # self.policy_entropy = perception.policy_entropy
        # self.policy_predictive_posterior = perception.policy_predictive_posterior

        self.context_transition_matrix = perception.context_transition_matrix

        self.perc.actions = self.actions
        self.perc.rewards = self.rewards
        self.policies = self.perc.policies
        self.possible_policies = self.perc.possible_policies


    def update_beliefs(self, t, tau, state, reward, action, observation):

                
        self.perc.update_beliefs_states(t, tau, reward, action, observation)
        
        likelihood_policies, posterior_policies = self.perc.update_beliefs_policies(t,tau)

        if (t == self.T-1 and tau < self.TAU-1):
            
            posterior_context = self.perc.update_beliefs_context(t,tau, likelihood_policies, posterior_policies)

            if tau == 0 or tau > 4:

                c =  np.argmax(posterior_context) + 1 #self.sample_context(t,tau, posterior_context)
                posterior_context = np.eye(self.perc.nc)[c-1]
                # self.perc.posterior_context[tau,:] = posterior_context[None,:]
                if c > self.perc.k:
                    print( f"inferred new context at {tau}")
                    self.perc.open_new_context()
                    self.perc.k += 1
                    self.perc.inferred_new_context[tau] = True
                    # posterior_context = []
            
                
            self.perc.update_beliefs_prior_policies(t,tau, posterior_context)
            self.perc.update_beliefs_prior_rewards(tau)
            self.perc.update_beliefs_prior_context(t,tau,posterior_context)

            a = 0


        if True and tau <30:
            print(f"--------------------\ntau,t: {tau,t}")
            print(f"action: {action}, observation: {observation}, reward: {reward}")

            print(f"\nq(r|pi,c); policy likelihood:")
            print(likelihood_policies.round(4))

            print(f"\nq(pi|c) policy posterior:")
            print(posterior_policies.round(4))
            
            

            if t == self.T-1:
                if tau > 0:
                    print(f"\nq(c):")
                    print(posterior_context)
                
                if self.perc.inferred_new_context[tau]:
                    print("opened new context!")

                print(f"\nprior context counts")
                print(self.perc.prior_context_counts[tau+1,0])

                print(f"\nprior context counts")
                print(self.perc.prior_context[tau+1,0])

                print(f"\nrewards counts:")
                print(f"obs, reward: {observation, reward}")
                for k in range(self.perc.k+1):
                    print(f"\n{self.perc.prior_rewards_counts[tau+1,t][:,:,k]}")
                
                print(f"\nprior_rewards")
                for k in range(self.perc.k+1):
                    print(self.perc.prior_rewards[tau+1,t][:,:,k].round(3))


                print(f"\npolicy counts")
                print(f"chosen policy:{action}")
                print(self.perc.prior_policies_counts[tau+1,t])
                print(self.perc.prior_policies[tau+1,t].round(4))
                


    def sample_context(self,t,tau,posterior_context):
        return np.random.choice(np.arange(self.perc.nc), p=posterior_context) + 1 # shift since, this is a context counter variable


    def sample_action(self,t,tau):

        post_policies = self.perc.posterior_policies[tau,t]
        prior_context = self.perc.prior_context[tau,t]
        post_policies = post_policies.dot(prior_context)
        # chosen_action = self.policies[np.argmax(post_policies)][t]
        
        post_actions = np.zeros(self.na)
        for a in range(self.na):
            post_actions[a] = post_policies[self.policies[:,t] == a].sum()

        chosen_action = np.random.choice(np.arange(self.na), p=post_actions)
        self.actions[tau,t] = chosen_action
        
        return chosen_action


class Agent():

    def __init__(self,
                 state_transition_matrix,
                 na,
                 nc,
                 env,
                 perception,
                 approx_pred_pol = True,
                 approx_pred_rew = True
                ):
        
        # direct assignments
        self.state_transition_matrix = state_transition_matrix
        self.na = na
        self.nc = nc

        #inherited from env class
        self.environment = env
        self.TAU = env.TAU
        self.T = env.T
        self.nr = env.nr
        self.nc = nc
        self.ns = env.ns
        self.actions = env.actions
        self.rewards = env.rewards
  

        #inherited from perception class
        self.perc = perception
        self.prior_rewards = perception.prior_rewards
        self.prior_rewards_counts = perception.prior_rewards_counts
        
        self.prior_policies = perception.prior_policies
        self.prior_policies_counts = perception.prior_policies_counts

        self.posterior_states = perception.posterior_states
        self.posterior_policies = perception.prior_policies
        self.forward_norms = perception.forward_norms
        self.likelihood_policies = perception.likelihood_policies
        self.posterior_policies = perception.posterior_policies

        self.prior_context = perception.prior_context
        self.posterior_context = perception.posterior_context
        # self.policy_entropy = perception.policy_entropy
        # self.policy_predictive_posterior = perception.policy_predictive_posterior

        self.perc.actions = self.actions
        self.perc.rewards = self.rewards
        self.policies = self.perc.policies
        self.possible_policies = self.perc.possible_policies
    

    def update_beliefs(self, t, tau, state, reward, action, observation):
        
        posterior_states = self.perc.update_beliefs_states(t, tau, reward, action, observation)
        
        likelihood_policies, posterior_policies = self.perc.update_beliefs_policies(t,tau)

        if t == 0 and tau != 0:
            prior_context = self.posterior_context[tau-1,-1]
            p = 0.95
            q = (1-p)/(self.nc-1)
            context_transition_matrix = np.eye(self.nc)*(1-2*q) + q
            # context_transition_matrix = np.eye(self.nc)
            # print(context_transition_matrix)
            self.prior_context[tau] = context_transition_matrix.dot(prior_context)
            # in future add here context transition matrix

        prior_context = self.prior_context[tau,t]
        posterior_context = self.perc.update_beliefs_context(t,tau, likelihood_policies, posterior_policies, prior_context)

        if t > 0:
            posterior_rewards = self.perc.update_beliefs_prior_rewards(t,tau,reward, posterior_states,
                                                                        posterior_policies,posterior_context)

        if (t == self.T-1 and tau < self.TAU-1):
            self.perc.update_beliefs_prior_policies(t, tau, posterior_context)

    def sample_action(self,t,tau):

        post_policies = self.posterior_policies[tau,t]
        post_context = self.posterior_context[tau,t]

        post_policies = post_policies.dot(post_context)
        # chosen_action = self.policies[np.argmax(post_policies)][t]
        
        post_actions = np.zeros(self.na)
        for a in range(self.na):
            post_actions[a] = post_policies[self.policies[:,t] == a].sum()

        chosen_action = np.random.choice(np.arange(self.na), p=post_actions)
        self.actions[tau,t] = chosen_action
        
        return chosen_action

