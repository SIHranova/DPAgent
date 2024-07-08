import numpy as np


class NonParamAgent():

    def __init__(self,
                 state_transition_matrix,
                 na,
                 nc,
                 env,
                 perception,
                 approx_pred_pol = True,
                 approx_pred_rew = True,
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
        self.kappa = perception.kappa
        self.prior_rewards = perception.prior_rewards
        self.prior_rewards_counts = perception.prior_rewards_counts
        
        self.prior_policies = perception.prior_policies
        self.prior_policies_counts = perception.prior_policies_counts

        self.posterior_states = perception.posterior_states
        self.posterior_policies = perception.prior_policies
        self.forward_norms = perception.forward_norms
        self.likelihood_policies = perception.likelihood_policies
        self.posterior_policies = perception.posterior_policies

        self.prior_contexts = perception.prior_contexts
        self.posterior_contexts = perception.posterior_contexts
        # self.policy_entropy = perception.policy_entropy
        # self.policy_predictive_posterior = perception.policy_predictive_posterior
        self.context_transition_matrix = perception.context_transition_matrix
        self.perc.actions = self.actions
        self.perc.rewards = self.rewards
        self.policies = self.perc.policies
        self.possible_policies = self.perc.possible_policies
    

    def update_beliefs(self, t, tau, state, reward, action, observation):
        
        #ok
        posterior_states = self.perc.update_beliefs_states(t, tau, reward, action, observation)
        
        likelihood_policies, posterior_policies = self.perc.update_beliefs_policies(t,tau)

        if t == 0 and tau != 0:
            prior_context = self.posterior_contexts[tau-1,-1]
            self.prior_contexts[tau] = self.context_transition_matrix.dot(prior_context)
            # in future add here context transition matrix

        prior_context = self.prior_contexts[tau,t]

        posterior_context = self.perc.update_beliefs_context(t,tau, likelihood_policies, posterior_policies, prior_context)

        if t > 0:
            #changed
            posterior_rewards = self.perc.update_beliefs_prior_rewards(t,tau,reward, posterior_states,
                                                                        posterior_policies,posterior_context)

        if (t == self.T-1 and tau < self.TAU-1):
            #changed
            self.perc.update_beliefs_prior_policies(t, tau, posterior_context)

        
    def sample_action(self,t,tau):

        post_policies = self.posterior_policies[tau,t]
        post_context = self.posterior_contexts[tau,t]

        post_policies = post_policies.dot(post_context)
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

        self.prior_contexts = perception.prior_contexts
        self.posterior_contexts = perception.posterior_contexts
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
            prior_context = self.posterior_contexts[tau-1,-1]
            p = 0.989
            q = (1-p)/(self.nc-1)
            context_transition_matrix = np.eye(self.nc)*(1-2*q) + q
            # context_transition_matrix = np.eye(self.nc)
            self.prior_contexts[tau] = context_transition_matrix.dot(prior_context)
            # in future add here context transition matrix

        prior_context = self.prior_contexts[tau,t]
        posterior_context = self.perc.update_beliefs_context(t,tau, likelihood_policies, posterior_policies, prior_context)

        if t > 0:
            posterior_rewards = self.perc.update_beliefs_prior_rewards(t,tau,reward, posterior_states,
                                                                        posterior_policies,posterior_context)

        if (t == self.T-1 and tau < self.TAU-1):
            self.perc.update_beliefs_prior_policies(t, tau, posterior_context)

        
    def sample_action(self,t,tau):

        post_policies = self.posterior_policies[tau,t]
        post_context = self.posterior_contexts[tau,t]

        post_policies = post_policies.dot(post_context)
        # chosen_action = self.policies[np.argmax(post_policies)][t]
        
        post_actions = np.zeros(self.na)
        for a in range(self.na):
            post_actions[a] = post_policies[self.policies[:,t] == a].sum()

        chosen_action = np.random.choice(np.arange(self.na), p=post_actions)
        self.actions[tau,t] = chosen_action
        
        return chosen_action

