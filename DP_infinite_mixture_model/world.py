import numpy as np

class World():

    def __init__(self, agent, environment,training_protocol=None):

        self.environment = environment
        self.agent = agent
        # self.perc - self.agent.perc
        self.TAU = environment.TAU
        self.T = environment.T
        self.nr = environment.nr


    def simulate_experiment(self, TAU=None):

        for tau in range(self.TAU):
            if tau == 21:
                a = 0
            for t in range(self.T):
                if t == 0:
                    action=None
                    state = self.environment.initialize_hidden_state(tau, starting_state=self.environment.nb)
                else:
                    state = self.environment.sample_hidden_state(t, tau, action)

                observation = self.environment.generate_observation(t,tau,state)

                reward = self.environment.sample_reward(t, tau, state)
                
                self.agent.update_beliefs(t, tau, state, reward, action, observation)

                if t < self.T-1:
                    action = self.agent.sample_action(t,tau)

                # if t == self.T-1:
                #     perc = self.agent.perc
                #     for c in range(perc.k):
                #         print(f"\n\ntrial: {tau}, t: {t}, context: {c}")
                #         print(f"action: {action}, reward: {reward}")
                #         print(f"\npost_states:\n{perc.posterior_states[tau,t,:,:,action,c]}")
                #         print(f"\nprior_context:\n{perc.prior_context[tau,t].round(3)}")
                #         print(f"\nposterior_context:\n{perc.posterior_context[tau,t].round(3)}")
                #         print(f"\nprior_rewards:\n{perc.prior_rewards_counts[tau+1,t,:,:,c].round(3)}")
                #         print(f"\nprior_policies:\n{perc.prior_policies_counts[tau+1,t,:,c].round(3)}")
                
                #     if tau==103:
                #         a=0
                        
                        
