import numpy as np

class World():

    def __init__(self, agent, environment,training_protocol=None):

        self.agent = agent
        self.environment = environment
        self.TAU = environment.TAU
        self.T = environment.T
        self.nr = environment.nr


    def simulate_experiment(self, TAU=None):

        if TAU == None:
            TAU = self.TAU

        for tau in range(TAU):

            for t in range(self.T):

                if tau == 0 and t==1:
                    pass

                if t == 0:
                    action=None
                    state = self.environment.initialize_hidden_state(tau, starting_state=2)
                else:
                    state = self.environment.sample_hidden_state(t, tau, action)

                observation = self.environment.generate_observation(t,tau,state)

                reward = self.environment.sample_reward(t, tau, state)
                
                self.agent.update_beliefs(t, tau, state, reward, action, observation)

                if t < self.T-1:
                    action = self.agent.sample_action(t,tau)
