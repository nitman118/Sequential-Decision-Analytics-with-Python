#1.3.1
import simpy
import numpy as np
import random

class EOQ:
    def __init__(self, mean_D, sig_D, p, c, H):
        self.env = simpy.Environment()
        self.mean_D = mean_D
        self.sig_D = sig_D
        self.R_t = 0
        self.p = p
        self.c = c
        self.H = H
        self.reward = 0
        self.end_time_horizon_event = self.env.timeout(H)
        self.new_day_event = self.env.timeout(1)
        self.env.process(self.customer_generator())
        self.num_cust = 0
        

    def customer_generator(self):
        while True:
            yield self.env.timeout(random.expovariate(self.mean_D))
            self.customer()
            self.num_cust += 1

    def customer(self):
        # print('customer')
        if self.R_t <= 0:
            self.reward += 0
        else:
            self.reward += self.c
        self.R_t = max(0, self.R_t - 1)

    def reset(self):
        self.R_t = 0
        self.reward = 0
        self.env = simpy.Environment()
        self.env.process(self.customer_generator())
        self.end_time_horizon_event = self.env.timeout(self.H)
        self.new_day_event = self.env.timeout(1)
        self.num_cust = 0

        return self.R_t, {'num_cust': self.num_cust}

    
    def step(self, action):
        self.R_t = self.R_t + action
        self.reward -= self.p * action
        self.num_cust = 0

        self.env.run(until=simpy.events.AnyOf(self.env, [self.new_day_event, self.end_time_horizon_event]))
        
        if self.end_time_horizon_event.processed:
            # The time horizon has been reached
            # print(self.env.now)
            # print('end')
            reward = 0
            done = True
            
        elif self.new_day_event.processed:
            # A new day has started
            # print('new day')
            # print(self.env.now)
            reward = self.reward
            self.reward = 0
            done = False
            self.new_day_event = self.env.timeout(1)
            
            # print(self.new_day_event.processed)

        # print(self.env.now, self.R_t, reward, done)
        return self.R_t, reward, done, {'num_cust': self.num_cust}





if __name__ == "__main__":
    env = EOQ(mean_D=150, sig_D=25, p=10, c=15, H=30)

    thetas = [x for x in range(300)]

    for theta in thetas:
        state, info = env.reset()
        done = False
        total_reward = 0
        i = 0
        while not done:
            state, reward, done, info = env.step(theta)
            i+=1
            total_reward += reward
            # print(info['num_cust'])
    
        print(f'Theta: {theta}. Total reward: {total_reward}')