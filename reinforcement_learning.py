import gym
import numpy as np
import matplotlib.pyplot as plt

# Create environment
env = gym.make("FrozenLake-v1", render_mode = 'rgb_array', is_slippery= False)
env.reset()
env.render()

#declare number of actions and states
action_size = env.action_space.n
print("Action size: ", action_size)
state_size = env.observation_space.n
print("State size: ", state_size)

done = False
env.reset()
'''
while not done:
    action = np.random.randint(0,4) # 0:Left 1:Down 2: Right, 3: Up
    #action = int(input('0/left 1/down 2/right 3/up:'))
    new_state, reward, done,truncated, info = env.step(action)
    #print(new_state, reward, done, info)
    time.sleep(1.0)
    print(f'S_t+1={new_state}, R_t+1={reward}, done={done}')
    env.render()
'''

#establish a new qtable
qtable = np.random.rand(state_size, action_size)
#qtable = np.zeros((state_size, action_size))
print(qtable)


def eval_policy(qtable_, num_of_episodes_, max_steps_):
    rewards = []
    for episode in range(num_of_episodes_):
        state = env.reset()
        step = 0
        done = False
        total_rewards = 0
        state = state[0]
        for step in range(max_steps_):
            action = np.argmax(qtable_[int(state), :])
            new_state, reward, done, truncated, info = env.step(action)
            total_rewards += reward

            if done:
                rewards.append(total_rewards)
                break
            state = new_state
    env.close()
    avg_reward = sum(rewards) / num_of_episodes_
    return avg_reward

print(f'Average reward: {eval_policy(qtable,100,100)}')

reward_best = -1000
total_episodes = 1001
max_steps = 100
printed_x = {} #for plotting purpose (dict)
#updating the qtable
for i in range(10):
    to_print = []
    qtable = np.zeros((state_size, action_size))   # all zero Q-table
    #qtable = np.random.rand(state_size, action_size) #random Q-table
    for episode in range(total_episodes):
        env.reset()
        state = 0
        step = 0
        done = False
        # reward_tot = 0
        for step in range(max_steps):
            state_no = state
            action = np.random.randint(0, 4)  # getting random actions
            new_state, reward, done, truncated, info = env.step(action)
            # reward_tot += reward
            new_state_no = new_state
            qtable[int(state_no)][action] = reward + 0.9 * np.amax(qtable[int(new_state_no)])
            #qtable[int(state_no)][action] =qtable[int(state_no)][action] + 0.5*(reward + 0.9 * np.amax(qtable[int(new_state_no)]) - qtable[int(state_no)][action])
            if done == True:
                break
            state = new_state
        if episode % 20 == 0:
            to_print.append(eval_policy(qtable,1000,100))
        #if episode % 100 == 0:
            #print(f'Best reward after episode {episode + 1} is {eval_policy(qtable, 10, 100)}')
    printed_x[i] = to_print

print (f'Tot reward of the found policy: {eval_policy(qtable,1000,100)}')
print(qtable)
plt.figure(figsize = (10, 4), dpi = 100)
plt.title("Deteministic cases with deterministic update rule")
#plt.title("Non - deteministic cases with deterministic update rule")
#plt.title("Non - deteministic cases with non-deterministic update rule")
x = list(range(1, 1020, 20))
for i in range(10):
    y = printed_x[i]
    plt.plot(x, y)
plt.xlabel("No. of epsiodes")
plt.show()
