import matplotlib.pyplot as plt
import torch
import pickle


# Prints the reward per episode graph
def reward_print(reward_per_episode, episodes, info):
    mins = int(min(reward_per_episode)) - abs(int(min(reward_per_episode)) * (.2))
    maxs = int(max(reward_per_episode)) + abs(int(max(reward_per_episode)) * (.3))
    plt.figure()
    plt.plot(reward_per_episode)
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Cumulative Reward', fontsize=20)
    plt.title(f'Cumulative Reward Per Episode ({info})', fontsize=24)
    plt.xticks([0, episodes * .2, episodes * .4, episodes * .6, episodes * .8, episodes], fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylim(ymin=mins, ymax=maxs)
    plt.xlim(xmin=0, xmax=episodes)
    plt.grid()
    plt.show()


# prints the epsilon decay graph
def ep_decay(eps, episodes):
    epsilon_values = [(eps ** i) * 1 for i in range(episodes)]
    plt.figure()
    plt.plot(epsilon_values, linewidth=4)
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Epsilon Value', fontsize=20)
    plt.title(f"Epsilon Decay for {eps}", fontsize=24)
    plt.xticks([0, episodes * .2, episodes * .4, episodes * .6, episodes * .8, episodes], fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylim(ymin=0, ymax=1)
    plt.xlim(xmin=0, xmax=episodes)
    plt.grid()
    plt.show()


def print_state_dict(filename, agent):
    if agent == "Actor":
        print("Actor")
        with open("pickles/" + filename + "actor.pickle", 'rb') as file:
            Actor = pickle.load(file)
        for key, value in Actor.items():
            print(f"{key}: {value}")
    elif agent == "Critic":
        print("Critic")
        with open("pickles/" + filename + "actor.pickle", 'rb') as file:
            Actor = pickle.load(file)
        for key, value in Actor.items():
            print(f"{key}: {value}")
