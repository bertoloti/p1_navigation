# Project 1: Navigation

## Introduction

This file describes the implementation of the Navigation project such us the choosen algorithm for the DQN agent to the files composing the project.

## DQN Algorithm

For this project the **SARSAMAX** (AKA Q Learning) algorithm was selected in order to solve the environment. The **State Value Function Approximation** was done by a Deep Neural Network using experience replay and double DQN. The agent uses a **Epsilon-Greedy Policy** for dealing with the exploration-exploitation dilema.

## Project files

The project contains three files:
	
- *Navigation.ipynb*: This is the main file implemented as a jupyter notebook that contains all the necessary code in order to get the Unity Environment set up and running and the agent trained and tested. To execute the file just follow the instructions within the notebook. You will find all the descriptions regarding the code as well as the plots for agent evaluation.

- *dqn_agent.py*: This python file contains the class of the DQN agent as well as the ReplayBuffer class.

- *model.py*: File with the Deep Q Network implemented in pytorch.

### Classes description

- *Agent*: Main class that implements the DQN SARSAMAX learning algorithm. Contains the principals functions for Q Learning such as step or act.
- *ReplayBuffer*: Class that implements the experience replay memory.
- *QNetwork*: Neural Network class, contains 4 fully connected layers with 32 neurons each and *relu* as activation function. Input and output layers have the state and action size input/output dimension respectively.

## Hyper-parameters

The selection of the hyper-parameters were the following:
- **LR = 5e-4** *learning rate*
- **Gamma = 0.99** *discount factor*
- **TAU = 1e-3** *parameter for soft update between target and local network weights*
- **UPDATE_EVERY = 4** *target network update every 4 steps*
- **BATH_SIZE = 64** *mini batch size for experience replay sampling*