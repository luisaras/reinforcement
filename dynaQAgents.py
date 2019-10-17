# dynaAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# Dyna Agent support by Anderson Tavares (artavares@inf.ufrgs.br)


from game import *
from learningAgents import ReinforcementAgent
import heapq

import random,util,math

class DynaQAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)
        - self.plan_steps (number of planning iterations)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, plan_steps=5, kappa=0, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        self.qvalues = dict()
        self.model = dict()
        self.plan_steps = plan_steps
        self.kappa = kappa
        self.queue = []
        self.theta = 0.00001
        self.count = 0

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        if state in self.qvalues:
            return self.qvalues[state][action]
        else:
            return 0.0


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return 0.0
        return max(self.getQValue(state, a) for a in legalActions)

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None
        if state not in self.qvalues:
            return random.choice(legalActions)
        return max(legalActions, key=lambda a : self.getQValue(state, a))

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        if not legalActions:
            # Terminal state
            return None
        elif util.flipCoin(self.epsilon):
            # Pick random action
            return random.choice(legalActions)
        else:
            # Pick greedy action
            return self.computeActionFromQValues(state)
        return action


    def update_ps(self, state, action, nextState, reward):
        # Prioritized Sweeping
        if state not in self.qvalues:
            self.qvalues[state] = util.Counter()
            self.model[state] = dict()

        self.model[state][action] = (reward, nextState)
        qsa = self.qvalues[state][action]
        new_qsa = self.computeValueFromQValues(nextState)
        p = abs(self.alpha * (reward + self.discount * new_qsa - qsa))
        if p > self.theta:
            heapq.heappush(self.queue, (p, state, action))

        for i in range(self.plan_steps):
            if not self.queue:
                break
            (_, state, action) = heapq.heappop(self.queue)
            (reward, nextState) = self.model[state][action]
            qsa = self.qvalues[state][action]
            new_qsa = self.computeValueFromQValues(nextState)
            self.qvalues[state][action] += self.alpha * (reward + self.discount * new_qsa - qsa)
            for p_state in self.model.keys():
                for p_action in self.model[p_state].keys():
                    (p_reward, p_nextState) = self.model[p_state][p_action]
                    if p_nextState != state:
                        continue
                    p_qsa = self.computeValueFromQValues(nextState)
                    p = abs(self.alpha * (p_reward + self.discount * new_qsa - qsa))
                    if p > self.theta:
                        heapq.heappush(self.queue, (p, state, action))


    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here.

          NOTE: You should never call this function,
          it will be called on your behalf

          NOTE2: insert your planning code here as well
        """
        if state not in self.qvalues:
          self.qvalues[state] = util.Counter()
          self.model[state] = dict()

        qsa = self.getQValue(state, action)
        new_qsa = self.computeValueFromQValues(nextState)
        self.qvalues[state][action] += self.alpha * (reward + self.discount * new_qsa - qsa)
        self.model[state][action] = (reward, nextState, self.count)

        for i in range(self.plan_steps):
            state = random.choice(self.model.keys())
            # Random state
            if self.kappa > 0:
                action = random.choice(self.getLegalActions(state))
            else:
                action = random.choice(self.model[state].keys())
            # Random action
            if action in self.model[state]:
                (reward, nextState, time) = self.model[state][action]
            else:
                (reward, nextState, time) = (0, 0, state)
            # Update
            if not nextState:
                continue
            qsa = self.qvalues[state][action]
            new_qsa = self.computeValueFromQValues(nextState)
            bonus = self.kappa * math.sqrt(self.count - time)
            self.qvalues[state][action] += self.alpha * (reward + bonus + self.discount * new_qsa - qsa)

        self.count += 1


    def update_ps(self, state, action, nextState, reward):
        # Prioritized Sweeping
        if state not in self.qvalues:
            self.qvalues[state] = util.Counter()
            self.model[state] = dict()

        self.model[state][action] = (reward, nextState)
        qsa = self.qvalues[state][action]
        new_qsa = self.computeValueFromQValues(nextState)
        p = abs(self.alpha * (reward + self.discount * new_qsa - qsa))
        if p > self.theta:
            heapq.heappush(self.queue, (p, state, action))

        for i in range(self.plan_steps):
            if not self.queue:
                break
            (_, state, action) = heapq.heappop(self.queue)
            (reward, nextState) = self.model[state][action]
            qsa = self.qvalues[state][action]
            new_qsa = self.computeValueFromQValues(nextState)
            self.qvalues[state][action] += self.alpha * (reward + self.discount * new_qsa - qsa)
            for p_state in self.model.keys():
                for p_action in self.model[p_state].keys():
                    (p_reward, p_nextState) = self.model[p_state][p_action]
                    if p_nextState != state:
                        continue
                    p_qsa = self.computeValueFromQValues(nextState)
                    p = abs(self.alpha * (p_reward + self.discount * new_qsa - qsa))
                    if p > self.theta:
                        heapq.heappush(self.queue, (p, state, action))


    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanDynaQAgent(DynaQAgent):
    "Exactly the same as DynaAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanDynaAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        DynaQAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of DynaAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = DynaQAgent.getAction(self, state)
        self.doAction(state,action)
        return action
