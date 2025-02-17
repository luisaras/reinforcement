# sarsaAgents.py
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
# SARSA Agent extension by Anderson Tavares (anderson@dcc.ufmg.br)

#EPISODE 2000 COMPLETE: RETURN WAS 0.84
#AVERAGE RETURNS FROM START STATE: 0.9938

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
import collections

import random, util, math

class SarsaAgent(ReinforcementAgent):
    """
      Sarsa Agent
      run with: python gridworld.py -a s -k 100
      (any gridworld run with '-a s' will work, except for the manual agent)
      Useful options:
      --epsilon value
      --edecay value
      --lambda value


      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - computeAction
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, epsilon_decay=1, lamda=0, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        self.epsilon_decay = epsilon_decay
        self.lamda = lamda
        self.qvalues = dict()
        self.evalues = dict()
        self.nextAction = None

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
        return max(legalActions, key=lambda a : self.getQValue(state, a))

    def computeAction(self, state):
        """
          Compute the action to take in the given state.  With
          probability self.epsilon, take a random action and
          take the greedy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, it
          chooses None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
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

    def getAction(self, state):
        """
          Returns the action computed in computeAction
        """
        if not self.nextAction:
          self.nextAction = self.computeAction(state)
        return self.nextAction

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        if state not in self.qvalues:
          self.qvalues[state] = util.Counter()
          self.evalues[state] = util.Counter()

        self.nextAction = self.computeAction(nextState)
        qsa = self.qvalues[state][action]
        new_qsa = self.getQValue(nextState, self.nextAction)
        delta = reward + self.discount * new_qsa - qsa

        self.evalues[state][action] += 1
        for state in self.evalues:
          for action in self.evalues[state]:
            self.qvalues[state][action] += self.alpha * delta * self.evalues[state][action]
            self.evalues[state][action] *= self.discount * self.lamda
          for a, v in self.evalues[state].items():
            if v < 0.000001:
              del self.evalues[state][a]

        self.epsilon *= self.epsilon_decay


    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanSarsaAgent(SarsaAgent):
    "Exactly the same as SarsaAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanSarsaAgent -a epsilon=0.1

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
        SarsaAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of SarsaAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = SarsaAgent.getAction(self, state)
        self.doAction(state,action)
        return action


class ApproximateSarsaAgent(PacmanSarsaAgent):
    """
       ApproximateSarsaAgent

       You should only have to overwrite getQValue
       and update.  All other SarsaAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanSarsaAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        features = self.featExtractor.getFeatures(state, action)
        weights = self.getWeights()
        return sum( [ weights[feature] * value for feature, value in features.items() ] )

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        self.nextAction = self.computeAction(nextState)
        features = self.featExtractor.getFeatures(state, action)
        weights = self.getWeights()
        delta = reward + self.discount * self.getQValue(nextState, self.nextAction) - self.getQValue(state, action)
        for feature, value in features.items():
          weights[feature] += self.alpha * delta * value

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanSarsaAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            print(len(self.weights.items()))
            for i in self.weights.items():
              print("%s = %s" % i)