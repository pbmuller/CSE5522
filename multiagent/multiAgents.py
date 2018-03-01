# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        # print(legalMoves)
        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        # print("best indices: {}".format(bestIndices))
        # print("best scores: {}".format(bestScore))
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        # raw_input("chosenIndex: {}".format(chosenIndex))

        """Add more of your code here if you want to"""

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        oldFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        #moving onto a food pellet is preferable to not moving onto a food pellet
        #moving onto a line of food pellets is even more preferable
        #being far way from a ghost is preferable
        #unless the ghost is scared, then being close to the ghost is preferable because we might eat
        "*** YOUR CODE HERE ***"
        directions = [-1, 0, 1]
        foodList = newFood.asList()
        #moving decreases the score by 1
        #eating increase the score by 10
        score = 10 * (oldFood[newPos[0]][newPos[1]] or newFood[newPos[0]][newPos[1]]) #start off with a score of 10 if we just ate a food pellet
        
        closestDist = 999999
        for food in foodList:
          dist = abs(food[0] - newPos[0]) + abs(food[1] - newPos[1])
          if dist < closestDist:
            closestDist = dist
        if closestDist == 999999:
          #this means that the move we make eats the last dot
          score = 1000
        else:
          score += (10 - closestDist)

        for ghostState in newGhostStates:
          ghostPos = ghostState.getPosition()
          if ghostPos == newPos:
            score = -1000
          else:
            for dx in directions:
              for dy in directions:
                if (ghostPos[0]+dx, ghostPos[1]+dy) == newPos:
                  score = -500

        return score

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """
    def max_value(self, agentIndex, depth, gameState):
      #don't really need to know the agentIndex because Pac-man is the only agent we will be trying to maximize
      maxValue = None
      maxAction = None
      for action in gameState.getLegalActions(agentIndex):
        successorValue = self.value(agentIndex, depth, gameState.generateSuccessor(agentIndex, action))
        maxValue = max(maxValue, successorValue)
        if maxValue == successorValue:
          maxAction = action
      return (maxValue, maxAction)
    
    def min_value(self, agentIndex, depth, gameState):
      value = None
      for action in gameState.getLegalActions(agentIndex):
        value = min(value, self.value(agentIndex, depth, gameState.generateSuccessor(agentIndex, action)))
      return value

    def value(self, agentIndex, depth, gameState):
      nextAgent = ((agentIndex + 1) % gameState.getNumAgents())

      #check if the game is terminal or if the depth has been reached
      if gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)
      elif depth == 0 and nextAgent == 0:
        return self.evaluationFunction(gameState)
      else:
        if nextAgent == 0:
          return self.max_value(nextAgent, depth - 1, gameState)[0]
        else:
          return self.min_value(nextAgent, depth, gameState)


    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """

        #return the best minimax value
        bestAction = self.max_value(0, self.depth, gameState)[1]
        return bestAction

        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

