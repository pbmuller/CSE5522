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
        oldPellets = currentGameState.getCapsules()
        newPellets = successorGameState.getCapsules()

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
        
        closestDist = float('inf')
        for food in foodList:
          dist = util.manhattanDistance(food, newPos)
          if dist < closestDist:
            closestDist = dist
        if closestDist == float('inf'):
          #this means that the move we make eats the last dot
          score = 1000
        else:
          if (10 - closestDist) > 0:
            score += (10 - closestDist)

        for ghostState in newGhostStates:
          ghostPos = ghostState.getPosition()
          if ghostPos == newPos:
            score += -1000
          else:
            for dx in directions:
              for dy in directions:
                if (ghostPos[0]+dx, ghostPos[1]+dy) == newPos:
                  score += -500

        return score + (100 * (len(oldPellets) - len(newPellets)))

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
      if gameState.isWin() or gameState.isLose() or depth == 0:
        return self.evaluationFunction(gameState), None
      legalActions = gameState.getLegalActions(agentIndex)
      bestScore = float('-inf')
      bestAction = legalActions[0]
      nextAgent = (agentIndex + 1) % gameState.getNumAgents()
      for legalAction in legalActions:
        successor = gameState.generateSuccessor(agentIndex, legalAction)
        score, action = self.min_value(nextAgent, depth, successor)
        bestScore = max(score, bestScore)
        if score == bestScore:
          bestAction = legalAction
      return bestScore, bestAction

    
    def min_value(self, agentIndex, depth, gameState):
      if gameState.isWin() or gameState.isLose(): # or (depth == 1 and agentIndex + 1 == gameState.getNumAgents()):
        return self.evaluationFunction(gameState), None

      legalActions = gameState.getLegalActions(agentIndex)
      bestScore = float('inf')
      nextAgent = (agentIndex + 1) % gameState.getNumAgents()
      for legalAction in legalActions:
        successor = gameState.generateSuccessor(agentIndex, legalAction)
        if nextAgent == 0:
          score, action = self.max_value(nextAgent, depth - 1, successor)
        else:
          score, action = self.min_value(nextAgent, depth, successor)
        bestScore = min(score, bestScore)
      return bestScore, None


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
        agentIndex = 0
        return self.max_value(agentIndex, self.depth, gameState)[1]

        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def max_value(self, agentIndex, depth, gameState, alpha, beta):
      if gameState.isWin() or gameState.isLose() or depth == 0:
        return self.evaluationFunction(gameState), None
      legalActions = gameState.getLegalActions(agentIndex)
      bestScore = float('-inf')
      bestAction = legalActions[0]
      nextAgent = (agentIndex + 1) % gameState.getNumAgents()
      for legalAction in legalActions:
        successor = gameState.generateSuccessor(agentIndex, legalAction)
        score, action = self.min_value(nextAgent, depth, successor, alpha, beta)
        bestScore = max(score, bestScore)
        if score == bestScore:
          bestAction = legalAction
        if bestScore > beta:
          return bestScore, bestAction
        alpha = max(alpha, bestScore)
      return bestScore, bestAction

    
    def min_value(self, agentIndex, depth, gameState, alpha, beta):
      if gameState.isWin() or gameState.isLose(): # or (depth == 1 and agentIndex + 1 == gameState.getNumAgents()):
        return self.evaluationFunction(gameState), None

      legalActions = gameState.getLegalActions(agentIndex)
      bestScore = float('inf')
      nextAgent = (agentIndex + 1) % gameState.getNumAgents()
      for legalAction in legalActions:
        successor = gameState.generateSuccessor(agentIndex, legalAction)
        if nextAgent == 0:
          score, action = self.max_value(nextAgent, depth - 1, successor, alpha, beta)
        else:
          score, action = self.min_value(nextAgent, depth, successor, alpha, beta)
        bestScore = min(score, bestScore)
        if bestScore < alpha:
          return bestScore, None
        beta = min(beta, bestScore)
      return bestScore, None

    def getAction(self, gameState):
      agentIndex = 0
      alpha = float('-inf')
      beta = float('inf')
      return self.max_value(agentIndex, self.depth, gameState, alpha, beta)[1]

      util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def max_value(self, agentIndex, depth, gameState):
      if gameState.isWin() or gameState.isLose() or depth == 0:
        return float(self.evaluationFunction(gameState)), None
      legalActions = gameState.getLegalActions(agentIndex)
      bestScore = float('-inf')
      bestAction = legalActions[0]
      nextAgent = (agentIndex + 1) % gameState.getNumAgents()
      for legalAction in legalActions:
        successor = gameState.generateSuccessor(agentIndex, legalAction)
        score, action = self.exp_value(nextAgent, depth, successor)
        bestScore = max(score, bestScore)
        if score == bestScore:
          bestAction = legalAction
      return bestScore, bestAction

    
    def exp_value(self, agentIndex, depth, gameState):
      if gameState.isWin() or gameState.isLose(): # or (depth == 1 and agentIndex + 1 == gameState.getNumAgents()):
        return float(self.evaluationFunction(gameState)), None

      legalActions = gameState.getLegalActions(agentIndex)
      scores = []
      nextAgent = (agentIndex + 1) % gameState.getNumAgents()
      for legalAction in legalActions:
        successor = gameState.generateSuccessor(agentIndex, legalAction)
        if nextAgent == 0:
          score, action = self.max_value(nextAgent, depth - 1, successor)
        else:
          score, action = self.exp_value(nextAgent, depth, successor)
        scores.append(score)
      return sum(scores)/len(scores), None


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
        agentIndex = 0
        return self.max_value(agentIndex, self.depth, gameState)[1]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: This evaluation function wound up being very similar to the reflex agent minus. 
      First, check if the state is terminal. I'm not actually sure if this condition will ever pass because I
      check the condition in my search methods, but better safe than sorry.

      I decided that the most important thing in the game was to eat all of the food and that it wouldn't be worth
      time to chase after pellets or ghosts unless they were very close by.

      I use the score function as the base for my evaluation function and will add up to 10 to find the nearest food, since eat
      ing a food increases the score by 10.

      I then find the closest ghost. If they are more not within our immediate vicinity we don't care about them. If they are
      right next to use, then we avoid them if they are not scared, and eat them if they are scared. 

      lastly, subtract the score by the number of remaining pellets * 100 (I'm not sure if any of the numbers I chose for these 
      constants matter a whole lot, but they are what I decided to go with). Subtracting pellets from the score means that we will
      always eat a pellet if we have the chance, but won't worry about hunting them down if we don't have the chance.
    """
    pos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    pellets = currentGameState.getCapsules()
    #moving onto a food pellet is preferable to not moving onto a food pellet
    #moving onto a line of food pellets is even more preferable
    #being far way from a ghost is preferable
    #unless the ghost is scared, then being close to the ghost is preferable because we might eat
    "*** YOUR CODE HERE ***"
    directions = [-1, 0, 1]
    #moving decreases the score by 1
    #eating increase the score by 10
    # score = 10 * (oldFood[newPos[0]][newPos[1]] or newFood[newPos[0]][newPos[1]]) #start off with a score of 10 if we just ate a food pellet
    if currentGameState.isWin():
      return float('inf')
    elif currentGameState.isLose():
      return float('-inf')

    closestDist = float('inf')
    score = currentGameState.getScore()
    for food in foodList:
      dist = util.manhattanDistance(food, pos)
      if dist < closestDist:
        closestDist = dist
    if (10 - closestDist) > 0:
      score += (10 - closestDist)

    for ghostState in ghostStates:
      ghostPos = ghostState.getPosition()
      for dx in directions:
        for dy in directions:
          if (ghostPos[0]+dx, ghostPos[1]+dy) == pos:
            if ghostState.scaredTimer == 0:
              score += -500
            else:
              score += 500 

    score -= 100 * len(pellets)
    return score
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

