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

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def getMinFoodDistance(self, gameState):
        """
        Given a game state, returns the minimum distance between pacman and food.
        Arguments:
            gameState: Instance of game state.
        Returns:
            float: Minimum distance(-1 indicates no foods).
        """
        allFoods = gameState.getFood().asList()
        pacPos = gameState.getPacmanPosition()
        minDistance = float(min([manhattanDistance(foodPos, pacPos) for foodPos in allFoods] if allFoods else [-1]))
        return minDistance

    def getTotalFoodCount(self, gameState):
        """
        Given a game state, returns the total number of foods.
        Arguments:
            gameState: Instance of game state.
        Returns:
            int: Number of foods.
        """
        return gameState.getFood().count()

    def getMinCapsuleDistance(self, gameState):
        """
        Given a game state, returns the minimum distance between pacman and capsule.
        Arguments:
            gameState: Instance of game state.
        Returns:
            float: Minimum distance(-1 indicates no capsules).
        """
        capsules = gameState.getCapsules()
        pacPos = gameState.getPacmanPosition()
        minDistance = float(min([manhattanDistance(capsulePos, pacPos) for capsulePos in capsules] if capsules else [-1]))
        return minDistance

    def getTotalCapsuleCount(self, gameState):
        """
        Given a game state, returns the total number of capsules.
        Arguments:
            gameState: Instance of game state.
        Returns:
            int: Number of capsules.
        """
        return len(gameState.getCapsules())

    def getMinGhostDistance(self, gameState):
        """
        Given a game state, returns the minimum distance between pacman and ghost and its scared time.
        Arguments:
            gameState: Instance of game state.
        Returns:
            float: Maximum scared time(-1 indicates no scared ghosts).
            float: Minimum distance.
        """
        ghostStates = gameState.getGhostStates()
        scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
        ghostPositions = [ghostState.getPosition() for ghostState in ghostStates]
        pacPos = gameState.getPacmanPosition()
        maxScared, minDistance = -1, float("inf")
        for scared, ghostPos in zip(scaredTimes, ghostPositions):
            if scared > 0 and scared > maxScared:
                maxScared = scared
                minDistance = float(manhattanDistance(ghostPos, pacPos))
            else:
                minDistance = float(min(minDistance, manhattanDistance(ghostPos, pacPos)))
        return maxScared, minDistance

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
        newFood = successorGameState.getFood().asList()
        # 10 points for every food you eat 
        """
        Returns a Grid of boolean food indicator variables.

        Grids can be accessed via list notation, so to check
        if there is food at (x,y), just call

        currentFood = state.getFood()
        if currentFood[x][y] == True: ...
        """
        newCapsule = successorGameState.getCapsules()
        # 200 points for every ghost you eat
        # but no point for capsule

        # For Ghost
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        # Position of ghost do not change regardless of your state 
        # because you can't predict the future
        ghostPositions = [ghostState.getPosition() for ghostState in newGhostStates]
        # Count down from 40 moves
        ghostStartPos = [ghostState.start.getPosition() for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"
        # Minimum Manhattan Distance between food and pacman(new GameState).
        newMinFoodDistance = self.getMinFoodDistance(successorGameState)
        # Total food count(current / new GameState).
        newFoodCount = self.getTotalFoodCount(successorGameState)
        foodCount = self.getTotalFoodCount(currentGameState)


        # Minimum Manhattan Distance between capsule and pacman(new GameState).
        newMinCapsuleDistance = self.getMinCapsuleDistance(successorGameState)
        # Total capsule count(current / new GameState).
        newCapsuleCount = self.getTotalCapsuleCount(successorGameState)
        capsuleCount = self.getTotalCapsuleCount(currentGameState)

        # Minimum Manhattan Distance between ghost and pacman(new GameState).
        scared, newMinDistanceGhost = self.getMinGhostDistance(successorGameState)

        # There is no food. => Win
        if newFoodCount == 0:
            return float("inf")

        """ Consider the minimum food distance. """
        if newFoodCount == foodCount:           # In this action pacman won't eat the food.
            dis = newMinFoodDistance
        else:                                   # In this action pacman will eat the food.
            dis = 0

        """ Consider the minimum capsule distance if there exists capsule. """
        cap = 0
        if capsuleCount != 0:
            if newCapsuleCount == capsuleCount: # In this action pacman won't eat the capsule.
                cap = newMinCapsuleDistance
            else:                               # In this action pacman will eat the capsule.
                cap = 0

        """ Consider the minimum ghost distance whether the ghost is scared(For capsule). """
        if scared == -1:                        # Ghost isn't scared.
            cap += 4 ** (2 - newMinDistanceGhost)
        else:                                   # Ghost is scared.
            cap -= 8 ** abs((8 / (-newMinDistanceGhost)))

        """ Consider the minimum ghost distance(For food). """
        for ghostPos in ghostPositions:         # Distance between ghosts and pacman.
            dis += 4 ** (2 - manhattanDistance(ghostPos, newPos))

        return (-0.5 * dis) - (1.5 * cap)       # Weighted sum


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
    """ Pseudo Code(Revised from Wikipedia) """
    """
    function minimax(node, depth, player):
        player = player mod total agents
        /* Terminate. */
        if node is a terminal node or depth = max_depth
            return the heuristic value of node
        /* Pacman. */
        if player is 0
            alpha := -inf
            foreach child of node
                if next player is 0     /* Pacman. */
                    max_alpha := max(alpha, minimax(child, depth+1, player+1))
                else                    /* Another ghost. */
                    max_alpha := max(alpha, minimax(child, depth, player+1))
                alpha := max(alpha, max_alpha)
        /* Ghosts. */
        else
            alpha := -inf
            foreach child of node
                if next player is 0     /* Pacman. */
                    min_alpha := min(alpha, minimax(child, depth+1, player+1))
                else                    /* Another ghost. */
                    min_alpha := min(alpha, minimax(child, depth, player+1))
                alpha := min(alpha, min_alpha)
        return alpha(heuristic value)
    """

    def minimax(self, state, depth, player=0):
        player = player % state.getNumAgents()              # Current player index(0: Pacman, other: Ghosts)
        legalActions = state.getLegalActions(player)        # Legal action list of player.

        """ Depth reaches the depth or in the terminal state(win / lose). """
        if depth == self.depth or state.isWin() or state.isLose():
            return state.getScore(), ""
        action = None                                       # action will return.

        if player == 0:                                     # Pacman turn.
            alpha = float("-inf")                           # Select maximum from children states.
            for act in legalActions:
                successorState = state.generateSuccessor(player, act)
                if (player + 1) == state.getNumAgents():    # Next round is still Pacman.
                    heuristic, _ = self.minimax(successorState, depth+1, player+1)
                else:                                       # Next round is Ghost turn.
                    heuristic, _ = self.minimax(successorState, depth, player+1)
                if heuristic > alpha:                       # Updates the maximum heuristic value and corresponding action.
                    alpha = heuristic
                    action = act
        else:                                               # Ghosts turn.
            alpha = float("inf")                            # Select minimum from children states.
            for act in legalActions:
                successorState = state.generateSuccessor(player, act)
                if (player + 1) == state.getNumAgents():    # Next round is still Pacman.
                    heuristic, _ = self.minimax(successorState, depth+1, player+1)
                else:                                       # Next round is Ghost turn.
                    heuristic, _ = self.minimax(successorState, depth, player+1)
                if heuristic < alpha:                       # Updates the minimum heuristic value and corresponding action.
                    alpha = heuristic
                    action = act
        return alpha, action

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
        "*** YOUR CODE HERE ***"
        heuristic, action = self.minimax(gameState, 0, 0)   # Current depth is 0, and player is 0 (Pacman).
        return action
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    """ Pseudo Code(Revised from Wikipedia) """
    """
    function alphabeta(node, depth, alpha, beta, player):
        player = player mod total agents
        /* Terminate. */
        if node is a terminal node or depth = max_depth
            return the heuristic value of node
        /* Pacman. */
        if player is 0
            max_alpha := -inf
            for each child of node      /* Minimum children nodes. */
                if next player is 0     /* Pacman. */
                    max_alpha := max(max_alpha, minimax(child, depth+1, alpha, beta, player+1))
                else                    /* Another ghost. */
                    max_alpha := max(max_alpha, minimax(child, depth, alpha, beta, player+1))
                if beta < max_alpha     /* The rest of children nodes wouldn't be selected by upper minimum node. */
                    break               /* Beta cut-off. */
                alpha := max(alpha, max_alpha)
            return max_alpha
        /* Ghosts. */
        else
            min_beta := +inf
            for each child of node      /* Maximum children nodes. */
                if next player is 0     /* Pacman. */
                    min_beta := min(min_beta, minimax(child, depth+1, alpha, beta, player+1))
                else                    /* Another ghost. */
                    min_beta := min(min_beta, minimax(child, depth, alpha, beta, player+1))
                if min_beta < alpha     /* The rest of children nodes wouldn't be selected by upper maximum node. */
                    break               /* Alpha cut-off. */
                beta := min(beta, min_beta)
            return min_beta
    /* Initial call. */
    alphabeta(origin, depth, -inf, +inf, player=0)
    """

    def alphabeta(self, state, depth, alpha, beta, player=0):
        player = player % state.getNumAgents()              # Current player index(0: Pacman, other: Ghosts)
        legalActions = state.getLegalActions(player)        # Legal action list of player.

        """ Depth reaches the depth or in the terminal state(win / lose). """
        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state), ""
        action = None                                       # action will return.

        if player == 0:                                     # Pacman turn.
            currentAlpha = float("-inf")                    # alpha value of pacman.
            for act in legalActions:
                successorState = state.generateSuccessor(player, act)
                if (player + 1) == state.getNumAgents():    # Next round is still Pacman.
                    heuristic, _ = self.alphabeta(successorState, depth+1, alpha, beta, player+1)
                else:                                       # Next round is Ghost turn.
                    heuristic, _ = self.alphabeta(successorState, depth, alpha, beta, player+1)
                if heuristic > currentAlpha:                # Updates alpha value of pacman and corresponding action.
                    currentAlpha = heuristic
                    action = act
                if beta < currentAlpha:                     # beta cut-off
                    break
                alpha = max(alpha, currentAlpha)            # Updates alpha value.
            return currentAlpha, action
        else:                                               # Ghosts turn.
            currentBeta = float("inf")                      # beta value of ghost.
            for act in legalActions:
                successorState = state.generateSuccessor(player, act)
                if (player + 1) == state.getNumAgents():    # Next round is still Pacman.
                    heuristic, _ = self.alphabeta(successorState, depth+1, alpha, beta, player+1)
                else:                                       # Next round is Ghost turn.
                    heuristic, _ = self.alphabeta(successorState, depth, alpha, beta, player+1)
                if heuristic < currentBeta:                 # Updates beta value of pacman and corresponding action.
                    currentBeta = heuristic
                    action = act
                if currentBeta < alpha:                     # alpha cut-off
                    break
                beta = min(beta, currentBeta)
            return currentBeta, action

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        heuristic, action = self.alphabeta(gameState, 0, float("-inf"), float("inf"), 0)
        return action
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    """ Pseudo Code(Revised from Wikipedia) """
    """
    function expectimax(node, depth, player)
        player = player mod total agents
        /* Terminate. */
        if node is a terminal node or depth = max_depth
            return the heuristic value of node
        /* Pacman. */
        if player is 0
            alpha := -inf
            foreach child of node
                if next player is 0     /* Pacman. */
                    max_alpha := max(alpha, minimax(child, depth+1, player+1))
                else                    /* Another ghost. */
                    max_alpha := max(alpha, minimax(child, depth, player+1))
                alpha := max(alpha, max_alpha)
        /* Ghosts. */
        else
            alpha := 0
            foreach child of node
                if next player is 0     /* Pacman. */
                    alpha := alpha + min(alpha, minimax(child, depth+1, player+1))
                else                    /* Another ghost. */
                    alpha := alpha + min(alpha, minimax(child, depth, player+1))
                alpha := Mean of alpha
        return alpha(heuristic value)
    """

    def expectimax(self, state, depth, player=0):
        player = player % state.getNumAgents()              # Current player index(0: Pacman, other: Ghosts)
        legalActions = state.getLegalActions(player)        # Legal action list of player.

        """ Depth reaches the depth or in the terminal state(win / lose). """
        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state), ""
        action = None                                       # action will return.
        if player == 0:                                     # Pacman turn.
            alpha = float("-inf")
            for act in legalActions:
                successorState = state.generateSuccessor(player, act)
                if (player + 1) == state.getNumAgents():    # Next round is still Pacman.
                    heuristic, _ = self.expectimax(successorState, depth+1, player+1)
                else:                                       # Next round is Ghost turn.
                    heuristic, _ = self.expectimax(successorState, depth, player+1)
                if heuristic > alpha:
                    alpha = heuristic
                    action = act
        else:                                               # Ghosts turn.
            alpha = []
            for act in legalActions:
                successorState = state.generateSuccessor(player, act)
                if (player + 1) == state.getNumAgents():    # Next round is still Pacman.
                    heuristic, _ = self.expectimax(successorState, depth+1, player+1)
                else:                                       # Next round is Ghost turn.
                    heuristic, _ = self.expectimax(successorState, depth, player+1)
                alpha.append(float(heuristic))
            alpha = sum(alpha) / len(alpha)                 # Average alpha
        return alpha, action

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        heuristic, action = self.expectimax(gameState, 0, player=0)
        return action
        util.raiseNotDefined()