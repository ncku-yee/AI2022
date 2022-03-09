# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    """ Pseudo Code """
    """
    Set all nodes to "not visited";
    s = new Stack();
    s.push(initial node with priority);
    while (s != empty) do {
        x = s.pop();
        if (x has not been visited) {
            visited[x] = true;          /* Visit node x. */
            for (every edge (x, y))     /* Using all edges. */
                if (y has not been visited)   
                    s.pop(y);           /* Use the edge (x, y). */
        }
    }
    """
    "*** YOUR CODE HERE ***"
    curr = {'node': problem.getStartState()}    # Current point's information.
    if problem.isGoalState(curr['node']):       # Current point is goal state.
        return []
    stack = util.Stack()                        # Record the point should be visited.
    stack.push(curr)                            # Push the starting point the stack.
    visited = []                                # Record the point has been visited(No point is visited).

    while not stack.isEmpty():
        curr = stack.pop()                      # Current point's information.

        if curr['node'] not in visited:         # Current point has not been visited.
            visited.append(curr['node'])        # Visit the current point.

            # Current point is goal state.
            if problem.isGoalState(curr['node']):
                return getActionList(curr)

            # Get the successors of current point.
            for successor, action, cost in problem.getSuccessors(curr['node']):

                # Successor hasn't been visited, push it into the stack that would be visited.
                if successor not in visited:
                    stack.push({'node': successor, 'action': action, 'parent_node': curr})

    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """ Search the shallowest nodes in the search tree first. """
    """ Pseudo Code """
    """
    Set all nodes to "not visited";
    q = new Queue();
    q.enqueue(initial node);
    while (q != empty) do {
        x = q.dequeue();
        if (x has not been visited) {
            visited[x] = true;          /* Visit node x. */
            for (every edge (x, y))     /* Using all edges. */
                if (y has not been visited)   
                    q.enqueue(y);       /* Use the edge (x, y). */
        }
    }
    """
    "*** YOUR CODE HERE ***"
    curr = {'node': problem.getStartState()}    # Current point' information.
    if problem.isGoalState(curr['node']):       # Current point is goal state.
        return []
    queue = util.Queue()                        # Record the point should be visited.
    queue.push(curr)                            # Push the starting point the queue.
    visited = []                                # Record the point has been visited(No point is visited).

    while not queue.isEmpty():
        curr = queue.pop()                      # Current point's information.

        if curr['node'] not in visited:         # Current point has not been visited.
            visited.append(curr['node'])        # Visit the current point.

            # Current point is goal state.
            if problem.isGoalState(curr['node']):
                return getActionList(curr)

            # Get the successors of current point.
            for successor, action, cost in problem.getSuccessors(curr['node']):

                # Successor hasn't been visited, push it into the queue that would be visited.
                if successor not in visited:
                    queue.push({'node': successor, 'action': action, 'parent_node': curr})

    util.raiseNotDefined()

def uniformCostSearch(problem):
    """ Search the node of least total cost first. """
    """ Pseudo Code """
    """
    Set all nodes to "not visited";
    q = new PriorityQueue();
    q.enqueue(initial node with priority, 0);   /* Enqueue the initial node with priority(cost) 0. */
    while (q != empty) do {
        x = q.dequeue();                        /* Get the lowest-priority(cost) item. */
        if (x has not been visited) {
            visited[x] = true;                  /* Visit node x. */
            for (every edge (x, y))             /* Using all edges. */
                newCost = cost from initial node to y;
                q.update(y, newCost);           /* Update the with priority(cost) newCost. */
        }
    }
    """
    "*** YOUR CODE HERE ***"
    curr = {'node': problem.getStartState(), 'cost': 0}     # Current point's information.
    if problem.isGoalState(curr['node']):                   # Current point is goal state.
        return []
    priority_queue = util.PriorityQueue()                   # Record the point should be visited.
    priority_queue.push(curr, 0)                            # Push the starting point into the priority queue with cost 0.
    visited = []                                            # Record the point has been visited(No point is visited).
    
    while not priority_queue.isEmpty():
        curr = priority_queue.pop()

        if curr['node'] not in visited:                     # Current point has not been visited.
            visited.append(curr['node'])                    # Visit the current point.

            # Current point is goal state.
            if problem.isGoalState(curr['node']):
                return getActionList(curr)

            # Get the successors of current point.
            for successor, action, cost in problem.getSuccessors(curr['node']):
                cost += curr['cost']                        # Total cost from starting point to successor point.

                # Update or push the point with cost as priority into the priority queue.
                priority_queue.update({'node': successor, 'action': action, 'parent_node': curr, 'cost': cost}, cost)

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """ Search the node that has the lowest combined cost and heuristic first. """
    """ Pseudo Code """
    """
    Set all nodes to "not visited";
    openList = new PriorityQueue();
    openList.enqueue(initial node with priority, 0);    /* Enqueue the initial node with priority(F) 0. */
    while (openList != empty) do {
        x = openList.dequeue();                         /* Get the lowest-priority(Smallest Evaluation) item. */
        if (x has not been visited) {
            visited[x] = true;                          /* Visit node x. */
            for (every edge (x, y))                     /* Using all edges. */
                H = Heuristic approximation;
                G = from initial node to y;
                F = G + H;
                openList.update(y, F);                  /* Update the with priority(F) F. */
        }
    }
    """
    "*** YOUR CODE HERE ***"
    curr = {'node': problem.getStartState(), 'G': 0, 'F': 0}    # Current point's information.
    if problem.isGoalState(curr['node']):                       # Current point is goal state.
        return []
    open_queue = util.PriorityQueue()                           # Record the point should be visited.
    open_queue.push(curr, 0)                                    # Push the starting point into the priority queue with cost 0.
    visited = []                                                # Record the point has been visited(No point is visited) acts as closed list.

    while not open_queue.isEmpty():
        curr = open_queue.pop()

        if curr['node'] not in visited:                         # Current point has not been visited.
            visited.append(curr['node'])                        # Visit the current point.

            # Current point is goal state.
            if problem.isGoalState(curr['node']):
                return getActionList(curr)

            for successor, action, cost in problem.getSuccessors(curr['node']):
                h = heuristic(successor, problem)               # Calculate the Heuristic approximation.
                g = cost + curr['G']                            # Cost from the starting point to successor point.
                f = g + h                                       # Evaluation(F(n) = G(n) + H(n))
                # Update or push the point with F as priority into the priority queue.
                open_queue.update({'node': successor, 'action': action, 'parent_node': curr, 'G': g, 'F': f}, f)

    util.raiseNotDefined()


def getActionList(goal):
    """
    Returns a list of actions from starting point to goal.
    Arguments:
        goal: A dicitonary stores the information(point, action, parent node, ...) about goal.
    Returns:
        list: A action list from starting point to goal.
    """
    actions = []                        # Action list.

    # Get parent node and the action from parent node to current node until parent node is starting point.
    while 'parent_node' in goal:
        actions.append(goal['action'])
        goal = goal['parent_node']

    actions.reverse()                   # Reverse the actions to get the answer.
    return actions

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
