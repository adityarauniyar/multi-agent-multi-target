import gym
from gym import spaces
from mdgym.envs.state import WorldState
from mdgym.utils.types import *
import numpy as np
from threading import Lock
import sys
from matplotlib.colors import hsv_to_rgb
import random
import math
import logging
from mdgym.utils.map_postprocessing import *
from mdgym.envs.objects.semantic_object import SemanticObject, ObjectType
from mdgym.envs.multi_drone_multi_actor.hawkins2DMap.ENV import *


ACTION_COST, IDLE_COST, GOAL_REWARD, COLLISION_REWARD, FINISH_REWARD, BLOCKING_COST = -0.3, -.5, 0.0, -2., 20., -1.

JOINT = False  # True for joint estimation of rewards for close-by agents

DIAGONAL_MOVEMENT = True

# TODO: Update this list of opposite actions
opposite_actions = {0: -1, 1: 3, 2: 4, 3: 1, 4: 2, 5: 7, 6: 8, 7: 5, 8: 6}

SEED = 123


def get_finish_reward():
    return FINISH_REWARD


def astar(world, start, goal, robots=[]):
    """robots is a list of robots to add to the world"""
    for (i, j) in robots:
        world[i, j] = 1
    try:
        # TODO: get the path from MStar using Networkx lib
        path = None
    except Exception as err:
        path = None
    for (i, j) in robots:
        world[i, j] = 0
    return path


class MUDMAFEnv(gym.Env):
    metadata = {"render.modes": ["human", "ansi"]}

    # Initialize env
    def __init__(
            self,
            num_agents=1,
            num_actors=1,
            observation_size=10,
            world0=None,
            world_png=None,
            grid_size=1.0,
            goals0=None,
            size=(10, 40),
            obstacle_prob_range=(0, .5),
            full_help=False,
            blank_world=False,
            SEED: int = 123
    ):
        """
        Args:
            size: size of a side of the square grid
            obstacle_prob_range: range of probabilities that a given block is an obstacle
            full_help
        """
        self.logger = logging.getLogger(__name__)

        self.logger.info(f"Starting the MUDMAFEnv with following parameters: num_agents = {num_agents},\n "
                         f"observation Size = {observation_size}, world size = {size}, grid size ={grid_size},\n "
                         f"goals length = {len(goals0) if goals0 is not None else 0}, obstacle prob= {obstacle_prob_range},\n"
                         f" full help = {full_help}, black world ={blank_world}\n")

        # Initialize member variables
        self.num_agents = num_agents
        self.num_actors = num_actors

        # a way of doing joint rewards
        self.individual_rewards = np.zeros(num_agents, dtype=float)
        self.observation_size = observation_size
        self.grid_size = grid_size
        self.SIZE = size
        self.PROB = obstacle_prob_range
        self.fresh = True
        self.FULL_HELP = full_help
        self.finished = False
        self.mutex = Lock()

        self.world_png = world_png

        # Initialize data structures
        self._set_world(world0, goals0, blank_world=blank_world, SEED=SEED)

        self.action_space = spaces.Tuple([spaces.Discrete(self.num_agents), spaces.Discrete(9)])

        self.viewer = None

    @staticmethod
    def is_connected(self, world0: np.ndarray | None) -> bool:
        sys.setrecursionlimit(10000)

        world0 = world0.copy()

        def first_free(new_world0):
            for x in range(new_world0.shape[0]):
                for y in range(new_world0.shape[1]):
                    if new_world0[x, y] == 0:
                        return x, y

        def floodfill(world, i, j):
            sx, sy = world.shape[0], world.shape[1]
            if i < 0 or i >= sx or j < 0 or j >= sy:  # out of bounds, return
                return
            if world[i, j] == -1:
                return
            world[i, j] = -1
            floodfill(world, i + 1, j)
            floodfill(world, i, j + 1)
            floodfill(world, i - 1, j)
            floodfill(world, i, j - 1)

        i, j = first_free(world0)
        floodfill(world0, i, j)
        if np.any(world0 == 0):
            return False
        else:
            return True

    def get_obstacle_map(self):
        return (self.world.agents_state == 0).astype(int)

    def get_actors_position(self):
        result = []
        for i in range(self.num_actors):
            result.append(self.world.get_actor_position_by_id(i))
        return result

    def get_agent_positions(self):
        result = []
        for i in range(self.num_actors):
            result.append(self.world.get_agents_position_by_id(i))
        return result

    def _set_world(self, obstacle_map0=None, actors0_start_pos=None, blank_world=False, SEED: int = 123):
        self.logger.info(f"Trying to set world now...")

        self.logger.debug(f"Setting seed as {SEED} for the environment...")
        np.random.seed(SEED)

        # blank_world is a flag indicating that the world given has no agent or goal positions
        def get_connected_region(obstacle_map: np.ndarray, regions_dict, x: int, y: int):

            sys.setrecursionlimit(1000000)
            '''returns a list of tuples of connected squares to the given tile
            this is memoized with a dict'''
            if (x, y) in regions_dict:
                return regions_dict[(x, y)]
            visited = set()
            sx, sy = obstacle_map.shape[0], obstacle_map.shape[1]
            work_list = [(x, y)]
            while len(work_list) > 0:
                (i, j) = work_list.pop()
                if i < 0 or i >= sx or j < 0 or j >= sy:  # out of bounds, return
                    continue
                if obstacle_map[i, j] == 1:
                    continue  # crashes
                if obstacle_map[i, j] > 0:
                    regions_dict[(i, j)] = visited
                if (i, j) in visited:
                    continue
                visited.add((i, j))
                work_list.append((i + 1, j))
                work_list.append((i, j + 1))
                work_list.append((i - 1, j))
                work_list.append((i, j - 1))
            regions_dict[(x, y)] = visited
            return visited

        # defines the State object, which includes initializing goals and agents
        # sets the world to world0 and goals, or if they are None randomizes world
        self.logger.debug(f"Obstacle status: {obstacle_map0}, World Png = {self.world_png}")
        if obstacle_map0 is not None or self.world_png is not None:
            self.logger.info("World is not none and extracting the start and goal location ...")
            if actors0_start_pos is None and not blank_world:
                raise Exception("you gave a world with no goals!")

            if obstacle_map0 is None and self.world_png is not None:
                self.OperationalSemanticObject = SemanticObject(ObjectType.OBSTACLE, self.world_png, OPERATIONAL_RGB_VALUE, 1)

                self.obstacle_map0 = np.where(self.OperationalSemanticObject.presence_grid == 1, 0, 1)
                self.logger.debug(f"Obstacle map assigned from the given image: \n{np.shape(self.obstacle_map0)}")

            if blank_world:
                world = self.obstacle_map0

                # RANDOMIZE THE POSITIONS OF AGENTS/DRONE
                agent_counter = 1
                drone_start_positions = []
                while agent_counter <= self.num_agents:
                    x, y = np.random.randint(0, world.shape[0]), np.random.randint(0, world.shape[1])
                    if world[x, y] == 0:
                        drone_start_positions.append((x, y, 0))
                        agent_counter += 1
                self.logger.debug(f"Randomly generated drone positions are : {drone_start_positions}")
                # RANDOMIZE THE GOALS OF AGENTS/ ACTOR START POSITIONS
                actors0_start_pos = []
                actors0_goal_pos = []
                goal_counter = 1
                agent_regions = dict()
                while goal_counter <= self.num_agents:
                    corresponding_drone_pos = drone_start_positions[goal_counter - 1]

                    # valid_tiles = get_connected_region(world, agent_regions, corresponding_drone_pos[0],
                    #                                    corresponding_drone_pos[1])

                    drone_start_pos_xy = (corresponding_drone_pos[0], corresponding_drone_pos[1])
                    valid_tiles = get_reachable_locations(world, drone_start_pos_xy)

                    x_actor_start, y_actor_start = valid_tiles[np.random.randint(0, len(valid_tiles))]
                    actor0_start_pos = (x_actor_start, y_actor_start, 0)

                    x_actor_goal, y_actor_goal = valid_tiles[np.random.randint(0, len(valid_tiles))]
                    actor0_goal_pos = (x_actor_goal, y_actor_goal, 0)

                    if actor0_start_pos not in actors0_start_pos and actor0_goal_pos not in actors0_goal_pos:
                        actors0_start_pos.append(actor0_start_pos)
                        actors0_goal_pos.append(actor0_goal_pos)
                        goal_counter += 1
                        self.logger.debug(f"Drone{goal_counter - 1} Start@{corresponding_drone_pos}; "
                                          f"Actor{goal_counter - 1} Start@{actor0_start_pos}."
                                          f"Actor{goal_counter - 1} Goal@{actor0_goal_pos}.")

                self.obstacle_map_initial = world
                self.initial_actor_starts = actors0_start_pos
                self.initial_actor_goals = actors0_goal_pos
                self.logger.debug(f"World: \n{world}")
                self.logger.info(f"Setting world with grid size = {self.grid_size}, world size = {world.shape},"
                                 f"\ngoal positions len = {len(actors0_start_pos)}, num agents = {self.num_agents}"
                                 f"\nAgents Start: \n{drone_start_positions}\n; Actors Start: \n{actors0_start_pos}"
                                 f"\nAgents Goals: \n{actors0_goal_pos}")
                self.operational_map = invert_array(world)
                self.world = WorldState(grid_size=self.grid_size,
                                        operational_map=self.operational_map,
                                        agents_start_position=drone_start_positions,
                                        actors_start_position=actors0_start_pos,
                                        actors_goal_position=actors0_goal_pos,
                                        num_agents=self.num_agents,
                                        num_actors=self.num_actors)
                return
            self.obstacle_map_initial = obstacle_map0
            self.initial_actor_starts = actors0_start_pos
            self.operational_map = invert_array(obstacle_map0)
            self.world = WorldState(grid_size=self.grid_size,
                                    operational_map=self.operational_map,
                                    actors_start_position=actors0_start_pos,
                                    actors_goal_position=actors0_start_pos,
                                    num_agents=self.num_agents,
                                    num_actors=self.num_actors)
            return

        # otherwise we have to randomize the world
        # RANDOMIZE THE STATIC OBSTACLES
        prob = np.random.triangular(self.PROB[0], .33 * self.PROB[0] + .66 * self.PROB[1], self.PROB[1])
        size = np.random.choice([self.SIZE[0], self.SIZE[0] * .5 + self.SIZE[1] * .5, self.SIZE[1]], p=[.5, .25, .25])
        world = (np.random.rand(int(size), int(size)) < prob).astype(int)

        # RANDOMIZE THE POSITIONS OF AGENTS/DRONE
        agent_counter = 1
        drone_start_positions = []
        while agent_counter <= self.num_agents:
            x, y = np.random.randint(0, world.shape[0]), np.random.randint(0, world.shape[1])
            if world[x, y] == 0:
                drone_start_positions.append((x, y, 0))
                agent_counter += 1
        self.logger.debug(f"Randomly generated drone positions are : {drone_start_positions}")
        # RANDOMIZE THE GOALS OF AGENTS/ ACTOR START POSITIONS
        actors0_start_pos = []
        actors0_goal_pos = []
        goal_counter = 1
        agent_regions = dict()
        while goal_counter <= self.num_agents:
            corresponding_drone_pos = drone_start_positions[goal_counter - 1]

            # valid_tiles = get_connected_region(world, agent_regions, corresponding_drone_pos[0],
            #                                    corresponding_drone_pos[1])

            drone_start_pos_xy = (corresponding_drone_pos[0], corresponding_drone_pos[1])
            valid_tiles = get_reachable_locations(world, drone_start_pos_xy)

            x_actor_start, y_actor_start = valid_tiles[np.random.randint(0, len(valid_tiles))]
            actor0_start_pos = (x_actor_start, y_actor_start, 0)

            x_actor_goal, y_actor_goal = valid_tiles[np.random.randint(0, len(valid_tiles))]
            actor0_goal_pos = (x_actor_goal, y_actor_goal, 0)

            if actor0_start_pos not in actors0_start_pos and actor0_goal_pos not in actors0_goal_pos:
                actors0_start_pos.append(actor0_start_pos)
                actors0_goal_pos.append(actor0_goal_pos)
                goal_counter += 1
                self.logger.debug(f"Drone{goal_counter - 1} Start@{corresponding_drone_pos}; "
                                  f"Actor{goal_counter - 1} Start@{actor0_start_pos}."
                                  f"Actor{goal_counter - 1} Goal@{actor0_goal_pos}.")

        self.obstacle_map_initial = world
        self.initial_actor_starts = actors0_start_pos
        self.initial_actor_goals = actors0_goal_pos
        self.logger.debug(f"World: \n{world}")
        self.logger.info(f"Setting world with grid size = {self.grid_size}, world size = {world.shape},"
                         f"\ngoal positions len = {len(actors0_start_pos)}, num agents = {self.num_agents}"
                         f"\nAgents Start: \n{drone_start_positions}\n; Actors Start: \n{actors0_start_pos}"
                         f"\nAgents Goals: \n{actors0_goal_pos}")
        self.operational_map = invert_array(world)
        self.world = WorldState(grid_size=self.grid_size,
                                operational_map=self.operational_map,
                                agents_start_position=drone_start_positions,
                                actors_start_position=actors0_start_pos,
                                actors_goal_position=actors0_goal_pos,
                                num_agents=self.num_agents,
                                num_actors=self.num_actors)

    # def observation_space(self) -> gym.spaces.Dict:
    #     """
    #     The observation space.
    #     """
    #     return get_item(self.observations.values()).space
    # @property
    # def action_space(self) -> gym.spaces.Discrete:
    #     """
    #     The action space.
    #     """
    #     return self._action_space

    # Returns an observation of an agent
    def _observe(self, agent_id):
        assert (agent_id >= 0)

        agent_observation_channel = self.world.agents_state.drones[agent_id].get_current_observation_channels()

        dx = self.world.get_agent_goal_by_id(agent_id)[0] - self.world.get_agents_position_by_id(agent_id)[0]
        dy = self.world.get_agent_goal_by_id(agent_id)[1] - self.world.get_agents_position_by_id(agent_id)[1]
        mag = (dx ** 2 + dy ** 2) ** .5
        if mag != 0:
            dx = dx / mag
            dy = dy / mag
        return agent_observation_channel, [dx, dy, mag]

    # Resets environment
    def _reset(self, agent_id, world0=None, goals0=None):
        self.logger.info(f"Resetting the world now")
        self.finished = False
        self.mutex.acquire()

        # Initialize data structures
        self._set_world(world0, goals0)
        self.fresh = True

        self.mutex.release()
        if self.viewer is not None:
            self.viewer = None
        on_goal = self.world.get_agents_position_by_id(agent_id) == self.world.get_agent_goal_by_id(agent_id)
        # we assume you don't start blocking anyone (the probability of this happening is insanely low)
        self.logger.info(f"Resetting the world in complete...")
        return self._listNextValidActions(agent_id), on_goal, False

    def _complete(self):
        return self.world.done()

    def get_astar_costs(self, start, goal):
        # returns a numpy array of same dims as self.world.state with the distance to the goal from each coord
        def lowest_f(fScore_in, openSet_in):
            # find entry in openSet with lowest fScore
            assert (len(openSet_in) > 0)
            min_f = 2 ** 31 - 1
            min_node = None
            for (i, j) in openSet_in:
                if (i, j) not in fScore_in: continue
                if fScore_in[(i, j)] < min_f:
                    min_f = fScore_in[(i, j)]
                    min_node = (i, j)
            return min_node

        def get_neighbours(node):
            # return set of neighbors to the given node
            n_moves = 9
            neighbors = set()
            for move in range(1, n_moves):  # we don't want to include 0, or it will include itself
                direction = self.world.agents_state.drones[0].translation_dirs
                dx = direction[0]
                dy = direction[1]
                ax = node[0]
                ay = node[1]
                if (ax + dx >= self.world.agents_state.shape[0] or ax + dx < 0
                        or ay + dy >= self.world.agents_state.shape[1] or ay + dy < 0):  # out of bounds
                    continue
                if self.world.agents_state[ax + dx, ay + dy] == -1:  # collide with static obstacle
                    continue
                neighbors.add((ax + dx, ay + dy))
            return neighbors

        # NOTE THAT WE REVERSE THE DIRECTION OF SEARCH SO THAT THE GSCORE WILL BE DISTANCE TO GOAL
        start, goal = goal, start

        # The set of nodes already evaluated
        closedSet = set()

        # The set of currently discovered nodes that are not evaluated yet.
        # Initially, only the start node is known.
        openSet = set()
        openSet.add(start)

        # For each node, which node it can most efficiently be reached from.
        # If a node can be reached from many nodes, cameFrom will eventually contain the
        # most efficient previous step.
        cameFrom = dict()

        # For each node, the cost of getting from the start node to that node.
        gScore = dict()  # default value infinity

        # The cost of going from start to start is zero.
        gScore[start] = 0

        # For each node, the total cost of getting from the start node to the goal
        # by passing by that node. That value is partly known, partly heuristic.
        fScore = dict()  # default infinity

        # our heuristic is euclidean distance to goal
        heuristic_cost_estimate = lambda x, y: math.hypot(x[0] - y[0], x[1] - y[1])

        # For the first node, that value is completely heuristic.
        fScore[start] = heuristic_cost_estimate(start, goal)

        while len(openSet) != 0:
            # current = the node in openSet having the lowest fScore value
            current = lowest_f(fScore, openSet)

            openSet.remove(current)
            closedSet.add(current)
            for neighbor in get_neighbours(current):
                if neighbor in closedSet:
                    continue  # Ignore the neighbor which is already evaluated.

                if neighbor not in openSet:  # Discover a new node
                    openSet.add(neighbor)

                # The distance from start to a neighbor
                # in our case the distance between is always 1
                tentative_gScore = gScore[current] + 1
                if tentative_gScore >= gScore.get(neighbor, 2 ** 31 - 1):
                    continue  # This is not a better path.

                # This path is the best until now. Record it!
                cameFrom[neighbor] = current
                gScore[neighbor] = tentative_gScore
                fScore[neighbor] = gScore[neighbor] + heuristic_cost_estimate(neighbor, goal)

                # parse through the gScores
        costs = self.world.agents_state.copy()
        for (i, j) in gScore:
            costs[i, j] = gScore[i, j]
        return costs

    def get_blocking_reward(self, agent_id):
        """calculates how many robots the agent is preventing from reaching goal
        and returns the necessary penalty"""
        # accumulate visible robots
        other_robots = []
        other_locations = []
        inflation = 10
        top_left = (self.world.get_agents_position_by_id(agent_id)[0] - self.observation_size // 2,
                    self.world.get_agents_position_by_id(agent_id)[1] - self.observation_size // 2)
        bottom_right = (top_left[0] + self.observation_size, top_left[1] + self.observation_size)
        for agent in range(1, self.num_agents):
            if agent == agent_id: continue
            x, y, _ = self.world.get_agents_position_by_id(agent)
            if x < top_left[0] or x >= bottom_right[0] or y >= bottom_right[1] or y < top_left[1]:
                continue
            other_robots.append(agent)
            other_locations.append((x, y))
        num_blocking = 0
        world = self.get_obstacle_map()
        for agent in other_robots:
            other_locations.remove(self.world.get_agents_position_by_id(agent))
            # before removing
            path_before = astar(world, self.world.get_agents_position_by_id(agent), self.world.get_agent_goal_by_id(agent),
                                robots=other_locations + [self.world.get_agents_position_by_id(agent_id)])
            # after removing
            path_after = astar(world, self.world.get_agent_goal_by_id(agent), self.world.get_agent_goal_by_id(agent),
                               robots=other_locations)
            other_locations.append(self.world.get_agents_position_by_id(agent))

            if path_before is None and path_after is None:
                continue
            if path_before is not None and path_after is None:
                continue
            if (path_before is None and path_after is not None) \
                    or len(path_before) > len(path_after) + inflation:
                num_blocking += 1

        return num_blocking * BLOCKING_COST

    # Executes an action by an agent
    def step(self, action_input: ThreeIntTuple, episode: int = 0):
        # episode is an optional variable which will be used on the reward discounting
        self.fresh = False
        n_actions = 9

        assert len(action_input) == 2, 'Action input should be a tuple with the form (agent_id, action)'
        assert action_input[1] in range(n_actions), 'Invalid action'
        assert action_input[0] in range(0, self.num_agents)

        # Parse action input
        agent_id = action_input[0]
        action = action_input[1]

        # Lock mutex (race conditions start here)
        self.mutex.acquire()

        # get start location of agent
        agentStartLocation = self.world.get_agents_position_by_id(agent_id)

        # Execute action & determine reward
        action_status = self.world.act(action, agent_id)
        valid_action = action_status >= 0
        #     2: action executed and left goal
        #     1: action executed and reached/stayed on goal
        #     0: action executed
        #    -1: out of bounds
        #    -2: collision with wall
        #    -3: collision with robot
        blocking = False
        reward = self.individual_rewards[agent_id]
        if action == 0:  # staying still
            if action_status == 1:  # stayed on goal
                reward = GOAL_REWARD
                x = self.get_blocking_reward(agent_id)
                reward += x
                if x < 0:
                    blocking = True
            elif action_status == 0:  # stayed off goal
                reward = IDLE_COST
        else:  # moving
            if action_status == 1:  # reached goal
                reward = GOAL_REWARD
            elif action_status == -3 or action_status == -2 or action_status == -1:  # collision
                reward = COLLISION_REWARD
            elif action_status == 2:  # left goal
                reward = ACTION_COST
            else:
                reward = ACTION_COST
        self.individual_rewards[agent_id] = reward

        if JOINT:
            visible = [False for i in range(self.num_agents)]
            v = 0
            # joint rewards based on proximity
            for agent in range(1, self.num_agents + 1):
                # tally up the visible agents
                if agent == agent_id:
                    continue
                top_left = (self.world.get_agents_position_by_id(agent_id)[0] - self.observation_size // 2,
                            self.world.get_agents_position_by_id(agent_id)[1] - self.observation_size // 2)
                pos = self.world.get_agents_position_by_id(agent)
                if top_left[0] <= pos[0] < top_left[0] + self.observation_size \
                        and top_left[1] <= pos[1] < top_left[1] + self.observation_size:
                    # if the agent is within the bounds for observation
                    v += 1
                    visible[agent - 1] = True
            if v > 0:
                reward = self.individual_rewards[agent_id - 1] / 2
                # set the reward to the joint reward if we are
                for i in range(self.num_agents):
                    if visible[i]:
                        reward += self.individual_rewards[i] / (v * 2)

        # Perform observation
        state = self._observe(agent_id)

        # Done?
        done = self.world.done()
        self.finished |= done

        # next valid actions
        nextActions = self._listNextValidActions(agent_id, action, episode=episode)

        # on_goal estimation
        on_goal = self.world.get_agents_position_by_id(agent_id) == self.world.get_agent_goal_by_id(agent_id)

        # Unlock mutex
        self.mutex.release()
        return state, reward, done, nextActions, on_goal, blocking, valid_action

    def _listNextValidActions(self, agent_id, prev_action=0, episode=0):
        available_actions = [0]  # staying still always allowed

        # Get current agent position
        agent_pos = self.world.get_agents_position_by_id(agent_id)
        ax, ay, ao = agent_pos
        n_moves = 9

        for action in range(1, n_moves):
            direction = self.world.agents_state.drones[0].translation_dirs
            dx, dy = direction[0], direction[1]
            if (ax + dx >= self.world.agents_state.shape[0] or ax + dx < 0 or ay + dy >= self.world.agents_state.shape[
                1] or ay + dy < 0):  # out of bounds
                continue
            if self.world.agents_state[ax + dx, ay + dy] < 0:  # collide with static obstacle
                continue
            if self.world.agents_state[ax + dx, ay + dy] > 0:  # collide with robot
                continue
            # check for diagonal collisions
            if DIAGONAL_MOVEMENT:
                if self.world.diagonal_collision(agent_id, (ax + dx, ay + dy, ao)):
                    continue
                    # otherwise we are ok to carry out the action
            available_actions.append(action)

        if opposite_actions[prev_action] in available_actions:
            available_actions.remove(opposite_actions[prev_action])

        return available_actions

    def drawStar(self, centerX, centerY, diameter, numPoints, color):
        outerRad = diameter // 2
        innerRad = int(outerRad * 3 / 8)
        # fill the center with the star
        angleBetween = 2 * math.pi / numPoints  # angle between star points in radians
        for i in range(numPoints):
            # p1 and p3 are on the inner radius, and p2 is the point
            pointAngle = math.pi / 2 + i * angleBetween
            p1X = centerX + innerRad * math.cos(pointAngle - angleBetween / 2)
            p1Y = centerY - innerRad * math.sin(pointAngle - angleBetween / 2)
            p2X = centerX + outerRad * math.cos(pointAngle)
            p2Y = centerY - outerRad * math.sin(pointAngle)
            p3X = centerX + innerRad * math.cos(pointAngle + angleBetween / 2)
            p3Y = centerY - innerRad * math.sin(pointAngle + angleBetween / 2)
            # draw the triangle for each tip.
            poly = rendering.FilledPolygon([(p1X, p1Y), (p2X, p2Y), (p3X, p3Y)])
            poly.set_color(color[0], color[1], color[2])
            poly.add_attr(rendering.Transform())
            self.viewer.add_onetime(poly)

    def create_rectangle(self, x, y, width, height, fill, permanent=False):
        ps = [(x, y), ((x + width), y), ((x + width), (y + height)), (x, (y + height))]
        rect = rendering.FilledPolygon(ps)
        rect.set_color(fill[0], fill[1], fill[2])
        rect.add_attr(rendering.Transform())
        if permanent:
            self.viewer.add_geom(rect)
        else:
            self.viewer.add_onetime(rect)

    def create_circle(self, x, y, diameter, size, fill, resolution=20):
        c = (x + size / 2, y + size / 2)
        dr = math.pi * 2 / resolution
        ps = []
        for i in range(resolution):
            x = c[0] + math.cos(i * dr) * diameter / 2
            y = c[1] + math.sin(i * dr) * diameter / 2
            ps.append((x, y))
        circ = rendering.FilledPolygon(ps)
        circ.set_color(fill[0], fill[1], fill[2])
        circ.add_attr(rendering.Transform())
        self.viewer.add_onetime(circ)

    def initColors(self):
        c = {a + 1: hsv_to_rgb(np.array([a / float(self.num_agents), 1, 1])) for a in range(self.num_agents)}
        return c

    def _render(self, mode='human', close=False, screen_width=800, screen_height=800, action_probs=None):
        if close:
            return
        # values is an optional parameter which provides a visualization for the value of each agent per step
        size = screen_width / max(self.world.agents_state.shape[0], self.world.agents_state.shape[1])
        colors = self.initColors()
        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.reset_renderer = True
        if self.reset_renderer:
            self.create_rectangle(0, 0, screen_width, screen_height, (.6, .6, .6), permanent=True)
            for i in range(self.world.agents_state.shape[0]):
                start = 0
                end = 1
                scanning = False
                write = False
                for j in range(self.world.agents_state.shape[1]):
                    if self.world.agents_state[i, j] != -1 and not scanning:  # free
                        start = j
                        scanning = True
                    if (j == self.world.agents_state.shape[1] - 1 or self.world.agents_state[i, j] == -1) and scanning:
                        end = j + 1 if j == self.world.agents_state.shape[1] - 1 else j
                        scanning = False
                        write = True
                    if write:
                        x = i * size
                        y = start * size
                        self.create_rectangle(x, y, size, size * (end - start), (1, 1, 1), permanent=True)
                        write = False
        for agent in range(1, self.num_agents + 1):
            i, j, _ = self.world.get_agents_position_by_id(agent)
            x = i * size
            y = j * size
            color = colors[self.world.agents_state[i, j]]
            self.create_rectangle(x, y, size, size, color)
            i, j, _ = self.world.get_agent_goal_by_id(agent)
            x = i * size
            y = j * size
            color = colors[self.world.goals[i, j]]
            self.create_circle(x, y, size, size, color)
            if self.world.get_agent_goal_by_id(agent) == self.world.get_agents_position_by_id(agent):
                color = (0, 0, 0)
                self.create_circle(x, y, size, size, color)
        if action_probs is not None:
            n_moves = 9
            for agent in range(1, self.num_agents + 1):
                # take the a_dist from the given data and draw it on the frame
                a_dist = action_probs[agent - 1]
                if a_dist is not None:
                    for m in range(n_moves):
                        dx, dy = self.world.agents_state.drones[0].translation_dirs
                        x = (self.world.get_agents_position_by_id(agent)[0] + dx) * size
                        y = (self.world.get_agents_position_by_id(agent)[1] + dy) * size
                        s = a_dist[m] * size
                        self.create_circle(x, y, s, size, (0, 0, 0))
        self.reset_renderer = False
        result = self.viewer.render(return_rgb_array=mode == 'rgb_array')
        return result


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    n_agents = 8
    env = MUDMAFEnv(num_agents=n_agents,
                    world0=None,
                    grid_size=1.0,
                    goals0=None,
                    size=(20, 20),
                    obstacle_prob_range=(.1, .2),
                    SEED=SEED
                    )
    logging.info(f"Starting Agent positions: {env.get_agent_positions()}")
    # print(coordinationRatio(env))
