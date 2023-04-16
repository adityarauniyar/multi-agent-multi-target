import time as timer
import heapq
import random
from single_agent_planner import compute_heuristics, a_star, get_location, get_sum_of_cost
import numpy as np
from typing import List, Tuple


def detect_collision(path1, path2):
    ##############################
    #  Return the first collision that occurs between two robot paths (or None if there is no collision)
    #           There are two types of collisions: vertex collision and edge collision.
    #           A vertex collision occurs if both robots occupy the same location at the same timestep
    #           An edge collision occurs if the robots swap their location at the same timestep.
    #           You should use "get_location(path, t)" to get the location of a robot at time t.

    larger_path_agent = path1 if len(path1) > len(path2) else path2
    smaller_path_agent = path1 if len(path1) < len(path2) else path2

    # Loop to check vertex collisions
    # print("[INFO] Checking vertex collisions...") 
    for t in range(max(len(path1), len(path2))):
        location1 = get_location(path=path1, time=t) if t < len(path1) else get_location(path=path1,
                                                                                         time=len(path1) - 1)
        location2 = get_location(path=path2, time=t) if t < len(path2) else get_location(path=path2,
                                                                                         time=len(path2) - 1)
        # print("[INFO] Agent \"a\" @ {}; and Agent \"b\" @ {}".format(location1, location2))

        # Checking vertex collision between agent1 and agent2
        if location1 == location2:
            # print("[WARN] Vertex collision found at location {} and timestep {}, reporting...".format(location1, t))
            return location1, None, t

    # loops to check edge collision for the two agents
    # for-looping the smaller path
    # print("[INFO] Checking Edge collisions...")
    for ti in range(len(path1) - 1):
        vertex_of_path1_at_iT = path1[ti]
        vertex_of_path1_at_next_iT = path1[ti + 1]
        # Checking if the next index from the smaller path is present in the larger path.
        if ti + 1 <= len(path2) - 1:
            vertex_of_path2_at_next_iT = path2[ti + 1]
            vertex_of_path2_at_iT = path2[ti]

            if vertex_of_path2_at_iT == vertex_of_path1_at_next_iT and vertex_of_path2_at_next_iT == vertex_of_path1_at_iT:
                # print("[WARN] Edge collision found from location {} to location {} at timestep {},
                # reporting...".format(vertex_of_path1_at_iT, vertex_of_path1_at_next_iT, ti + 1))
                return vertex_of_path1_at_iT, vertex_of_path1_at_next_iT, ti + 1

    return None, None, None


def detect_collisions(paths):
    collisions = []
    ##############################
    # Return a list of first collisions between all robot pairs.
    #           A collision can be represented as dictionary that contains the id of the two robots, the vertex or edge
    #           causing the collision, and the timestep at which the collision occurred.
    #           You should use your detect_collision function to find a collision between two robots.
    # print("Paths({}) in-hand to detect for collision: ".format(len(paths)), paths)

    # Making combination of agent-agent pair to check collisions
    # TODO: The order on which these collisions matter needs to be verified.
    for i in range(len(paths) - 1):
        for j in range(i + 1, len(paths)):
            # print("Checking collisions between agent {} and agent {}".format(i,j))
            # NOTE that if this reports an edge collision the vertex1 to vertex2 is of ith agent 
            vertex1, vertex2, timestep = detect_collision(paths[i], paths[j])
            if vertex1 is not None:
                collision = {'a1': i,
                             'a2': j,
                             'loc': [vertex1],
                             'timestep': timestep}
                if vertex2 is not None:
                    collision['loc'].append(vertex2)

                collisions.append(collision)

    return collisions


def resolve_collision(collision):
    constraints = []
    ##############################
    #  Return a list of (two) constraints to resolve the given collision
    #           Vertex collision: the first constraint prevents the first agent to be at the specified location at the
    #                            specified timestep, and the second constraint prevents the second agent to be at the
    #                            specified location at the specified timestep.
    #           Edge collision: the first constraint prevents the first agent to traverse the specified edge at the
    #                          specified timestep, and the second constraint prevents the second agent to traverse the
    #                          specified edge at the specified timestep
    if collision is not None:
        resolve1 = {'agent': collision['a1'],
                    'loc': collision['loc'],
                    'timestep': collision['timestep']}

        resolve2 = {'agent': collision['a2'],
                    'loc': resolve1['loc'][::-1],
                    'timestep': collision['timestep']}

        constraints.append(resolve1)
        constraints.append(resolve2)

    return constraints


class CBSSolver(object):
    """The high-level search of CBS."""

    def __init__(
            self,
            my_map: np.ndarray,
            starts: List[Tuple[int, int]],
            goals: List[Tuple[int, int]]
    ):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """

        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)

        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.CPU_time = 0

        self.open_list = []

        # compute heuristics for the low-level search
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

    def push_node(self, node):
        heapq.heappush(self.open_list, (node['cost'], len(node['collisions']), self.num_of_generated, node))
        self.num_of_generated += 1

    def pop_node(self):
        _, _, id, node = heapq.heappop(self.open_list)
        self.num_of_expanded += 1
        return node

    def find_solution(self) -> List[List[Tuple[int, int]]]:
        """ Finds paths for all agents from their start locations to their goal locations
        """

        # print('Start CBS')
        self.start_time = timer.time()

        # print("[INFO] (CBS) Starts: ", self.starts)
        # print("[INFO] (CBS) Goals: ", self.goals)
        # print("[INFO] (CBS) My Map: ", self.my_map)
        # print("[INFO] (CBS) Number of agents: ", self.num_of_agents)

        # Generate the root node
        # constraints   - list of constraints
        # paths         - list of paths, one for each agent
        #               [[(x11, y11), (x12, y12), ...], [(x21, y21), (x22, y22), ...], ...]
        # collisions     - list of collisions in paths
        root = {'cost': 0,
                'constraints': [],
                'paths': [],
                'collisions': []}
        for i in range(self.num_of_agents):  # Find initial path for each agent
            new_path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i],
                              i, root['constraints'])
            if new_path is None:
                raise BaseException('No solutions')
            root['paths'].append(new_path)

        root['cost'] = get_sum_of_cost(root['paths'])
        root['collisions'] = detect_collisions(root['paths'])
        self.push_node(root)
        self.num_of_generated += 1

        # print("[DEBUG] CBS Root node: ", root)

        #  Testing
        # # print("Displaying collisions from CBS solver: ", root['collisions'])

        #  Testing
        # for collision in root['collisions']:
        #     # print("Resolving constraints: ", resolve_collision(collision))

        ##############################
        #  High-Level Search
        #           Repeat the following as long as the open list is not empty:
        #             1. Get the next node from the open list (you can use self.pop_node()
        #             2. If this node has no collision, return solution
        #             3. Otherwise, choose the first collision and convert to a list of constraints (using your
        #                resolve_collision function). Add a new child node to your open list for each constraint
        #           Ensure to create a copy of any objects that your child nodes might inherit

        # Iterating till the open_list is NOT EMPTY 
        while len(self.open_list) > 0:
            CPU_time_s = timer.time() - self.start_time

            # print("[INFO] Popping from the open list...")
            currNode = self.pop_node()
            self.num_of_expanded += 1
            # print("[DEBUG] CBS Current Node: ", currNode)

            # Checking if there is any collisions in the paths, if NOT returning the paths
            if len(currNode['collisions']) == 0 or CPU_time_s >= 600:
                # print("[ALARM] There are no collisions, hence returning the planned path...")
                self.print_results(currNode)
                return currNode['paths']

                # Just getting the first collision from the list of collisions and resolving that
            collision = currNode['collisions'][0]
            constraints = resolve_collision(collision)
            # print("[DEBUG] Current constraint stack: ", constraints)

            # Iterating over constraints and planning paths based on it
            for constraint in constraints:

                # Creating a new node that constains paths that resolves the current Node constraints 
                # print("[DEBUG] Current constraint: ", constraint)
                new_node = {'cost': get_sum_of_cost(currNode['paths']),
                            'constraints': [constraint],
                            'paths': currNode['paths'].copy(),
                            'collisions': []}

                if len(currNode['constraints']) > 0:
                    new_node['constraints'] = currNode['constraints'].copy()
                    new_node['constraints'].append(constraint)

                # print("[DEBUG] New RAW node with the constraint: ", new_node)
                # Plainning paths for agent whose constraint is recently defined and also considers constraint
                agent_ID = constraint['agent']
                # print("[DEBUG] Plainning a new path for agent ({}) with constraints : ({})".format(agent_ID,
                # new_node['constraints']))
                new_path = a_star(self.my_map, self.starts[agent_ID], self.goals[agent_ID], self.heuristics[agent_ID],
                                  agent_ID, new_node['constraints'])

                if new_path is not None and len(new_path) is not []:
                    new_node['paths'][agent_ID] = new_path
                    # print("[DEBUG] (CBS) Updated path on the new node: ", )
                    new_node['collisions'] = detect_collisions(new_node['paths'])
                    new_node['cost'] = get_sum_of_cost(new_node['paths'])
                    self.push_node(new_node)
                    self.num_of_generated += 1
                    # print("[DEBUG] New node with udpated constraint: ", new_node)
                else:
                    # print("[ALERT] (CBS) This Node doesn't lead to a solution...")
                    pass

        ##############################
        raise BaseException('No solutions')

    def print_results(self, node):
        print("\n Found a solution! \n")
        CPU_time = timer.time() - self.start_time
        print("CPU time (s):    {:.2f}".format(CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(node['paths'])))
        print("Expanded nodes:  {}".format(self.num_of_expanded))
        print("Generated nodes: {}".format(self.num_of_generated))
