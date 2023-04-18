import heapq
import numpy as np


def move(loc, dir_seq):
    directions = [(0, 0),  # 0: stay in place
                  (1, 0),  # 1: east
                  (-1, 0),  # 2: west
                  (0, 1),  # 3: north
                  (0, -1),  # 4: south
                  (1, 1),  # 5: north-east
                  (-1, 1),  # 6: north-west
                  (1, -1),  # 7: south-east
                  (-1, -1)]  # 8: south-west
    return loc[0] + directions[dir_seq][0], loc[1] + directions[dir_seq][1]


def is_valid_motion(old_loc, new_loc):
    # Check if two agents are in the same location (vertex collision)
    if len(set(new_loc)) != len(new_loc):
        return False

    # Check edge collision
    for i in range(len(new_loc)):
        for j in range(len(old_loc)):
            if i == j:
                continue
            if new_loc[i] == old_loc[j] and new_loc[j] == old_loc[i]:
                return False

    return True


def get_sum_of_cost(paths):
    rst = 0
    for path in paths:
        rst += len(path) - 1
    return rst


def compute_heuristics(my_map, goal):
    # Use Dijkstra to build a shortest-path tree rooted at the goal location
    open_list = []
    closed_list = dict()
    root = {'loc': goal, 'cost': 0}
    heapq.heappush(open_list, (root['cost'], goal, root))
    closed_list[goal] = root
    while len(open_list) > 0:
        (cost, loc, curr) = heapq.heappop(open_list)
        for dir_seq in range(1, 9):
            child_loc = move(loc, dir_seq)
            child_cost = cost + 1
            if child_loc[0] < 0 or child_loc[0] >= len(my_map) \
                    or child_loc[1] < 0 or child_loc[1] >= len(my_map[0]):
                continue
            if my_map[child_loc[0]][child_loc[1]]:
                continue
            child = {'loc': child_loc, 'cost': child_cost}
            if child_loc in closed_list:
                existing_node = closed_list[child_loc]
                if existing_node['cost'] > child_cost:
                    closed_list[child_loc] = child
                    # open_list.delete((existing_node['cost'], existing_node['loc'], existing_node))
                    heapq.heappush(open_list, (child_cost, child_loc, child))
            else:
                closed_list[child_loc] = child
                heapq.heappush(open_list, (child_cost, child_loc, child))

    # build the heuristics table
    h_values = dict()
    for loc, node in closed_list.items():
        h_values[loc] = node['cost']
    return h_values


def build_constraint_table(constraints, agent):
    lookupTable = dict()
    max_timestep = 0
    log_overlay = "[DEBUG] (build_constraint_table) "
    ##############################
    #  Return a table that constains the list of constraints of
    #               the given agent for each time step. The table can be used
    #               for a more efficient constraint violation check in the 
    #               is_constrained function.
    # constraint = {'agent': 2, 'loc': [(3,4)], 'timestep': 5}

    # Making sure that the key as per the constraint exist.
    if constraints is not None:
        # print("Building lookup table for agent({})...".format(agent)) 
        for constraint in constraints:
            # # print(log_overlay + "Current constraint: ", constraint)
            if str(agent) in lookupTable:
                if not str(constraint['timestep']) in lookupTable[str(agent)]:
                    lookupTable[str(agent)][str(constraint['timestep'])] = {'loc': []}
            else:
                lookupTable[str(agent)] = {str(constraint['timestep']): {'loc': []}}
            # # print("[DEBUG] (build_constraint_table) lookuptable for the agent {}:".format(lookupTable[str(agent)]))

            # Adding constraint the given agent at the given timestep
            for constraint_loc in constraint['loc']:
                agent_constraints_at_T = lookupTable[str(agent)][str(constraint['timestep'])]['loc']
                # Checking if new constraint added is unique in the constraint list 
                # print(log_overlay + "Current constraint loc: ", constraint_loc)
                if constraint_loc not in agent_constraints_at_T:
                    # # print("[DEBUG] (build_constraint_table) Adding {} into {} with timestep ({})...".format(tuple(constraint_loc), lookupTable[str(agent)][str(constraint['timestep'])]['loc'], constraint['timestep']))
                    lookupTable[str(agent)][str(constraint['timestep'])]['loc'].append(tuple(constraint_loc))

            # Updating the max timestep of constraint for a given agent
            if max_timestep < constraint['timestep']:
                max_timestep = constraint['timestep']

    ##############################
    return lookupTable, max_timestep


def is_constrained(curr_loc, next_loc, next_time, constraint_table):
    ##############################
    #  Check if a move from curr_loc to next_loc at time step next_time violates
    #               any given constraint. For efficiency the constraints are indexed in a constraint_table
    #               by time step, see build_constraint_table.

    if constraint_table is not None and str(next_time) in constraint_table:
        # print("[DEBUG] (is_constrained) Constaint Table:", constraint_table)
        next_time_vertex_constraints_loc = constraint_table[str(next_time)]['loc']
        for next_time_vertex_constraint_loc in next_time_vertex_constraints_loc:
            # Check for vertex collision:
            if next_loc == next_time_vertex_constraint_loc:
                # print("[WARN] (is_constrained) Vertex collision with another agent detected: next_loc({}) and next_time_vertex_constraint_loc({})".format(next_loc, next_time_vertex_constraint_loc))
                return True

                # Check for edge collision
            if curr_loc is not None and curr_loc == next_time_vertex_constraint_loc and str(
                    next_time - 1) in constraint_table:
                curr_time_vertex_constraints_loc = constraint_table[str(next_time - 1)]['loc']
                for curr_time_vertex_constraint_loc in curr_time_vertex_constraints_loc:
                    if curr_time_vertex_constraint_loc == next_loc:
                        # print("[WARN] (is_constrained) Edge collision with another agent detected: next_loc({}) and curr_time_vertex_constraint({})".format(next_loc, curr_time_vertex_constraint_loc))
                        return True
    else:
        # print("[ERROR] Something is wrong with the constraint table, Table:{}; next_time: {}".format(constraint_table, next_time))
        pass

    # if constraint_table is not None and str(next_time - 1) in constraint_table:
    #     # print("Constraint Table: ", constraint_table)
    #     curr_time_vertex_constraints_loc = constraint_table[str(next_time - 1)]['loc']
    #     for curr_time_vertex_constraint_loc in curr_time_vertex_constraints_loc:
    #         # Check for vertex collisions
    #         if curr_loc == curr_time_vertex_constraint_loc:
    #             # print("Vertex collision with another agent detected: curr_loc({}) and curr_time_vertex_constraint_loc({})".format(curr_loc, curr_time_vertex_constraint_loc))
    #             return True

    #         # Check for Edge collisions
    #         if next_loc is not None and next_loc == curr_time_vertex_constraint_loc and str(next_time) in constraint_table:
    #             next_time_vertex_constraints_loc = constraint_table[str(next_time)]['loc']
    #             for next_time_vertex_constraint_loc in next_time_vertex_constraints_loc:
    #                 if curr_loc == next_time_vertex_constraint_loc:
    #                     # print("Edge collision with another agent detected: next_loc({}) and curr_time_vertex_constraint({})".format(next_loc, curr_time_vertex_constraint_loc))
    #                     return True

    ##############################
    return False


def get_location(path, time):
    if time < 0:
        return path[0]
    elif time < len(path):
        return path[time]
    else:
        return path[-1]  # wait at the goal location


def get_path(goal_node):
    path = []
    curr = goal_node
    while curr is not None:
        path.append(curr['loc'])
        ## print('loc: {}, timestep: {}'.format(curr['loc'], curr['timestep']))
        curr = curr['parent']
    path.reverse()
    return path


def push_node(open_list, node):
    heapq.heappush(open_list, (node['g_val'] + node['h_val'], node['h_val'], node['loc'], node))


def pop_node(open_list):
    _, _, _, curr = heapq.heappop(open_list)
    return curr


def compare_nodes(n1, n2):
    """Return true is n1 is better than n2."""
    return n1['g_val'] + n1['h_val'] < n2['g_val'] + n2['h_val']


def in_map(map, loc):
    if loc[0] >= len(map) or loc[1] >= len(map[0]) or min(loc) < 0:
        return False
    else:
        return True


def all_in_map(map, locs):
    for loc in locs:
        if not in_map(map, loc):
            return False
    return True


def a_star(my_map, start_loc, goal_loc, h_values, agent, constraints):
    """ my_map      - binary obstacle map
        start_loc   - start position
        goal_loc    - goal position
        agent       - the agent that is being re-planned
        constraints - constraints defining where robot should or cannot go at each timestep
    """

    ##############################
    #  Extend the A* search to search in the space-time domain
    #           rather than space domain, only.

    open_list = []
    closed_list = dict()
    h_value = h_values[start_loc]
    num_open_spaces = np.count_nonzero(np.array(my_map) == False)

    constraint_lookupTable, max_timestep_constraint = build_constraint_table(constraints, agent)
    # print("[DEBUG] (A-star) Constraint Lookup table: ", constraint_lookupTable)
    # print("[DEBUG] (A-star) Max Timestep constraint: ", max_timestep_constraint)

    root = {'loc': start_loc, 'g_val': 0, 'h_val': h_value, 'timestep': 0, 'parent': None}

    # print("Root Node: ", root)
    # # print("Map: ", my_map)
    # print("Number of Open(Non obstable Spaces): ", num_open_spaces)
    # # print("Heuristics: ", h_value)

    push_node(open_list, root)
    closed_list[(root['loc'], root['timestep'])] = root

    # print("Iterating over open lists...")
    while len(open_list) > 0:
        # print("[DEBUG] (A-star) Popping from the open list...")
        currNode = pop_node(open_list)
        # print("[DEBUG] (A-star) Agent: {}; Timestep: ".format(agent, currNode['timestep']))
        # print("[DEBUG] (A-star) Current Node: ", currNode)
        #############################
        #  Adjust the goal test condition to handle goal constraints

        is_collision_when_stayed = False
        if currNode['loc'] == goal_loc:
            # print("[DEBUG] Current Node is the GOAL location...")
            if str(agent) in constraint_lookupTable:
                # FIXME : Why the max here?
                # print("[INFO] Constraint table for the agent {}:".format(agent), constraint_lookupTable[str(agent)])
                # 2.3.1 Checking if staying at the goal location doesnt lead to collision with any other agents in future
                if currNode['timestep'] <= max_timestep_constraint:
                    for ts_onwards in range(currNode['timestep'], max_timestep_constraint + 1):
                        # print("[INFO] (A-star) Checking if current state collides with any other agents after this timestep (ts({})...".format(ts_onwards))
                        if str(ts_onwards) in constraint_lookupTable[str(agent)] and goal_loc in \
                                constraint_lookupTable[str(agent)][str(ts_onwards)]['loc']:
                            # print("[WARN] This leads to collision in future, skipping...")
                            is_collision_when_stayed = True
                            break
                            # Making sure that the max_constraint_vertex is applied to all the timesteps after it
                # else: 
                #     if str(max_timestep_constraint) in constraint_lookupTable[str(agent)] and goal_loc in constraint_lookupTable[str(agent)][str(max_timestep_constraint)]['loc'] :
                #         # print("[WARN] This leads to collision, as some agent is already at current planner path, skipping...")
                #         is_collision_when_stayed = True
                #         break 

            if not is_collision_when_stayed:
                # print("[INFO] (A-star) Goal location reached, terminating...")
                return get_path(currNode)
            else:
                continue

        # 2.3.3 Time to terminated after agent has looked possibly the entire map for any paths. 
        if currNode['timestep'] > len(my_map) * len(my_map[0]):
            # print("[WARN] Terminating after timestep {}, current timestep {}".format(currNode['timestep'], len(my_map) * len(my_map[0])))
            break

            #############################

        # Range of motions till 5 considers that the 5th motion waits on its cell 
        for dir_seq in range(9):

            # Getting possible vertex positions from list of possible directions 
            childNode_loc = move(currNode['loc'], dir_seq)
            # print("[DEBUG] (A-star) Child Node Location: {}; Timestep: {}".format(childNode_loc, currNode['timestep'] + 1))

            # Check if the vertex generated is within the dimension of the map 
            if not in_map(my_map, childNode_loc):
                # print("[WARN] Genarated vertex not in map dimension, skipping...")
                continue

            # Checking of the vertex position is an obstacle 
            if my_map[childNode_loc[0]][childNode_loc[1]]:
                # print(f"[WARN] {(childNode_loc[0],childNode_loc[1]]} The vertex generated is an obstacle, "
                #       "skipping...")
                continue

            # Check for Vertex and Edge collision
            if is_constrained(currNode['loc'], childNode_loc, currNode['timestep'] + 1,
                              constraint_lookupTable[str(agent)] if str(agent) in constraint_lookupTable else None):
                continue

            # Creating a child node 
            childNode = {'loc': childNode_loc,
                         'g_val': currNode['g_val'] + 1,
                         'h_val': h_values[childNode_loc],
                         'timestep': currNode['timestep'] + 1,
                         'parent': currNode}
            # # print("Child Node: ", childNode)

            # Check if the child node is in the closed list 
            if (childNode['loc'], childNode['timestep']) in closed_list:
                existing_node = closed_list[(childNode['loc'], childNode['timestep'])]
                if compare_nodes(childNode, existing_node):
                    # print("[DEBUG] (A-star) Child already in closed list, so updating and adding...")
                    closed_list[(childNode['loc'], childNode['timestep'])] = childNode
                    push_node(open_list, childNode)
                else:
                    # print("[WARN] Child already on the list, skipping...")
                    pass
            # if not add it to the open list 
            else:
                # print("[INFO] (A-star) Adding child to the open lists.")
                closed_list[(childNode['loc'], childNode['timestep'])] = childNode
                push_node(open_list, childNode)

    ##############################
    return None  # Failed to find solutions
