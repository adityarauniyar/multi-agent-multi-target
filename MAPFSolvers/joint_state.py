import time as timer
import numpy as np
from single_agent_planner import compute_heuristics, get_sum_of_cost, push_node, pop_node, all_in_map, get_path, is_valid_motion, compare_nodes


class JointStateSolver(object):
    """A planner that plans for all robots together."""

    def __init__(self, my_map, starts, goals):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """

        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)

        self.CPU_time = 0

        # compute heuristics for the low-level search
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))
    
    def move_joint_state(self, locs, dir):

        new_locs = []
        for i in range(len(locs)):
            new_locs.append((locs[i][0] + dir[i][0], locs[i][1] + dir[i][1]))
        return new_locs

    def generate_motions_recursive(self, num_agents,cur_agent):
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]
        
        joint_state_motions = []

        if cur_agent == num_agents:
            return joint_state_motions

        for dir in directions:
            next_motion = [dir]
            motions = self.generate_motions_recursive(num_agents,cur_agent+1)
            if len(motions) == 0:
                joint_state_motions.append(next_motion)
            else:
                for motion in motions:
                    joint_state_motions.append(next_motion + motion)
        return joint_state_motions
    
    def joint_state_a_star(self, my_map, starts, goals, h_values, num_agents):
        """ my_map             - binary obstacle map
            starts             - start positiona
            goals              - goal positions
            h_values           - heuristics values for agent and its vertex
            num_agents         - total number of agents in the map 
        """
        
        """
        In principle, one can find an optimal collision-free solution for a MAPF instance by planning for all
        agents simultaneously in joint location space by finding a shortest path on a graph whose vertices
        correspond to tuples of cells, namely one for each agent. Figure 3 shows this graph for our example
        MAPF instance. However, the number of vertices of the graph grows exponentially with the number
        of agents, which makes this search algorithm too slow in practice. Therefore, one needs to develop
        search algorithms that exploit the problem structure better to gain efficiency. We now discuss two
        such search algorithms, namely prioritized planning and conflict-based search.
        """

        ##############################
        #  Extend the A* search to search in the joint configurations of agents.
        print("Map: ", my_map)
        print("Starts: ", starts)
        print("Goals: ", goals)
        print("Heuristics: ", h_values)
        print("Number of agents: ", num_agents)
        
        # Creating Open and closed list 
        open_list = []
        closed_list = dict()
        
        # Generate root node 

        root = {'loc': tuple(starts), 
                'g_val': 0,
                'h_val': np.sum(np.array([h_values[agentID][starts[agentID]] for agentID in range(num_agents)])),
                'parent': None}
                       
        # Add root node to the open list
        push_node(open_list, root)
        closed_list[root['loc']] = root      
        print("Root: ", root)
        print("Open List: ", open_list)         
        
        # Iterate over open list
        print("\nStarting Joint-State Iteration...")
        while len(open_list) > 0:            
            # Remove N while smallest N.f ( = N.g + N.h) from Open 
            currNode = pop_node(open_list=open_list)
            print("Current Node = ", currNode)
            # if N.s = goal, then return the corresponding path 
            print("Status(currNode['loc'] == tuple(goals)): ", currNode['loc'] == tuple(goals))
            if currNode['loc'] == tuple(goals):
                print("GOALLLLLL !!!")
                return get_path(currNode)
            
            # Getting exponentially growing motions 
            directions = self.generate_motions_recursive(num_agents, 0)
            print("Motions({}): ".format(len(directions)))
            
            # For every successor's of N.s with f = N.                  
            for dir in directions:
                moved_joint_state_loc = self.move_joint_state(currNode['loc'],dir)
                childNode_loc = tuple(moved_joint_state_loc)
                print("Moved Joint State: ", moved_joint_state_loc)
                
                is_child_valid = True
                
                # Check if the vertex is on the map 
                if not all_in_map(my_map, moved_joint_state_loc):
                    print("Genarated vertex not in map dimension.")
                    continue
                
                # Check if the vertex generated is an obstacle
                for agentID in range(num_agents):
                    if my_map[childNode_loc[agentID][0]][childNode_loc[agentID][1]]:
                        print("Obstacle occupancy, IGNORING...")
                        is_child_valid = False
                        break 
                if not is_child_valid: 
                    continue
                
                # If moving from 𝑁. 𝑠 to 𝑠′ lead to a conflict, then continue
                if not is_valid_motion(currNode['loc'], moved_joint_state_loc): 
                    print("Motion leads to conflict...")                   
                    continue
                    
                    
                # Generate child node 𝑁!= 𝑠 = 𝑠!, 𝑔 = 𝑁. 𝑠 + 𝑐 𝑁. 𝑠, 𝑠!, ℎ = ℎ 𝑠!                
                childNode = {'loc': childNode_loc, 
                             'g_val': currNode['g_val'] + 1,
                             'h_val': np.sum(np.array([h_values[agentID][moved_joint_state_loc[agentID]] for agentID in range(num_agents)])),
                             'parent': currNode}
                # print("Child Node: ", childNode)
                
                # If  there is an existing node 𝑁"#$with 𝑁"#$. 𝑠 = 𝑁!. s
                if childNode_loc in closed_list:
                    existing_node = closed_list[(childNode_loc)]
                    # If 𝑁"#$∈ 𝑂𝑝𝑒𝑛 and𝑁!. 𝑓 < 𝑁"#$. 𝑓 then 𝑁"#$← 𝑁′
                    if compare_nodes(childNode, existing_node):
                        closed_list[(childNode_loc)] = childNode
                        push_node(open_list, childNode)
                # Else insert 𝑁!into 𝑂𝑝𝑒𝑛
                else:
                    print("Adding Child Node...")
                    closed_list[(childNode_loc)] = childNode
                    push_node(open_list, childNode)
        
        ##############################

        return None  # Failed to find solutions

    def find_solution(self):
        """ Finds paths for all agents from their start locations to their goal locations."""

        start_time = timer.time()
        result = []

        path = self.joint_state_a_star(self.my_map, self.starts, self.goals, self.heuristics,self.num_of_agents)

        if path is None:
                raise BaseException('No solutions')
                
        # Convert the path to a list of paths for each agent
        for i in range(self.num_of_agents):
            result.append([])
            for node in path:
                result[i].append(node[i])
        # Delete duplicate goal positions
        final_paths = []
        for path in result: 
            goal = None
            num_delete = 0
            for point in reversed(path):
                if goal == None:
                    goal = point
                elif point == goal:
                    num_delete = num_delete + 1
                else:
                    break 
            if num_delete > 0:
                path = path[:-num_delete]
            final_paths.append(path)

        self.CPU_time = timer.time() - start_time

        print("\n Found a solution! \n")
        print("CPU time (s):    {:.2f}".format(self.CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(final_paths)))

        return final_paths
