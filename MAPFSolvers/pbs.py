import time as timer
import heapq
from collections import deque
import random
from single_agent_planner import compute_heuristics, a_star, get_location, get_sum_of_cost
from topological_sort import TopologyGraph
from cbs import detect_collisions 
import numpy as np

def generate_priority_pairs(collision):
    """
    When a collision occurs between two agents, PBS explores both scenarios where one is prioritized
    over another and vice versa. In particular, when PBS expands a PT node N to resolve a collision,
    PBS generates two child PT nodes N1 nd N2 that correspond to the ordered priority pairs j ≺ i
    and i ≺ j.
    """
    
    priority_pairs = []
    ##############################
    #  Generate a priority pair from a collision
    # Collision : [{'a1': 0, 'a2': 1, 'loc': [(1, 4)], 'timestep': 3}]    
    
    # print("[DEBUBG] (generate_priority_pairs) Generating priority pairs from collision: ", collision)
    if collision is not None:
        priority_pairs.append((collision['a2'], collision['a1']))
        priority_pairs.append((collision['a1'], collision['a2']))
    # print("[DEBUBG] (generate_priority_pairs) Priority pairs: ", priority_pairs)
    ##############################

    return priority_pairs

def get_lower_priority_agents(priority_pairs, agent):
    tg = TopologyGraph(directed=True)
    tg.clear_graph()

    # construct graph
    for pair in priority_pairs:
        tg.Edge(pair[0], pair[1])

    if not tg.has_node(agent):
        return [agent]

    # Get the nodes behind a given node in a topological ordering
    return tg.get_subsequent_nodes_in_topological_ordering(agent)

def get_higher_priority_agents(priority_pairs, agent):
    tg = TopologyGraph(directed=True)
    tg.clear_graph()

    # construct graph
    for pair in priority_pairs:
        tg.Edge(pair[1], pair[0])

    if not tg.has_node(agent):
        return [agent]

    # Get the nodes behind a given node in a topological ordering
    return tg.get_subsequent_nodes_in_topological_ordering(agent)

def collide_with_higher_priority_agents(node, agent):
        collisions = node['collisions']
        priority_pairs = node['priority_pairs']

        if collisions == [] or priority_pairs == []:
            # print("[DEBUG] (PBS::collide_with_higher_priority_agent) Collisions or priority_pairs EMPTY.")
            return []

        higher_priority_agents = get_higher_priority_agents(node['priority_pairs'], agent)

        for collision in collisions:
            if collision['a1'] == agent and collision['a2'] in higher_priority_agents:
                return True
            elif collision['a2'] == agent and collision['a1'] in higher_priority_agents:
                return True

        return False 

class PBSSolver(object):
    """The high-level search of PBS."""

    def __init__(self, my_map, starts, goals):
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
        self.search_stack = deque()

        # compute heuristics for the low-level search
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

    def push_node(self, node):
        # Max heap
        heapq.heappush(self.open_list, (-node['cost'], -len(node['collisions']), self.num_of_generated, node))
        self.num_of_generated += 1

    def pop_node(self):
        _, _, id, node = heapq.heappop(self.open_list)
        self.num_of_expanded += 1
        return node
            
    def update_plan(self, node, ai):
        ##############################
        #  High-Level Search
        
        # LIST ← topological sorting on partially ordered set ({i} ∪ {j|i ≺N j}) // i and all js
        # that are having lower priorities than i specified by N.priority_pairs
        
        num_open_spaces = np.count_nonzero(np.array(self.my_map) == False)
        
        lower_priority_agents_than_ai = get_lower_priority_agents(node['priority_pairs'], ai)
        # print("[DEBUG] (PBS::udpate_plan) Lower priority Agents than ({}): ".format(ai), lower_priority_agents_than_ai)
                
        for j in lower_priority_agents_than_ai:
            
            # if j = i or aj collides with ak in N.paths, where k ≺N j then
            # print("[DEBUG] Current agent ({}) with lower prioirty than agent ({})...".format(j, ai))
            if j == ai or collide_with_higher_priority_agents(node, agent=j):
                
                higher_priority_agents_than_aj = get_higher_priority_agents(node['priority_pairs'], agent=j)
                # print("[DEBUG] (PBS::udpate_plan) Higher priority Agents than ({}): ".format(j), higher_priority_agents_than_aj)
                
                # Example constraint: [{'agent': 0, 'loc': [(1, 4)], 'timestep': 3}, {'agent': 1, 'loc': [(1, 4)], 'timestep': 3}]
                constraints = []
                
                # Creating constraint of higher_priority_agents paths 
                for higher_priority_agent in higher_priority_agents_than_aj:
                    if higher_priority_agent < len(node['paths']) and j is not higher_priority_agent:
                        for ts in range(2 * num_open_spaces):
                            location = get_location(node['paths'][higher_priority_agent], ts)
                            constraint = {
                                'agent'     : higher_priority_agent,
                                'loc'       : [location],
                                'timestep'  : ts                                
                            }
                            constraints.append(constraint)
                # print("[INFO] (PBS) Current stack constraints: ", constraints)
                new_path = a_star(self.my_map, self.starts[j], self.goals[j], self.heuristics[j], j, constraints)
                
                # print("[CRITICAL] (PBS) New path for agent ({}):".format(j), new_path)
                if new_path is None or new_path is []:
                    return False
                
                # Updating the new paths to the agent in the older ones
                if j < len(node['paths']):
                    node['paths'][j] = new_path.copy()
                else:
                    node['paths'].append(new_path)
                        
        # print("[DEBUG] (PBS::udpate_plan) Node after updating the path: ", node)               
        
        ##############################
        
        return True


    def find_solution(self):
        """ Finds paths for all agents from their start locations to their goal locations
        """

        # print('Start PBS')
        self.start_time = timer.time()

        # print("[INFO] (PBS) Starts: ", self.starts)
        # print("[INFO] (PBS) Goals: ", self.goals)
        # print("[INFO] (PBS) My Map: ", self.my_map)
        # print("[INFO] (PBS) Number of agents: ", self.num_of_agents)

        # Generate the root node
        # priority_pairs   - list of priority pairs
        # paths         - list of paths, one for each agent
        #               [[(x11, y11), (x12, y12), ...], [(x21, y21), (x22, y22), ...], ...]
        # collisions     - list of collisions in paths
        root = {'cost': 0,
                'priority_pairs': [],
                'paths': [],
                'collisions': []}
        for i in range(self.num_of_agents):  # Find initial path for each agent
            self.update_plan(root, i)

        root['cost'] = get_sum_of_cost(root['paths'])
        root['collisions'] = detect_collisions(root['paths'])
        self.search_stack.append(root)
        self.num_of_generated += 1

        ##############################
        #  High-Level Search
        #           Repeat the following as long as the search_stack is not empty:
        #             1. Get the next node from the search_stack (you can use self.search_stack.pop()
        #             2. If this node has no collision, return solution
        #             3. Otherwise, choose the first collision and generate a priority pair (using your
        #                generate_priority_pairs function). Add a new child node to your search_stack for each constraint
        #           Ensure to create a copy of any objects that your child nodes might inherit
        
        # print("[INFO] (PBS) Iterating while the stack is not empty") 
        while self.search_stack:
            CPU_time_s = timer.time() - self.start_time
            currNode = self.search_stack.pop()
            self.num_of_expanded += 1
            # print("[DEBUG] (PBS) Current Node: ", currNode)
            
            if len(currNode['collisions']) == 0 or CPU_time_s >= 600:
                # print("[ALARM] There is no collisions now, terminating...")
                self.print_results(currNode)
                return currNode['paths']
            
            collision = currNode['collisions'][0]
            # print("[DEBUG] (PBS) First vertex or edge collision in currNode.collisisons: ", collision)
            # Example Collision : [{'a1': 0, 'a2': 1, 'loc': [(1, 4)], 'timestep': 3}]
            
            # Suppose the agents involved in collision is 0 and 1 indexed
            for priority_pair in generate_priority_pairs(collision):
                # agent = collision['a1']
                # if ai == 1:
                #     agent = collision['a2']
                agent = priority_pair[0]
                
                new_node = {'cost': 0,
                            'priority_pairs': [],
                            'paths': currNode['paths'].copy(),
                            'collisions': currNode['collisions'].copy()}
                
                if (currNode['priority_pairs'] is not None):
                    new_node['priority_pairs'] = currNode['priority_pairs'].copy()
                    new_node['priority_pairs'].append(priority_pair)
                    
                # print("[DEBUG] (PBS) New RAW node with the constraint: ", new_node)
                # print("[DEBUG] (PBS) agent = ", agent)                    
                
                success = self.update_plan(new_node, agent)
                
                if success:
                    new_node['cost'] = get_sum_of_cost(new_node['paths'])
                    new_node['collisions'] = detect_collisions(new_node['paths'])
                    # print("[INFO] (PBS) Successfully updated the plan with NODE: ", new_node)
                    self.num_of_generated += 1
                    self.search_stack.append(new_node)
            
            
        
        
        ##############################
        raise BaseException('No solutions')


    def print_results(self, node):
        print("\n Found a solution! \n")
        CPU_time = timer.time() - self.start_time
        print("CPU time (s):    {:.2f}".format(CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(node['paths'])))
        print("Expanded nodes:  {}".format(self.num_of_expanded))
        print("Generated nodes: {}".format(self.num_of_generated))



if __name__ == '__main__':
    solver = PBSSolver([], [], [])

    node1 = {'cost': 300,
            'priority_pairs': [],
            'paths': [],
            'collisions': []}
    
    node2 = {'cost': 200,
            'priority_pairs': [],
            'paths': [],
            'collisions': []}
    
    node3 = {'cost': 100,
            'priority_pairs': [],
            'paths': [],
            'collisions': []}

    solver.push_node(node1)
    solver.push_node(node2)
    solver.push_node(node3)

    print(solver.pop_node())
    print(solver.pop_node())
    print(solver.pop_node())
