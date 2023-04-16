import time as timer
from single_agent_planner import compute_heuristics, a_star, get_sum_of_cost


class PrioritizedPlanningSolver(object):
    """A planner that plans for each robot sequentially."""

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

    def find_solution(self):
        """ Finds paths for all agents from their start locations to their goal locations."""

        start_time = timer.time()
        result = []
        constraints = []
        
        # print("Obstacle map = ", self.my_map)
        # print("Starts: ", self.starts)
        # print("Goals: ", self.goals)
        # print("num_of_agents: ", self.num_of_agents)

        for i in range(self.num_of_agents):  # Find path for each agent
            CPU_time_s = timer.time() - start_time
            if CPU_time_s >= 600:
                print("TIME EXCEEDED")
                break
            # print("Num constraints {}".format(len(constraints)))
            # print("Constraint:\n ", constraints)
            path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i],
                          i, constraints)
            # print("### Path found for agent {}, from Start({}) to Goal({}): ".format(i, self.starts[i], self.goals[i]), path)
            if path is None:
                raise BaseException('No solutions')
            result.append(path)

            ##############################
            #  Add constraints here
            #         Useful variables:
            #            * path contains the solution path of the current (i'th) agent, e.g., [(1,1),(1,2),(1,3)]
            #            * self.num_of_agents has the number of total agents
            #            * constraints: array of constraints to consider for future A* searches
            # print("Adding constraints...")
            for j in range(i+1, self.num_of_agents):
                if j >= i:
                    # print("Adding constraint for agent {} after processing current agent ({})".format(j,i))
                    for timestep in range(len(self.my_map) * len(self.my_map[0])):
                        constraint = { 'agent': j, 
                                       'loc': [path[min(timestep, len(path)-1)]],
                                       'timestep': timestep}
                        constraints.append(constraint)


            ##############################

        self.CPU_time = timer.time() - start_time

        print("\n Found a solution! \n")
        print("CPU time (s):    {:.2f}".format(self.CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(result)))
        print(result)
        return result
