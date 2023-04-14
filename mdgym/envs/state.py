import numpy as np

from mdgym.spaces.agentstate import AgentState
from mdgym.spaces.drone_space import DronesSpace
from mdgym.utils.types import TwoIntTupleList, TwoIntTuple, ThreeIntTuple, List

import logging


class WorldState:
    """
    WorldState
    """

    def __init__(
            self,
            grid_size: float = 1.0,
            operational_map: np.ndarray = np.ones((3, 3)),
            start_positions: List[ThreeIntTuple] | None = None,
            goal_positions: List[ThreeIntTuple] | None = None,
            viewing_angle: float = 90.0,
            viewing_range: float = 15.0,
            observation_space_size: int = 10,
            num_agents: int = 1
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Starting the WorldState with following parameters: num_agents = {num_agents},\n "
                         f"observation Size = {observation_space_size}, world size = {operational_map.size}, "
                         f"grid size ={grid_size},"
                         f" Actors position length = {len(goal_positions) if goal_positions is not None else 0}, ")

        if goal_positions is None:
            goal_positions = [(0, 0, 0)]
        if start_positions is None:
            start_positions = [(0, 0, 0)]

        self.goals = goal_positions.copy()
        self.num_agents = num_agents
        self.operational_map = operational_map
        self.logger.debug(f"Operational map @ World State:\n{self.operational_map}")

        self.state = DronesSpace(
            grid_size=grid_size,
            operational_map=operational_map,
            start_positions=start_positions,
            goal_positions=goal_positions,
            viewing_angle=viewing_angle,
            viewing_range=viewing_range,
            observation_space_size=observation_space_size,
            num_agents=num_agents
        )

        assert (len(self.agents_curr_pos) == num_agents)

    @property
    def agents_curr_pos(self) -> List[ThreeIntTuple]:
        agents_curr_pos = [(-1, -1, -1) for _ in range(self.num_agents)]
        for agent_id in range(self.num_agents):
            agents_curr_pos[agent_id] = self.state.drones[agent_id].current_position

        return agents_curr_pos

    @property
    def agents_prev_pos(self) -> List[ThreeIntTuple]:
        agents_prev_pos = [(-1, -1, -1) for _ in range(self.num_agents)]
        for agent_id in range(self.num_agents):
            agents_prev_pos[agent_id] = self.state.drones[agent_id].previous_position

        return agents_prev_pos

    @property
    def agents_goal_pos(self) -> List[ThreeIntTuple]:
        agents_goal_pos = [(-1, -1, -1) for _ in range(self.num_agents)]
        for agent_id in range(self.num_agents):
            agents_goal_pos[agent_id] = self.state.drones[agent_id].goal_position

        return agents_goal_pos

    def get_position_by_id(self, agent_id) -> ThreeIntTuple:
        return self.agents_curr_pos[agent_id]

    def get_past_position_by_id(self, agent_id) -> ThreeIntTuple:
        return self.agents_prev_pos[agent_id]

    def get_goal_by_id(self, agent_id) -> ThreeIntTuple:
        return self.agents_goal_pos[agent_id]

    def diagonal_collision(self, agent_id: int, new_position_with_orient: ThreeIntTuple):
        """
        diagonalCollision(id,(x,y)) returns true if agent with id "id" collided diagonally with
        any other agent in the state after moving to coordinates (x,y)
        agent_id: id of the desired agent to check for
        newPos: coord the agent is trying to move to (and checking for collisions)
        """

        #        def eq(f1,f2):return abs(f1-f2)<0.001
        def collide(a1: TwoIntTuple, a2: TwoIntTuple, b1: TwoIntTuple, b2: TwoIntTuple):
            """
            a1,a2 are coords for agent 1, b1,b2 coords for agent 2, returns true if these collide diagonally
            """
            return np.isclose((a1[0] + a2[0]) / 2., (b1[0] + b2[0]) / 2.) and np.isclose((a1[1] + a2[1]) / 2.,
                                                                                         (b1[1] + b2[1]) / 2.)

        assert (len(new_position_with_orient) == 3);
        new_position = (new_position_with_orient[0], new_position_with_orient[1])

        # up until now we haven't moved the agent, so getPos returns the "old" location
        last_position_with_orient = self.get_position_by_id(agent_id)
        last_position = (last_position_with_orient[0], last_position_with_orient[1])

        for agent in range(0, self.num_agents):
            if agent == agent_id:
                continue
            a_past_with_orient = self.get_past_position_by_id(agent)
            a_past = (a_past_with_orient[0], a_past_with_orient[1])

            a_pres_with_orient = self.get_position_by_id(agent)
            a_pres = (a_pres_with_orient[0], a_pres_with_orient[1])
            if collide(a_past, a_pres, last_position, new_position):
                return True
        return False

    # try to move agent and return the status
    def move_agent(self, new_position, agent_id) -> int:
        """
        try to execute action and return whether action was executed or not and why
        returns:
            2: action executed and left goal
            1: action executed and reached goal (or stayed on)
            0: action executed
           -1: out of bounds
           -2: collision with wall
           -3: collision with robot
        """

        # Moving to the same place OR staying still
        if new_position == self.agents_curr_pos[agent_id]:
            self.state.drones[agent_id].move_to(new_position=new_position)
            if new_position == self.agents_goal_pos[agent_id]:
                self.logger.debug(f"Agent{agent_id} is at the GOAL location{new_position} and moved to it. ")
                return 1
            else:
                self.logger.debug(f"Agent{agent_id} moved to new location{new_position}")
                return 0

        # Check and update if it is a valid move
        if not self.state.drones[agent_id].is_valid_action(new_position=new_position):
            return -1

        # Check and update if it doesn't collide with any other agents
        if (new_position[0], new_position[1]) in self.agents_curr_pos:
            return -3
            # This check is fine as we have already checked if the new_position is the agents current position

        # Check if there is no edge collision that includes diagonals between agents
        if self.diagonal_collision(agent_id=agent_id, new_position_with_orient=new_position):
            return -3

        # All the valid move conditions
        # Agent just reaching the goal location
        assert (self.state.drones[agent_id].move_to(new_position=new_position) == True)

        if new_position == self.agents_goal_pos[agent_id]:
            self.logger.debug(f"Agent{agent_id} is at the GOAL location{new_position} and moved to it. ")
            return 1

        elif self.state.drones[agent_id].previous_position == self.state.drones[agent_id].goal_position:
            self.logger.debug(f"Agent{agent_id} moved to new location{new_position} from goal location.")
            return 2

        else:
            self.logger.debug(f"Agent{agent_id} moved to new location{new_position}")
            return 0

    def get_translated_pos(self, agent_id: int, action: int) -> ThreeIntTuple:
        return self.state.drones[agent_id].get_new_translated_position_by_seq(action)

    def get_rotated_pos(self, agent_id: int, action: int) -> ThreeIntTuple:
        return self.state.drones[agent_id].get_new_rotated_position_by_seq(action)

    def act(self, agent_id: int, translation_seq: int, rotation_seq: int = None) -> int:
        # 0     1   2   3   4   5   6   7   8
        # still E   W   N   S   NE  NW  SE  SW
        new_pos = self.get_translated_pos(agent_id, translation_seq)

        # Check if rotation is needed as action
        if rotation_seq is not None:
            new_pos[2] += self.state.drones[agent_id].rotation_dirs[rotation_seq]

        moved_res = self.move_agent(new_pos, agent_id)
        return moved_res

    # Compare with a plan to determine job completion
    def done(self) -> bool:
        num_complete = 0
        for i in range(self.num_agents):
            if self.state.drones[i].current_position == self.state.drones[i].goal_position:
                num_complete += 1
        return num_complete == self.num_agents  # , numComplete/float(len(self.agents))
