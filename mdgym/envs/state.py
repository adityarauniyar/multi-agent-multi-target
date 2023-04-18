from typing import Tuple

import numpy as np

from mdgym.spaces.state import State
from mdgym.spaces.drone_space import DronesSpace
from mdgym.utils.types import TwoIntTupleList, TwoIntTuple, ThreeIntTuple, List, AgentType

import logging


class WorldState:
    """
    WorldState
    """

    def __init__(
            self,
            grid_size: float = 1.0,
            operational_map: np.ndarray = np.ones((3, 3)),
            agents_start_position: List[ThreeIntTuple] | None = None,
            actors_start_position: List[ThreeIntTuple] | None = None,
            actors_goal_position: List[ThreeIntTuple] | None = None,
            viewing_angle: float = 90.0,
            viewing_range: float = 15.0,
            observation_space_size: int = 10,
            num_agents: int = 1,
            num_actors: int = 1,
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Starting the WorldState with following parameters: num_agents = {num_agents},\n "
                         f"observation Size = {observation_space_size}, world size = {operational_map.size}, "
                         f"grid size ={grid_size},"
                         f" Actors position length = {len(actors_start_position) if actors_start_position is not None else 0}, ")

        if actors_start_position is None:
            actors_start_position = [(0, 0, 0)]
        if agents_start_position is None:
            agents_start_position = [(0, 0, 0)]

        self.goals = actors_start_position.copy()
        self.num_agents = num_agents
        self.num_actors = num_actors

        self.operational_map = operational_map
        self.logger.debug(f"Operational map @ World State:\n{self.operational_map}")

        self.agents_state = DronesSpace(
            grid_size=grid_size,
            operational_map=operational_map,
            start_positions=agents_start_position,
            goal_positions=None,
            viewing_angle=viewing_angle,
            viewing_range=viewing_range,
            observation_space_size=observation_space_size,
            num_agents=num_agents
        )

        self.actors_state = [State(
            grid_size=grid_size,
            operational_map=operational_map,
            start_position=actors_start_position[actor_id],
            goal_position=actors_start_position[actor_id]
            if actors_goal_position is None else actors_goal_position[actor_id],
            agent_type=AgentType.ACTOR,
            agent_id=actor_id
        ) for actor_id in range(self.num_actors)]

        # Define the path of each actor to be traveled
        self.actors_paths = None

        assert (len(self.agents_curr_pos) == num_agents)

    @property
    def agents_curr_pos(self) -> List[ThreeIntTuple]:
        agents_curr_pos = [(-1, -1, -1) for _ in range(self.num_agents)]
        for agent_id in range(self.num_agents):
            agents_curr_pos[agent_id] = self.agents_state.drones[agent_id].current_position

        return agents_curr_pos

    @property
    def agents_prev_pos(self) -> List[ThreeIntTuple]:
        agents_prev_pos = [(-1, -1, -1) for _ in range(self.num_agents)]
        for agent_id in range(self.num_agents):
            agents_prev_pos[agent_id] = self.agents_state.drones[agent_id].previous_position

        return agents_prev_pos

    @property
    def agents_goal_pos(self) -> List[ThreeIntTuple]:
        agents_goal_pos = [(-1, -1, -1) for _ in range(self.num_agents)]
        for agent_id in range(self.num_agents):
            agents_goal_pos[agent_id] = self.agents_state.drones[agent_id].goal_position

        return agents_goal_pos

    def get_agents_position_by_id(self, agent_id) -> ThreeIntTuple:
        return self.agents_curr_pos[agent_id]

    def get_agent_past_position_by_id(self, agent_id) -> ThreeIntTuple:
        return self.agents_prev_pos[agent_id]

    def get_agent_goal_by_id(self, agent_id) -> ThreeIntTuple:
        return self.agents_goal_pos[agent_id]

    def get_agents_to_actors_tracking_id(self) -> List[int | None]:
        agents_to_actors_tracking_ids = [None for _ in range(self.num_agents)]

        for agent_id in range(self.num_agents):
            agents_to_actors_tracking_ids[agent_id] = self.agents_state.drones[agent_id].assigned_to

        return agents_to_actors_tracking_ids

    def get_agent_to_actor_id_tracking_id(self, agent_id: int) -> BaseException | int | None:
        if agent_id not in range(self.num_agents):
            return BaseException("Invalid agent id")
        # TODO: Update this function to return the corresponding actor assigned to an agent that is the drone
        return self.agents_state.drones[agent_id].assigned_to

    @property
    def actors_curr_pos(self) -> List[ThreeIntTuple]:
        actors_curr_pos = [(-1, -1, -1) for _ in range(self.num_actors)]
        for actor_id in range(self.num_actors):
            actors_curr_pos[actor_id] = self.actors_state[actor_id].current_position

        return actors_curr_pos

    @property
    def actors_prev_pos(self) -> List[ThreeIntTuple]:
        actors_prev_pos = [(-1, -1, -1) for _ in range(self.num_actors)]
        for actor_id in range(self.num_actors):
            actors_prev_pos[actor_id] = self.actors_state[actor_id].previous_position

        return actors_prev_pos

    @property
    def actors_goal_pos(self) -> List[ThreeIntTuple]:
        actors_goal_pos = [(-1, -1, -1) for _ in range(self.num_actors)]
        for actor_id in range(self.num_actors):
            actors_goal_pos[actor_id] = self.actors_state[actor_id].goal_position

        return actors_goal_pos

    def get_actor_position_by_id(self, actor_id: int) -> BaseException | tuple[int, int, int]:
        if actor_id not in range(self.num_actors):
            return BaseException("actor_id not in range")
        return self.actors_state[actor_id].current_position

    def get_actor_past_position_by_id(self, actor_id: int) -> BaseException | tuple[int, int, int]:
        if actor_id not in range(self.num_actors):
            return BaseException("actor_id not in range")
        return self.actors_state[actor_id].previous_position

    def get_actor_goal_by_id(self, actor_id: int) -> BaseException | tuple[int, int, int]:
        if actor_id not in range(self.num_actors):
            return BaseException("actor_id not in range")
        return self.actors_state[actor_id].goal_position

    def get_unassigned_agents(self) -> None | List[int]:
        available_agents = []
        for agent_id in range(self.num_agents):
            if not self.agents_state.drones[agent_id].is_assigned:
                available_agents.append(agent_id)

        return available_agents

    def get_assigned_agents(self) -> None | List[int]:
        available_agents = []
        for agent_id in range(self.num_agents):
            if self.agents_state.drones[agent_id].is_assigned:
                available_agents.append(agent_id)

        return available_agents

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
        last_position_with_orient = self.get_agents_position_by_id(agent_id)
        last_position = (last_position_with_orient[0], last_position_with_orient[1])

        for agent in range(0, self.num_agents):
            if agent == agent_id:
                continue
            a_past_with_orient = self.get_agent_past_position_by_id(agent)
            a_past = (a_past_with_orient[0], a_past_with_orient[1])

            a_pres_with_orient = self.get_agents_position_by_id(agent)
            a_pres = (a_pres_with_orient[0], a_pres_with_orient[1])
            if collide(a_past, a_pres, last_position, new_position):
                return True
        return False

    # try to move agent and return the status
    def move_agent(self, new_position: ThreeIntTuple, agent_id: int) -> int:
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
            self.agents_state.drones[agent_id].move_to(new_position=new_position)
            if new_position == self.agents_goal_pos[agent_id]:
                self.logger.debug(f"Agent{agent_id} is at the GOAL location{new_position} and moved to it. ")
                return 1
            else:
                self.logger.debug(f"Agent{agent_id} moved to new location{new_position}")
                return 0

        # Check and update if it is a valid move
        if not self.agents_state.drones[agent_id].is_valid_action(new_position=new_position):
            self.logger.warning(f"Agent{agent_id} is out of bounds for new location {new_position}.")
            return -1

        # Check and update if it doesn't collide with any other agents
        if (new_position[0], new_position[1]) in self.agents_curr_pos:
            self.logger.warning(f"Agent{agent_id} led to vertex collision with new location {new_position}.")
            return -3
            # This check is fine as we have already checked if the new_position is the agents current position

        # Check if there is no edge collision that includes diagonals between agents
        # if self.diagonal_collision(agent_id=agent_id, new_position_with_orient=new_position):
        #     self.logger.warning(f"Agent{agent_id} led to diagonal collision with new location {new_position}.")
        #     return -3

        # All the valid move conditions
        # Agent just reaching the goal location
        assert (self.agents_state.drones[agent_id].move_to(new_position=new_position) == True)

        if new_position == self.agents_goal_pos[agent_id]:
            self.logger.debug(f"Agent{agent_id} is at the GOAL location{new_position} and moved to it. ")
            return 1

        elif self.agents_state.drones[agent_id].previous_position == self.agents_state.drones[agent_id].goal_position:
            self.logger.debug(f"Agent{agent_id} moved to new location{new_position} from goal location.")
            return 2

        else:
            self.logger.debug(f"Agent{agent_id} moved to new location{new_position}")
            return 0

    def get_translated_pos(self, agent_id: int, action: int) -> ThreeIntTuple:
        return self.agents_state.drones[agent_id].get_new_translated_position_by_seq(action)

    def get_rotated_pos(self, agent_id: int, action: int) -> ThreeIntTuple:
        return self.agents_state.drones[agent_id].get_new_rotated_position_by_seq(action)

    def act(self, agent_id: int, translation_seq: int, rotation_seq: int = None) -> int:
        # 0     1   2   3   4   5   6   7   8
        # still E   W   N   S   NE  NW  SE  SW
        new_pos = self.get_translated_pos(agent_id, translation_seq)

        # Check if rotation is needed as action
        if rotation_seq is not None:
            new_pos[2] += self.agents_state.drones[agent_id].rotation_dirs[rotation_seq]

        moved_res = self.move_agent(new_pos, agent_id)
        return moved_res

    # Compare with a plan to determine job completion
    def done(self) -> bool:
        num_complete = 0
        for i in range(self.num_agents):
            if self.agents_state.drones[i].current_position == self.agents_state.drones[i].goal_position:
                num_complete += 1
        return num_complete == self.num_agents  # , numComplete/float(len(self.agents))
