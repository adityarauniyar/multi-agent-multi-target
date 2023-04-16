from mdgym.envs.multi_drone_multi_actor.mudmaf_env import MUDMAFEnv
from mdgym.utils.types import *
import numpy as np
from MAPFSolvers.cbs import CBSSolver

import logging


class MAPFFilming(MUDMAFEnv):
    def __init__(
            self,
            num_agents: int = 1,
            num_actors: int = 1,
            observation_size: int = 10,
            world0: np.ndarray | None = None,
            grid_size: float = 1.0,
            size: TwoIntTuple = (10, 40),
            obstacle_prob_range=(0, .5),
            full_help=False,
            blank_world=False,
            SEED: int = 123
    ):
        super().__init__(
            num_agents=num_agents,
            num_actors=num_actors,
            observation_size=observation_size,
            world0=world0,
            grid_size=grid_size,
            size=size,
            obstacle_prob_range=obstacle_prob_range,
            full_help=full_help,
            blank_world=blank_world,
            SEED=SEED
        )

    @property
    def next_step(self):
        pass

    @property
    def list_of_obstacles(self):
        """Returns a list of tuples representing the coordinates of all obstacles in the obstacle map."""
        obstacle_map = self.obstacle_map_initial

        obstacle_coords = np.argwhere(obstacle_map == 1)
        obstacle_coords = [(i, j) for i, j in obstacle_coords]
        return obstacle_coords

    def get_cbs_mapf(self) -> List[List[Tuple[ThreeIntTuple]]]:
        current_agents_start_pos = self.get_agent_positions()
        current_agents_start_pos_xy = [(x, y) for x, y, _ in current_agents_start_pos]

        current_actors_pos = self.get_actors_position()
        current_actors_pos_xy = [(x, y) for x, y, _ in current_actors_pos]

        # Assigning agents to actors
        for actor_id in range(self.num_agents):
            self.logger.debug(
                f"Current Agent {actor_id} tacks {self.world.agents_state.drones[actor_id].current_actor_id}")
            self.world.agents_state.drones[actor_id].current_actor_id = actor_id
        # Assigning and its goal location for optimal viewing angle
        current_agents_goal_pos = current_actors_pos
        current_agents_goal_pos_xy = [(x, y) for x, y, _ in current_agents_goal_pos]

        # Getting the cbs paths from this start and goal locations
        self.logger.info(f"Getting CBS MAPF paths for \nAgents Starting at {current_agents_start_pos_xy}"
                         f", and\n Actors at {current_actors_pos_xy}...")
        self.logger.info(f"Obstacle map as:  \n{self.obstacle_map_initial}")

        cbs_planner = CBSSolver(self.obstacle_map_initial,
                                starts=current_agents_start_pos_xy,
                                goals=current_agents_goal_pos_xy)

        # Get the x y paths from start to goal location
        paths_xy = cbs_planner.find_solution()
        self.logger.debug(f"Current Agents paths WITHOUT orientation= \n{paths_xy}")

        # Update the paths to contain the angle with respect to its corresponding actor assignment
        paths_xyo = paths_xy.copy()
        for agent_id in range(self.num_agents):
            tracking_actor_id = self.world.get_agent_to_actor_id_tracking_id(agent_id)
            for ts in range(len(paths_xy[agent_id])):
                x, y = paths_xy[agent_id][ts][0], paths_xy[agent_id][ts][1]
                actor_x, actor_y, _ = self.world.get_actor_position_by_id(tracking_actor_id)

                dy = float(actor_y - y)
                dx = float(actor_x - x)

                orientation_in_deg = int(np.rad2deg(np.arctan2(dy, dx)))
                paths_xyo[agent_id][ts] = (x, y, orientation_in_deg)
                print(paths_xyo[agent_id][ts])

        self.logger.info(f"Paths with orientation are: {paths_xyo}")

        # Return the paths to the main_mapf engine
        return paths_xyo


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    n_agents = 3
    n_actors = 3

    SEED = 123

    env = MAPFFilming(num_agents=n_agents,
                      num_actors=n_actors,
                      world0=None,
                      grid_size=1.0,
                      size=(20, 20),
                      obstacle_prob_range=(.1, .2),
                      SEED=SEED
                      )

    env.get_cbs_mapf()

    # print(coordinationRatio(env))
