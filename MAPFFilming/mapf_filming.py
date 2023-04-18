import heapq

from mdgym.envs.multi_drone_multi_actor.mudmaf_env import MUDMAFEnv
from mdgym.utils.types import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from MAPFSolvers.cbs import CBSSolver
import matplotlib.animation as animation
import random
import math

from matplotlib.path import Path
from matplotlib.patches import PathPatch, Polygon

import cv2
import os
import logging


class MAPFFilming(MUDMAFEnv):
    def __init__(
            self,
            num_agents: int = 1,
            num_actors: int = 1,
            agent_viewing_range=15,
            agent_viewing_angle=90,
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

        # Get assigned agents
        assigned_agents = self.world.get_assigned_agents()
        # Get assigned agents start positions
        assigned_agents_start_pos_xy = [(self.world.get_agents_position_by_id(agent_id)[0],
                                         self.world.get_agents_position_by_id(agent_id)[1])
                                        for agent_id in assigned_agents]
        # Get assigned agents goal positions
        assigned_agents_goal_pos_xy = [(self.world.get_agent_goal_by_id(agent_id)[0],
                                        self.world.get_agent_goal_by_id(agent_id)[1])
                                       for agent_id in assigned_agents]

        # Getting the cbs paths from this start and goal locations
        self.logger.info(f"Getting CBS MAPF paths for \nAgents Starting at {assigned_agents_start_pos_xy}"
                         f", and\n Actors at {current_actors_pos_xy}...")
        self.logger.info(f"Obstacle map as:  \n{self.obstacle_map_initial}")

        cbs_planner = CBSSolver(self.obstacle_map_initial,
                                starts=assigned_agents_start_pos_xy,
                                goals=assigned_agents_goal_pos_xy)

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
                # print(paths_xyo[agent_id][ts])

        self.logger.info(f"Paths with orientation are: {paths_xyo}")

        # Return the paths to the main_mapf engine
        return paths_xyo

    def assign_agents_actors_viewpoints(self):

        # TODO: Write code to refresh assignment status and tracking id

        # Get 12 point circular formation of view points locations for each of the actors
        def get_circle_points(grid_location: ThreeIntTuple, radius: int):
            self.logger.debug(f"Calculating the viewpoint formation for center location at {grid_location}")
            center_x, center_y, _ = grid_location
            points = []
            for i in range(12):
                angle = math.radians(30 * i)
                x = int(round(center_x + radius * math.cos(angle)))
                y = int(round(center_y + radius * math.sin(angle)))
                if x < 0 or x >= np.shape(self.obstacle_map_initial)[0] \
                        or y < 0 or y >= np.shape(self.obstacle_map_initial)[1] \
                        or self.obstacle_map_initial[x, y] == 1:
                    self.logger.debug(f"Viewpoint {(x, y)} is out of bounds, skipping...")
                    continue
                points.append((x, y))
            return points

        # Go over each location of each agent and check of the location doesn't lie in non-operational area and
        # assign the closest location to the closest agent and update the agent to be engaged.
        num_actors = self.num_actors

        formation_radius = 4 * self.grid_size

        for actor_id in range(num_actors):
            # Get the formation of viewpoints for the current actor id
            current_actor_pos = self.world.get_actor_position_by_id(actor_id=actor_id)
            formation_viewpoints = get_circle_points(current_actor_pos, formation_radius)
            self.logger.debug(f"Formation points for actor{actor_id}@{current_actor_pos} are {formation_viewpoints}")

            # do this for the viewpoints location for other agents till all the agents are assigned.
            available_agents_id = self.world.get_unassigned_agents()
            self.logger.debug(f"Current available agents are {available_agents_id}")

            # Tuple in format: (path_cost, actor_id, agent_id, formation viewpoint(x,y))
            view_point_allocations = []

            for agent_id in available_agents_id:
                curr_agent_pos_xy = (self.world.agents_state.drones[agent_id].current_position[0],
                                     self.world.agents_state.drones[agent_id].current_position[1])

                current_min_viewpoint = None
                current_min_path_cost = 1e10
                min_viewpoint_index = None
                data_paths_to_viewpoints = []

                for view_point_id in range(len(formation_viewpoints)):

                    current_viewpoint = formation_viewpoints[view_point_id]
                    self.logger.debug(f"current viewpoint is at {current_viewpoint}")

                    cbs_solver = CBSSolver(my_map=self.obstacle_map_initial,
                                           starts=[curr_agent_pos_xy],
                                           goals=[current_viewpoint])

                    path = cbs_solver.find_solution()
                    self.logger.debug(f"Path from CBS: {path}")
                    current_path_cost = len(path[0])

                    if current_path_cost < current_min_path_cost:
                        min_viewpoint_index = view_point_id
                        current_min_viewpoint = current_viewpoint
                        current_min_path_cost = current_path_cost

                data = (current_min_path_cost, actor_id, agent_id, current_min_viewpoint)
                self.logger.debug(f"Inserting data with value: {data}")

                heapq.heappush(view_point_allocations, data)

            self.logger.debug(f"view_point_allocations: {view_point_allocations}")
            path_cost, assign_actor_id, assign_agent_id, assign_viewpoint_xy = heapq.heappop(view_point_allocations)

            self.logger.info(
                f"Assigning: Agent {assign_agent_id} tracks {assign_actor_id} @ {assign_viewpoint_xy}")

            self.world.actors_state[assign_actor_id].is_assigned = True
            self.world.agents_state.drones[assign_agent_id].is_assigned = True
            self.world.actors_state[assign_actor_id].assigned_to = assign_agent_id
            self.world.agents_state.drones[assign_agent_id].assigned_to = assign_actor_id
            self.world.agents_state.drones[assign_agent_id].goal_position = (assign_viewpoint_xy[0],
                                                                             assign_viewpoint_xy[1], 0)

        # TODO: Add logic to track actors such that if agents are more than actors they occupy 90deg viewpoints
        #   and if actors are more than agents, then do ? (WIP)

        # if there are more agents than the actors then assign other agents with a 90 deg viewpoint than the initial
        # ones

        # if the actors are more than the agents, we only care till the time all agents have something to do.

        pass

    def get_actors_mapf_paths(self):

        # We could either read the paths from a drawing on the image

        # or come up with a random path for certain timestep

        # Should we return the paths or store it in a property? WIP
        pass

    def run_tracking(self):
        # Get the formation filming locations for all the actors.
        # Assign agents to these filming locations as per the nearest locations
        self.assign_agents_actors_viewpoints()
        # Get a conflict resolved paths for these agents to reach the formation location
        agents_paths_xyo = self.get_cbs_mapf()

        # modify the paths such that all agents contain paths of same length
        max_length_of_paths = max([len(path)] for path in agents_paths_xyo)[0]
        for agent_id in range(self.num_agents):
            agent_path_length = len(agents_paths_xyo[agent_id])
            if agent_path_length == max_length_of_paths:
                continue
            else:
                for ts in range(agent_path_length, max_length_of_paths):
                    new_location = agents_paths_xyo[agent_id][ts - 1]
                    agents_paths_xyo[agent_id].append(new_location)

        # Update new location of all agents and actors from ts 0 to completion
        # **********************************
        bSavePlots = True  # Choose to save plots and create a video from it
        directory = f"data/actors{self.num_actors}_agents{self.num_agents}_obs_den{self.PROB[1]}"
        # **********************************

        for ts in range(max_length_of_paths):
            # Update the paths of the agents for corresponding timestep
            moved_res = 0
            for agent_id in range(self.num_agents):
                new_position = agents_paths_xyo[agent_id][ts]
                if ts > 0:
                    moved_res = self.world.move_agent(new_position=new_position,
                                                      agent_id=agent_id)
                else:
                    continue

                if moved_res < 0:
                    self.logger.error(f"Cannot move agent{agent_id} to {new_position}, "
                                      f"Response {moved_res}; Timestep = {ts}")
                    # raise Exception(f"Cannot move agent{agent_id} to new location({new_position}, Response move = {
                    # moved_res}")
            self.render_current_positions(timestep=ts, save_plots=bSavePlots, output_dir=directory)
        # Render agent positions for each of the actor timestep
        images_relative_filepaths = [
            f'{directory}/plot_actors{self.num_actors}_agents{self.num_agents}_ts{timestep}.png'
            for timestep in range(max_length_of_paths)]
        if bSavePlots:
            self.create_video(images_relative_filepaths,
                              output_path=f'{directory}/actors{self.num_actors}_agents{self.num_agents}.mp4',
                              fps=2)

    @staticmethod
    def plot_star(x, y, color):
        vertices = np.array(
            [(0.0, 0.5), (0.1, 0.18), (0.5, 0.18), (0.2, -0.1), (0.3, -0.5), (0.0, -0.2), (-0.3, -0.5), (-0.2, -0.1),
             (-0.5, 0.18), (-0.1, 0.18), (0.0, 0.5)])
        codes = [Path.MOVETO] + [Path.LINETO] * 9 + [Path.CLOSEPOLY]
        vertices *= 0.2  # scale the size of the star
        vertices += (x, y)  # shift the position of the star
        path = Path(vertices, codes)
        patch = PathPatch(path, facecolor=color, edgecolor=color)
        plt.gca().add_patch(patch)

    @staticmethod
    def plot_drone_icon(ax, x, y, color):
        # Define the coordinates of the drone icon
        drone_coords = [(0, 0), (2, 2), (2, 4), (1, 5), (1, 8), (0, 9), (-1, 8), (-1, 5), (-2, 4), (-2, 2)]

        # Scale the coordinates and shift them to the given (x,y) location
        scaled_coords = [(x + c[0], y + c[1]) for c in drone_coords]

        # Create a polygon object with the scaled coordinates and fill it with the given color
        drone_icon = Polygon(scaled_coords, closed=True, facecolor=color, edgecolor=color)
        ax.add_patch(drone_icon)

    def render_current_positions(self, timestep: int, save_plots: bool = False, output_dir="data"):
        self.logger.info("Rending current agent and actor positions...")
        random.seed(self.num_actors)
        fig, ax = plt.subplots()
        fig.suptitle(f"Timestep: {timestep}; Agents: {self.num_agents}; Actors: {self.num_actors}")
        ax.imshow(self.operational_map, origin='lower', cmap='gray')

        current_agents_position = self.get_agent_positions()
        current_actors_position = self.get_actors_position()

        agents_colors = [f'#{random.randint(0, 0xFFFFFF):06x}' for _ in range(self.num_agents)]

        for agent_id in range(self.num_agents):
            x, y, orientation = current_agents_position[agent_id]
            viewing_angle = self.world.agents_state.drones[agent_id].viewing_angle
            viewing_range = self.world.agents_state.drones[agent_id].viewing_range

            ax.add_artist(plt.Circle((x, y), radius=0.3, color=agents_colors[agent_id]))
            # self.plot_drone_icon(ax, x, y, color=agents_colors[agent_id])

            wedge = Wedge((x, y), viewing_range, orientation - viewing_angle / 2, orientation + viewing_angle / 2,
                          alpha=0.5)
            ax.add_artist(wedge)

            centerline = plt.Line2D((x, x + viewing_range * np.cos(np.radians(orientation))),
                                    (y, y + viewing_range * np.sin(np.radians(orientation))), linewidth=0.1,
                                    color='k')
            ax.add_artist(centerline)

        actors_colors = [f'#{random.randint(0, 0xFFFFFF):06x}' for _ in range(self.num_actors)]
        for actor_id in range(self.num_actors):
            x, y, orientation = current_actors_position[actor_id]

            self.plot_star(x, y, color=actors_colors[actor_id])

            # ax.add_artist(plt.Circle((x, y), radius=0.3, color=actors_colors[actor_id]))

        if save_plots:
            fig.set_dpi(300)

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            plt.savefig(f'{output_dir}/plot_actors{self.num_actors}_agents{self.num_agents}_ts{timestep}.png',
                        dpi=fig.dpi)

        plt.show()

    def create_video(self, image_paths, output_path, fps=30):
        # Get the size of the first image
        img = cv2.imread(image_paths[0])
        height, width, _ = img.shape

        # Create the video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'avc1', 'MJPG', etc. depending on the codec
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Write each image to the video
        for path in image_paths:
            img = cv2.imread(path)
            out.write(img)

        # Release the video writer and print output path
        out.release()
        self.logger.info(f"Video saved to {os.path.abspath(output_path)}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    n_agents = 5
    n_actors = 5

    SEED = 123123

    env = MAPFFilming(num_agents=n_agents,
                      num_actors=n_actors,
                      world0=None,
                      grid_size=1.0,
                      size=(25, 25),
                      obstacle_prob_range=(0.01, 0.011),
                      SEED=SEED,
                      )

    env.run_tracking()

    # print(coordinationRatio(env))
