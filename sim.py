import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import heapq
import random

# --- Core Simulation Parameters ---
NUM_ROBOTS = 4
MAP_SIZE = 50
ROBOT_DIAMETER = 3
CORRIDOR_WIDTH_CELLS = 3
OBSTACLE_SPACING_CELLS = 10

# --------------------------------------------------------------------------
# 1. MAP AND ENVIRONMENT LOGIC
# --------------------------------------------------------------------------
class World:
    def __init__(self, size, corridor_width, obstacle_spacing):
        self.size = size
        self.world_map = self._create_world_map(size, corridor_width, obstacle_spacing)

    def _create_world_map(self, size, corridor_width, obstacle_spacing):
        world = np.zeros((size, size), dtype=np.int8)
        world[0, :] = 1; world[-1, :] = 1; world[:, 0] = 1; world[:, -1] = 1
        for r in range(obstacle_spacing, size, obstacle_spacing):
            for c in range(obstacle_spacing, size, obstacle_spacing):
                if r + corridor_width < size and c + corridor_width < size:
                    world[r:r+corridor_width, c:c+corridor_width] = 1
        return world

    def draw(self, ax):
        ax.imshow(self.world_map, cmap='binary', origin='lower')

# --------------------------------------------------------------------------
# 2. SENSOR LOGIC (LIDAR)
# --------------------------------------------------------------------------
class Lidar:
    def __init__(self, num_rays=90, max_range=25.0, fov_deg=90):
        self.num_rays = num_rays
        self.max_range = max_range
        self.fov_rad = np.deg2rad(fov_deg)

    def scan(self, robot_pose, world_map, other_robots):
        scan_endpoints = []
        start_angle = robot_pose['theta'] - self.fov_rad / 2
        end_angle = robot_pose['theta'] + self.fov_rad / 2
        ray_angles = np.linspace(start_angle, end_angle, self.num_rays)

        for angle in ray_angles:
            hit = False
            for distance in np.arange(0, self.max_range, 0.1):
                x_check = robot_pose['x'] + distance * np.cos(angle)
                y_check = robot_pose['y'] + distance * np.sin(angle)
                map_x, map_y = int(x_check), int(y_check)

                if not (0 <= map_x < world_map.shape[1] and 0 <= map_y < world_map.shape[0]) or \
                   world_map[map_y, map_x] == 1:
                    scan_endpoints.append((x_check, y_check))
                    hit = True
                    break
            
                for other in other_robots:
                    other_pose = other.get_pose()
                    dist_to_other = np.hypot(x_check - other_pose['x'], y_check - other_pose['y'])
                    if dist_to_other < other.diameter / 2:
                        scan_endpoints.append((x_check, y_check))
                        hit = True
                        break
                if hit:
                    break
        
            if not hit:
                x_end = robot_pose['x'] + self.max_range * np.cos(angle)
                y_end = robot_pose['y'] + self.max_range * np.sin(angle)
                scan_endpoints.append((x_end, y_end))

        return np.array(scan_endpoints)

# --------------------------------------------------------------------------
# 3. ROBOT STATE AND ODOMETRY LOGIC
# --------------------------------------------------------------------------
class Robot:
    def __init__(self, x, y, theta, diameter):
        self.pose = {'x': x, 'y': y, 'theta': theta}
        self.diameter = diameter

    def move(self, linear_velocity, angular_velocity, dt):
        self.pose['theta'] += angular_velocity * dt
        self.pose['x'] += linear_velocity * np.cos(self.pose['theta']) * dt
        self.pose['y'] += linear_velocity * np.sin(self.pose['theta']) * dt

    def get_pose(self):
        return self.pose

    def draw(self, ax, color):
        robot_patch = patches.Circle((self.pose['x'], self.pose['y']), self.diameter / 2, facecolor=color)
        ax.add_patch(robot_patch)
        ax.plot([self.pose['x'], self.pose['x'] + self.diameter * np.cos(self.pose['theta'])],
                      [self.pose['y'], self.pose['y'] + self.diameter * np.sin(self.pose['theta'])], 'w-')

# --------------------------------------------------------------------------
# 4. GOAL ASSIGNING LOGIC
# --------------------------------------------------------------------------
class GoalManager:
    def __init__(self, num_robots, world_map, robot_diameter):
        self.num_robots = num_robots
        self.start_poses = self._generate_random_poses(world_map, robot_diameter, num_robots)
        self.goals = self._generate_random_poses(world_map, robot_diameter, num_robots)

    def _generate_random_poses(self, world_map, robot_diameter, num_poses):
        poses = []
        safe_map = Planner._inflate_map(world_map, robot_diameter)
        free_cells = np.argwhere(safe_map == 0)
    
        if len(free_cells) < num_poses:
            raise ValueError("Not enough free cells to spawn all robots. Check map and robot size.")

        selected_indices = np.random.choice(len(free_cells), num_poses, replace=False)
        for index in selected_indices:
            pos = free_cells[index]
            pose = {
                'x': pos[1] + 0.5,
                'y': pos[0] + 0.5,
                'theta': random.uniform(0, 2 * np.pi)
            }
            poses.append(pose)
        return poses

    def get_current_goal(self, robot_index):
        if robot_index < len(self.goals):
            return self.goals[robot_index]
        return None

    def update_goal_status(self, robot_index, robot_pose):
        goal_pose = self.get_current_goal(robot_index)
        if goal_pose is not None:
            goal_pos = np.array([goal_pose['x'], goal_pose['y']])
            robot_pos = np.array([robot_pose['x'], robot_pose['y']])
            dist_to_goal = np.linalg.norm(robot_pos - goal_pos)
            if dist_to_goal < 2.0:
                self.goals[robot_index] = None


# --------------------------------------------------------------------------
# 5. HIGH-LEVEL PLANNING LOGIC (A* Algorithm)
# --------------------------------------------------------------------------
class Planner:
    def __init__(self):
        pass

    def _heuristic(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def _get_neighbors(self, node, world_map):
        neighbors = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            neighbor = (node[0] + dx, node[1] + dy)
            if 0 <= neighbor[0] < world_map.shape[1] and \
               0 <= neighbor[1] < world_map.shape[0] and \
               world_map[neighbor[1], neighbor[0]] == 0:
                neighbors.append(neighbor)
        return neighbors

    def _reconstruct_path(self, came_from, current):
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        return np.array(total_path[::-1]) + 0.5

    @staticmethod
    def _inflate_map(world_map, robot_diameter):
        inflated_map = np.copy(world_map)
        inflation_radius = int(np.ceil(robot_diameter / 2.0))
        if inflation_radius == 0: return inflated_map
        obstacle_rows, obstacle_cols = np.where(world_map == 1)
        for r, c in zip(obstacle_rows, obstacle_cols):
            min_r = max(0, r - inflation_radius)
            max_r = min(world_map.shape[0], r + inflation_radius + 1)
            min_c = max(0, c - inflation_radius)
            max_c = min(world_map.shape[1], c + inflation_radius + 1)
            inflated_map[min_r:max_r, min_c:max_c] = 1
        return inflated_map

    def plan_path(self, robot_pose, goal_pose, world_map, robot_diameter, other_robots):
        if goal_pose is None:
            return np.array([])

        inflated_map = self._inflate_map(world_map, robot_diameter)

        for other in other_robots:
            other_pose = other.get_pose()
            radius = int(np.ceil(other.diameter))
            min_r = max(0, int(other_pose['y']) - radius)
            max_r = min(inflated_map.shape[0], int(other_pose['y']) + radius + 1)
            min_c = max(0, int(other_pose['x']) - radius)
            max_c = min(inflated_map.shape[1], int(other_pose['x']) + radius + 1)
            inflated_map[min_r:max_r, min_c:max_c] = 1

        start_node = (int(robot_pose['x']), int(robot_pose['y']))
        goal_node = (int(goal_pose['x']), int(goal_pose['y']))
    
        if not (0 <= goal_node[1] < inflated_map.shape[0] and 0 <= goal_node[0] < inflated_map.shape[1]) or \
           inflated_map[start_node[1], start_node[0]] == 1 or \
           inflated_map[goal_node[1], goal_node[0]] == 1:
            return np.array([])

        open_set = []
        heapq.heappush(open_set, (0, start_node))
        came_from = {}
        g_score = {node: float('inf') for node in np.ndindex(inflated_map.shape)}
        g_score[start_node] = 0
        f_score = {node: float('inf') for node in np.ndindex(inflated_map.shape)}
        f_score[start_node] = self._heuristic(start_node, goal_node)
        open_set_hash = {start_node}

        while open_set:
            current = heapq.heappop(open_set)[1]
            open_set_hash.remove(current)
            if current == goal_node:
                return self._reconstruct_path(came_from, current)
            for neighbor in self._get_neighbors(current, inflated_map):
                tentative_g_score = g_score[current] + self._heuristic(current, neighbor)
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, goal_node)
                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)
    
        return np.array([])

# --------------------------------------------------------------------------
# 6. LOW-LEVEL MOVING LOGIC (CONTROLLER)
# --------------------------------------------------------------------------
class Controller:
    def __init__(self, kp_linear=0.5, kp_angular=2.0, stuck_threshold=20):
        self.kp_linear = kp_linear
        self.kp_angular = kp_angular
        self.waypoint_index = 0
        self.stuck_counter = 0
        self.stuck_threshold = stuck_threshold
        self.last_pose = None

    def is_stuck(self):
        return self.stuck_counter >= self.stuck_threshold

    def compute_velocities(self, robot_pose, path, scan_data):
        if self.last_pose:
            dist_moved = np.hypot(robot_pose['x'] - self.last_pose['x'], robot_pose['y'] - self.last_pose['y'])
            if dist_moved < 0.05:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0
        self.last_pose = robot_pose.copy()

        if scan_data.size > 0:
            robot_pos = np.array([robot_pose['x'], robot_pose['y']])
            distances = np.linalg.norm(scan_data - robot_pos, axis=1)
            if np.min(distances) < ROBOT_DIAMETER:
                return 0.0, 0.8

        if not isinstance(path, np.ndarray) or path.size == 0 or self.waypoint_index >= len(path):
            return 0.0, 0.0

        robot_pos = np.array([robot_pose['x'], robot_pose['y']])
        while self.waypoint_index < len(path) - 1 and \
              np.linalg.norm(path[self.waypoint_index] - robot_pos) < 2.0:
            self.waypoint_index += 1

        target_waypoint = path[self.waypoint_index]
        angle_to_target = np.arctan2(target_waypoint[1] - robot_pos[1], target_waypoint[0] - robot_pos[0])
        angle_error = (angle_to_target - robot_pose['theta'] + np.pi) % (2 * np.pi) - np.pi
        angular_v = self.kp_angular * angle_error
        linear_v = self.kp_linear * np.linalg.norm(target_waypoint - robot_pos)
        linear_v = min(linear_v, 3.0)

        return linear_v, angular_v
    
    def reset(self):
        self.waypoint_index = 0
        self.stuck_counter = 0
        self.last_pose = None

# --------------------------------------------------------------------------
# 7. MAIN SIMULATION
# --------------------------------------------------------------------------
def main():
    world = World(MAP_SIZE, CORRIDOR_WIDTH_CELLS, OBSTACLE_SPACING_CELLS)
    goal_manager = GoalManager(NUM_ROBOTS, world.world_map, ROBOT_DIAMETER)

    robots = []
    planners = []
    controllers = []
    start_poses = goal_manager.start_poses
    for i in range(NUM_ROBOTS):
        start_pose = start_poses[i]
        robots.append(Robot(start_pose['x'], start_pose['y'], start_pose['theta'], ROBOT_DIAMETER))
        planners.append(Planner())
        controllers.append(Controller())

    lidar = Lidar()
    fig, ax = plt.subplots(figsize=(10, 10))
    
    current_paths = {i: np.array([]) for i in range(NUM_ROBOTS)}
    robot_colors = plt.cm.jet(np.linspace(0, 1, NUM_ROBOTS))

    def update(frame):
        ax.clear()
        world.draw(ax)
    
        all_robot_poses = [r.get_pose() for r in robots]

        for i in range(NUM_ROBOTS):
            robot = robots[i]
            planner = planners[i]
            controller = controllers[i]
            current_pose = robot.get_pose()
            current_goal = goal_manager.get_current_goal(i)
            other_robots = [robots[j] for j in range(NUM_ROBOTS) if i != j]

            goal_manager.update_goal_status(i, current_pose)
        
            if controller.is_stuck() or (current_goal is not None and current_paths[i].size == 0):
                if controller.is_stuck():
                    print(f"Robot {i} is stuck! Replanning...")
            
                path = planner.plan_path(current_pose, current_goal, world.world_map, robot.diameter, other_robots)
                current_paths[i] = path
                controller.reset()

            scan_data = lidar.scan(current_pose, world.world_map, other_robots)
            path_to_follow = current_paths.get(i, np.array([]))
            linear_v, angular_v = controller.compute_velocities(current_pose, path_to_follow, scan_data)
            robot.move(linear_v, angular_v, dt=0.1)

            robot.draw(ax, color=robot_colors[i])
        
            if path_to_follow.size > 0:
                ax.plot(path_to_follow[:, 0], path_to_follow[:, 1], '--', color=robot_colors[i])
            if current_goal is not None:
                ax.plot(current_goal['x'], current_goal['y'], '*', color=robot_colors[i], markersize=15)
    
        ax.set_title(f"Multi-Robot Simulation with Replanning (Frame {frame})")
        ax.set_xlim(0, MAP_SIZE); ax.set_ylim(0, MAP_SIZE)
        ax.set_aspect('equal')

    ani = FuncAnimation(fig, update, frames=500, interval=50, repeat=False)
    plt.show()

if __name__ == '__main__':
    main()
