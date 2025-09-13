import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import heapq
import random
import time
import json 

MAP_SIZE = 50
ROBOT_DIAMETER = 3
CORRIDOR_WIDTH_CELLS = 3
SPACE_BETWEEN_PILLARS = 7
MAX_FRAMES = 2000 

DWA_DT = 0.1
DWA_PREDICT_TIME = 1.5
DWA_MAX_SPEED = 3.0
DWA_MAX_YAWRATE = 2.5
DWA_MAX_ACCEL = 5.0
DWA_MAX_ROT_ACCEL = 8.0
DWA_V_RESOLUTION = 0.1
DWA_YAW_RESOLUTION = 0.1
DWA_HEADING_WEIGHT = 0.15
DWA_CLEARANCE_WEIGHT = 0.25
DWA_VELOCITY_WEIGHT = 0.2

# --------------------------------------------------------------------------
# 0. METRICS TRACKING
# --------------------------------------------------------------------------
class MetricsManager:
    """Manages the collection and reporting of simulation metrics."""
    def __init__(self, num_robots, world_map):
        self.num_robots = num_robots
        self.robot_data = {i: [] for i in range(num_robots)}
        self.current_run = {i: {} for i in range(num_robots)}
        self.min_inter_robot_distance = float('inf')
        self.deadlock_count = 0
        self.in_deadlock_state = False
        
        free_space = np.sum(world_map == 0)
        self.robot_density = num_robots / free_space if free_space > 0 else 0

    def start_tracking(self, robot_id, initial_path, start_frame):
        if not self.current_run[robot_id]:
            path_length = np.sum(np.linalg.norm(np.diff(initial_path, axis=0), axis=1)) if initial_path.size > 0 else 0
            self.current_run[robot_id] = {
                'start_frame': start_frame,
                'replans': 0,
                'distance_traveled': 0.0,
                'initial_path_length': path_length,
                'collided': False,
                'status': 'In Progress',
                'frame_speeds': [],
                'velocities': []
            }

    def record_replan(self, robot_id):
        if self.current_run[robot_id]:
            self.current_run[robot_id]['replans'] += 1

    def record_collision(self, robot_id):
        if self.current_run[robot_id]:
            self.current_run[robot_id]['collided'] = True

    def record_frame_data(self, robot_id, dist_moved_this_frame, linear_v, angular_v):
        if self.current_run[robot_id]:
            self.current_run[robot_id]['distance_traveled'] += dist_moved_this_frame
            self.current_run[robot_id]['frame_speeds'].append(dist_moved_this_frame)
            self.current_run[robot_id]['velocities'].append((linear_v, angular_v))

    def goal_reached(self, robot_id, end_frame):
        if self.current_run[robot_id] and self.current_run[robot_id]['status'] == 'In Progress':
            run_data = self.current_run[robot_id]
            run_data['end_frame'] = end_frame
            run_data['nav_time_frames'] = run_data['end_frame'] - run_data['start_frame']
            run_data['status'] = 'Success'
            self.robot_data[robot_id].append(run_data)
            self.current_run[robot_id] = {}
    
    def record_min_distance(self, dist):
        self.min_inter_robot_distance = min(self.min_inter_robot_distance, dist)

    def check_and_record_deadlock(self, are_robots_stuck_list):
        is_system_deadlocked = all(are_robots_stuck_list)
        
        if is_system_deadlocked and not self.in_deadlock_state:
            self.deadlock_count += 1
            self.in_deadlock_state = True
            print("!!! Deadlock Detected !!!")
        elif not is_system_deadlocked:
            self.in_deadlock_state = False

    def simulation_end(self, final_frame):
        for i in range(self.num_robots):
            if self.current_run[i]:
                run_data = self.current_run[i]
                run_data['status'] = 'Failed'
                run_data['nav_time_frames'] = final_frame - run_data['start_frame']
                self.robot_data[i].append(run_data)
        
    def get_summary_dict(self):
        """Calculates and returns all summary metrics in a dictionary."""
        total_goals = 0
        total_successes = 0
        total_collisions = 0
        total_replans = 0
        total_avg_speed = 0.0
        successful_completion_times = []
        max_completion_frame = 0

        for i in range(self.num_robots):
            data = self.robot_data[i]
            if not data: continue
            
            total_goals += len(data)
            for run in data:
                total_replans += run['replans']
                if run['collided']:
                    total_collisions +=1
                if run['status'] == 'Success':
                    total_successes += 1
                    nav_time = run.get('nav_time_frames', 0)
                    avg_speed = run['distance_traveled'] / nav_time if nav_time > 0 else 0.0
                    total_avg_speed += avg_speed
                    successful_completion_times.append(nav_time)
                    if run.get('end_frame', 0) > max_completion_frame:
                        max_completion_frame = run.get('end_frame', 0)

        success_rate = (total_successes / total_goals) * 100 if total_goals > 0 else 0
        overall_avg_speed = total_avg_speed / total_successes if total_successes > 0 else 0.0
        makespan = max_completion_frame
        sum_of_completion_times = sum(successful_completion_times)
        min_dist_val = self.min_inter_robot_distance if self.min_inter_robot_distance != float('inf') else -1

        summary = {
            'num_robots': self.num_robots,
            'robot_density': self.robot_density,
            'success_rate': success_rate,
            'avg_speed_successful': overall_avg_speed,
            'makespan_frames': makespan,
            'sum_completion_times_frames': sum_of_completion_times,
            'min_inter_robot_dist': min_dist_val,
            'total_collisions': total_collisions,
            'total_replans': total_replans,
            'deadlock_count': self.deadlock_count,
        }
        return summary

# --------------------------------------------------------------------------
# --- CLASSES --------------------------------------------------------------
# --------------------------------------------------------------------------
class World: 
     def __init__(self, size, pillar_width, space_between_pillars): 
         self.size = size 
         self.world_map = self._create_world_map(size, pillar_width, space_between_pillars) 
     def _create_world_map(self, size, pillar_width, gap_size): 
         world = np.zeros((size, size), dtype=np.int8) 
         world[0, :] = 1; world[-1, :] = 1; world[:, 0] = 1; world[:, -1] = 1 
         num_pillars = 4 
         total_layout_size = (num_pillars * pillar_width) + ((num_pillars + 1) * gap_size) 
         if total_layout_size > size: 
             print(f"⚠️ Warning: 4x4 grid with specified pillar/gap size is too large for the map. No obstacles drawn.") 
             return world 
         centering_offset = (size - total_layout_size) // 2 
         first_pillar_pos = centering_offset + gap_size 
         step_distance = pillar_width + gap_size 
         for r_idx in range(num_pillars): 
             r = first_pillar_pos + r_idx * step_distance 
             for c_idx in range(num_pillars): 
                 c = first_pillar_pos + c_idx * step_distance 
                 if 0 <= r < size and 0 <= c < size: 
                     world[r:r+pillar_width, c:c+pillar_width] = 1 
         return world 
     def draw(self, ax): 
         ax.imshow(self.world_map, cmap='binary', origin='lower') 

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
                 if not (0 <= map_x < world_map.shape[1] and 0 <= map_y < world_map.shape[0]) or world_map[map_y, map_x] == 1: 
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
                 if hit: break 
             if not hit: 
                 x_end = robot_pose['x'] + self.max_range * np.cos(angle) 
                 y_end = robot_pose['y'] + self.max_range * np.sin(angle) 
                 scan_endpoints.append((x_end, y_end)) 
         return np.array(scan_endpoints) 

class Robot: 
     def __init__(self, x, y, theta, diameter): 
         self.pose = {'x': x, 'y': y, 'theta': theta} 
         self.velocity = {'lin': 0.0, 'ang': 0.0} 
         self.diameter = diameter 
     def move(self, linear_velocity, angular_velocity, dt): 
         dist_moved = linear_velocity * dt 
         self.velocity['lin'] = linear_velocity 
         self.velocity['ang'] = angular_velocity 
         self.pose['x'] += np.cos(self.pose['theta']) * dist_moved 
         self.pose['y'] += np.sin(self.pose['theta']) * dist_moved 
         self.pose['theta'] += self.velocity['ang'] * dt 
         self.pose['theta'] = (self.pose['theta'] + np.pi) % (2 * np.pi) - np.pi 
         return dist_moved 
     def get_pose(self): 
         return self.pose 
     def get_velocity(self): 
         return self.velocity 
     def draw(self, ax, color): 
         robot_patch = patches.Circle((self.pose['x'], self.pose['y']), self.diameter / 2, facecolor=color) 
         ax.add_patch(robot_patch) 
         ax.plot([self.pose['x'], self.pose['x'] + self.diameter * np.cos(self.pose['theta'])], 
                       [self.pose['y'], self.pose['y'] + self.diameter * np.sin(self.pose['theta'])], 'w-') 

class GoalManager: 
     def __init__(self, num_robots, world_map, robot_diameter): 
         self.num_robots = num_robots 
         self.world_map = world_map
         self.robot_diameter = robot_diameter
         try:
            self.start_poses = self._generate_random_poses(num_robots) 
            self.goals = self._generate_random_poses(num_robots) 
         except ValueError as e:
            print(f"Error initializing GoalManager: {e}")
            raise 
     def _generate_random_poses(self, num_poses): 
         poses = [] 
         safe_map = Planner._inflate_map(self.world_map, self.robot_diameter) 
         available_cells = np.argwhere(safe_map == 0).tolist() 
         if len(available_cells) < num_poses:
             raise ValueError(f"Cannot find enough available cells ({len(available_cells)}) for {num_poses} robots.")
         random.shuffle(available_cells) 
         min_dist_sq = (2 * self.robot_diameter) ** 2 
         while len(poses) < num_poses and available_cells: 
             selected_cell = available_cells.pop(0) 
             pose = {'x': selected_cell[1] + 0.5, 'y': selected_cell[0] + 0.5, 'theta': random.uniform(0, 2 * np.pi)} 
             is_valid = True 
             for p in poses: 
                 if ((p['x']-pose['x'])**2 + (p['y']-pose['y'])**2) < min_dist_sq: 
                     is_valid = False 
                     break 
             if is_valid: poses.append(pose) 
         if len(poses) < num_poses: raise ValueError("Could not find enough valid, non-overlapping spawn locations.") 
         return poses 
     def get_current_goal(self, robot_index): 
         if robot_index < len(self.goals): return self.goals[robot_index] 
         return None 
     def update_goal_status(self, robot_index, robot_pose, metrics_manager, current_frame): 
         goal_pose = self.get_current_goal(robot_index) 
         if goal_pose is not None: 
             dist_to_goal = np.hypot(robot_pose['x'] - goal_pose['x'], robot_pose['y'] - goal_pose['y']) 
             if dist_to_goal < 2.0:  
                 metrics_manager.goal_reached(robot_index, current_frame) 
                 self.goals[robot_index] = None 
     def all_goals_reached(self): 
         return all(goal is None for goal in self.goals) 

class Planner: 
     def __init__(self): pass 
     def _heuristic(self, a, b): return np.linalg.norm(np.array(a) - np.array(b)) 
     def _get_neighbors(self, node, world_map): 
         neighbors = [] 
         for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]: 
             neighbor = (node[0] + dx, node[1] + dy) 
             if 0 <= neighbor[0] < world_map.shape[1] and 0 <= neighbor[1] < world_map.shape[0] and world_map[neighbor[1], neighbor[0]] == 0: 
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
             min_r, max_r = max(0, r - inflation_radius), min(world_map.shape[0], r + inflation_radius + 1) 
             min_c, max_c = max(0, c - inflation_radius), min(world_map.shape[1], c + inflation_radius + 1) 
             inflated_map[min_r:max_r, min_c:max_c] = 1 
         return inflated_map 
     def plan_path(self, robot_pose, goal_pose, world_map, robot_diameter, other_robots, is_replan=False): 
         if goal_pose is None: return np.array([]) 
         inflated_map = self._inflate_map(world_map, robot_diameter) 
         if is_replan: 
             for other in other_robots: 
                 other_pose, radius = other.get_pose(), int(np.ceil(other.diameter)) 
                 min_r, max_r = max(0, int(other_pose['y']) - radius), min(inflated_map.shape[0], int(other_pose['y']) + radius + 1) 
                 min_c, max_c = max(0, int(other_pose['x']) - radius), min(inflated_map.shape[1], int(other_pose['x']) + radius + 1) 
                 inflated_map[min_r:max_r, min_c:max_c] = 1 
         start_node, goal_node = (int(robot_pose['x']), int(robot_pose['y'])), (int(goal_pose['x']), int(goal_pose['y'])) 
         if not (0 <= goal_node[1] < inflated_map.shape[0] and 0 <= goal_node[0] < inflated_map.shape[1]) or inflated_map[start_node[1], start_node[0]] == 1 or inflated_map[goal_node[1], goal_node[0]] == 1: 
             return np.array([]) 
         open_set, came_from = [], {} 
         heapq.heappush(open_set, (0, start_node)) 
         g_score = {node: float('inf') for node in np.ndindex(inflated_map.shape)} 
         g_score[start_node] = 0 
         f_score = {node: float('inf') for node in np.ndindex(inflated_map.shape)} 
         f_score[start_node] = self._heuristic(start_node, goal_node) 
         open_set_hash = {start_node} 
         while open_set: 
             current = heapq.heappop(open_set)[1] 
             open_set_hash.remove(current) 
             if current == goal_node: return self._reconstruct_path(came_from, current) 
             for neighbor in self._get_neighbors(current, inflated_map): 
                 tentative_g_score = g_score[current] + self._heuristic(current, neighbor) 
                 if tentative_g_score < g_score.get(neighbor, float('inf')): 
                     came_from[neighbor], g_score[neighbor] = current, tentative_g_score 
                     f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, goal_node) 
                     if neighbor not in open_set_hash: 
                         heapq.heappush(open_set, (f_score[neighbor], neighbor)) 
                         open_set_hash.add(neighbor) 
         return np.array([]) 

class DWAController: 
     def __init__(self): 
         self.waypoint_index = 0 
     def reset(self): 
         self.waypoint_index = 0 
     def compute_velocities(self, robot_pose, robot_velocity, path, scan_data): 
         if not isinstance(path, np.ndarray) or path.size == 0 or self.waypoint_index >= len(path): 
             return 0.0, 0.0 
         robot_pos = np.array([robot_pose['x'], robot_pose['y']]) 
         while self.waypoint_index < len(path) - 1 and np.linalg.norm(path[self.waypoint_index] - robot_pos) < 3.0: 
             self.waypoint_index += 1 
         target_waypoint = path[self.waypoint_index] 
         angle_to_target = np.arctan2(target_waypoint[1] - robot_pos[1], target_waypoint[0] - robot_pos[0]) 
         initial_angle_error = (angle_to_target - robot_pose['theta'] + np.pi) % (2 * np.pi) - np.pi 
         dynamic_window = self._generate_dynamic_window(robot_velocity, initial_angle_error) 
         best_score = -float('inf') 
         best_velocity_cmd = (0.0, 0.0) 
         for v in np.arange(dynamic_window[0], dynamic_window[1] + DWA_V_RESOLUTION, DWA_V_RESOLUTION): 
             for y in np.arange(dynamic_window[2], dynamic_window[3] + DWA_YAW_RESOLUTION, DWA_YAW_RESOLUTION): 
                 trajectory = self._simulate_trajectory(robot_pose, v, y) 
                 if not trajectory: continue 
                 heading_score = DWA_HEADING_WEIGHT * self._score_heading(trajectory, target_waypoint) 
                 clearance_score = DWA_CLEARANCE_WEIGHT * self._score_clearance(trajectory, scan_data) 
                 velocity_score = DWA_VELOCITY_WEIGHT * self._score_velocity(v) 
                 if clearance_score <= 0.0: 
                     total_score = -float('inf') 
                 else: 
                     total_score = heading_score + clearance_score + velocity_score 
                 if total_score > best_score: 
                     best_score = total_score 
                     best_velocity_cmd = (v, y) 
         return best_velocity_cmd 
     def _generate_dynamic_window(self, robot_velocity, angle_error): 
         vs = [0.0, DWA_MAX_SPEED, -DWA_MAX_YAWRATE, DWA_MAX_YAWRATE] 
         vd = [robot_velocity['lin'] - DWA_MAX_ACCEL * DWA_DT, 
               robot_velocity['lin'] + DWA_MAX_ACCEL * DWA_DT, 
               robot_velocity['ang'] - DWA_MAX_ROT_ACCEL * DWA_DT, 
               robot_velocity['ang'] + DWA_MAX_ROT_ACCEL * DWA_DT] 
         dw = [max(vs[0], vd[0]), min(vs[1], vd[1]), 
               max(vs[2], vd[2]), min(vs[3], vd[3])] 
         alignment_tolerance = np.deg2rad(30)
         if abs(angle_error) > alignment_tolerance: 
             dw[0] = 0.0
             dw[1] = 0.0
         return dw 
     def _simulate_trajectory(self, robot_pose, v, y): 
         pose = dict(robot_pose) 
         trajectory = [] 
         time = 0 
         while time <= DWA_PREDICT_TIME: 
             time += DWA_DT 
             pose['x'] += v * np.cos(pose['theta']) * DWA_DT 
             pose['y'] += v * np.sin(pose['theta']) * DWA_DT 
             pose['theta'] += y * DWA_DT 
             trajectory.append(np.array([pose['x'], pose['y'], pose['theta']])) 
         return trajectory 
     def _score_heading(self, trajectory, target): 
         final_pose = trajectory[-1] 
         angle_to_target = np.arctan2(target[1] - final_pose[1], target[0] - final_pose[0]) 
         angle_error = abs((angle_to_target - final_pose[2] + np.pi) % (2 * np.pi) - np.pi) 
         return np.pi - angle_error 
     def _score_clearance(self, trajectory, scan_data): 
         if scan_data.size == 0: return DWA_MAX_SPEED 
         min_dist_on_traj = float('inf') 
         for point in trajectory: 
             distances = np.linalg.norm(scan_data - point[:2], axis=1) 
             min_dist_on_traj = min(min_dist_on_traj, np.min(distances)) 
         if min_dist_on_traj < ROBOT_DIAMETER / 2.0: 
             return 0.0 
         return min(min_dist_on_traj, DWA_MAX_SPEED) 
     def _score_velocity(self, v): 
         return v 

# --------------------------------------------------------------------------
# 7. MAIN SIMULATION FUNCTION
# --------------------------------------------------------------------------
def run_simulation(num_robots, save_video=False):

    world = World(MAP_SIZE, CORRIDOR_WIDTH_CELLS, SPACE_BETWEEN_PILLARS)
    metrics_manager = MetricsManager(num_robots, world.world_map)
    try:
        goal_manager = GoalManager(num_robots, world.world_map, ROBOT_DIAMETER)
    except ValueError as e:
        print(f"Halting simulation for {num_robots} robots due to setup error: {e}")
        return {"error": str(e), "num_robots": num_robots}


    robots, planners, controllers = [], [], []
    start_poses = goal_manager.start_poses
    for i in range(num_robots):
        start_pose = start_poses[i]
        robots.append(Robot(start_pose['x'], start_pose['y'], start_pose['theta'], ROBOT_DIAMETER))
        planners.append(Planner())
        controllers.append(DWAController())

    lidar = Lidar()
    current_paths = {i: np.array([]) for i in range(num_robots)}
    last_frame = 0

    def simulation_step(frame):
        nonlocal last_frame
        last_frame = frame

        all_poses = [r.get_pose() for r in robots]
        are_robots_stuck_this_frame = [False] * num_robots

        for i in range(num_robots):
            for j in range(i + 1, num_robots):
                dist = np.hypot(all_poses[i]['x'] - all_poses[j]['x'], all_poses[i]['y'] - all_poses[j]['y'])
                metrics_manager.record_min_distance(dist)
                if dist < ROBOT_DIAMETER:
                    metrics_manager.record_collision(i)
                    metrics_manager.record_collision(j)
                    robots[i].velocity = {'lin': 0, 'ang': 0}
                    robots[j].velocity = {'lin': 0, 'ang': 0}

        for i in range(num_robots):
            robot, planner, controller = robots[i], planners[i], controllers[i]
            current_pose, current_vel = robot.get_pose(), robot.get_velocity()
            current_goal = goal_manager.get_current_goal(i)
            other_robots = [robots[j] for j in range(num_robots) if i != j]

            goal_manager.update_goal_status(i, current_pose, metrics_manager, frame)
            
            path_to_follow = current_paths.get(i, np.array([]))
            if current_goal is not None and path_to_follow.size == 0:
                path = planner.plan_path(current_pose, current_goal, world.world_map, robot.diameter, other_robots, is_replan=False)
                current_paths[i] = path
                controller.reset()
                metrics_manager.start_tracking(i, path, frame)
            
            elif frame > 0 and frame % 50 == 0 and np.linalg.norm([current_vel['lin'], current_vel['ang']]) < 0.1 and path_to_follow.size > 0:
                 path = planner.plan_path(current_pose, current_goal, world.world_map, robot.diameter, other_robots, is_replan=True)
                 current_paths[i] = path
                 controller.reset()
                 metrics_manager.record_replan(i)
            
            scan_data = lidar.scan(current_pose, world.world_map, other_robots)
            linear_v, angular_v = controller.compute_velocities(current_pose, current_vel, path_to_follow, scan_data)
            dist_moved = robot.move(linear_v, angular_v, dt=DWA_DT)
            
            if current_goal is not None and abs(linear_v) < 0.01 and abs(angular_v) < 0.01:
                are_robots_stuck_this_frame[i] = True

            metrics_manager.record_frame_data(i, dist_moved, linear_v, angular_v)
        
        metrics_manager.check_and_record_deadlock(are_robots_stuck_this_frame)


    if save_video:
        fig, ax = plt.subplots(figsize=(10, 10))
        robot_colors = plt.cm.jet(np.linspace(0, 1, num_robots))

        def update_visual(frame):
            ax.clear()
            world.draw(ax)
            
            simulation_step(frame)

            for i in range(num_robots):
                robots[i].draw(ax, color=robot_colors[i])
                path_to_follow = current_paths.get(i, np.array([]))
                if path_to_follow.size > 0:
                    ax.plot(path_to_follow[:, 0], path_to_follow[:, 1], '--', color=robot_colors[i], alpha=0.7)
                current_goal = goal_manager.get_current_goal(i)
                if current_goal is not None:
                    ax.plot(current_goal['x'], current_goal['y'], '*', color=robot_colors[i], markersize=15, markeredgecolor='black')
            
            ax.set_title(f"Multi-Robot Simulation ({num_robots} robots) - Frame {frame}")
            ax.set_xlim(0, MAP_SIZE); ax.set_ylim(0, MAP_SIZE)
            ax.set_aspect('equal')
            if frame % 20 == 0: print(f"Processing video... Frame {frame}/{MAX_FRAMES}")

        def frame_generator(max_f):
            for f in range(max_f):
                if goal_manager.all_goals_reached():
                    print(f"\nAll goals reached at frame {f}. Ending simulation.")
                    break
                yield f
        
        ani = FuncAnimation(fig, update_visual, frames=frame_generator(MAX_FRAMES), interval=int(DWA_DT*1000), repeat=False)
        
        from matplotlib import animation
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='AI'), bitrate=1800)
        
        video_filename = f'simulation_{num_robots}_robots_{time.strftime("%Y%m%d-%H%M%S")}.mp4'
        print(f"Starting to save simulation to {video_filename}...")
        ani.save(video_filename, writer=writer)
        print(f"Video saved successfully to {video_filename}")

    else:
        for frame in range(MAX_FRAMES):
            if goal_manager.all_goals_reached():
                print(f"All goals reached at frame {frame}.")
                break
            simulation_step(frame)
            if frame % 250 == 0:
                print(f"  ... Simulating frame {frame}/{MAX_FRAMES}")
        print(f"Headless simulation finished at frame {last_frame}.")


    metrics_manager.simulation_end(last_frame)
    return metrics_manager.get_summary_dict()


if __name__ == '__main__':
    print("Running a standalone simulation example with 4 robots...")
    
    metrics_results = run_simulation(num_robots=4, save_video=True)
    
    print("\n" + "="*50)
    print(" S T A N D A L O N E   R U N   M E T R I C S")
    print("="*50)
    print(json.dumps(metrics_results, indent=2))
    print("="*50)
