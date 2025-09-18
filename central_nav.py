import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

import uuid

from poly_point_isect import isect_segments_include_segments

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

class Intersection:
    def __init__(self, position: tuple[float, float], diameter: float, occupant: int | None = None):
        self.position = position
        self.radius_sq = (diameter / 2)**2
        self.occupant = occupant

    @property
    def radius(self) -> float:
        return (self.radius_sq)**0.5
    
    @radius.setter
    def radius(self, value: float):
        self.radius_sq = value**2
    
    @property
    def diameter(self) -> float:
        return self.radius * 2
    
    @diameter.setter
    def diameter(self, value: float):
        self.radius = value / 2

    @property
    def occupied(self) -> bool:
        return self.occupant is not None
    
    def can_enter(self, robot: int) -> bool:
        return self.occupant is None or self.occupant == robot # either unoccupied, or occupied by us

    def enter(self, robot: int) -> bool: # returns whether robot can proceed
        if self.can_enter(robot):
            self.occupant = robot
            return True
        else:
            return False
    
    def leave(self, robot: int):
        if self.occupant == robot: # we're originally occupying the intersection
            self.occupant = None
    
    def distance_sq(self, position: tuple[float, float]) -> float:
        x, y = position
        x0, y0 = self.position
        return (x-x0)**2 + (y-y0)**2

    def distance(self, position: tuple[float, float]) -> float:
        return self.distance_sq(position)**0.5

    def __repr__(self) -> str:
        return f'Intersection({self.position}, radius: {self.radius}, occupant: {self.occupant})'
    
    def draw(self, ax: plt.Axes):
        x, y = self.position
        patch = patches.Circle((x, y), self.radius, facecolor='red' if self.occupied else 'green') 
        ax.add_patch(patch) 

class CentralNav:
    def __init__(
            self,
            min_path_dist: float = 3,
            ix_radius: float = 3.5,
            min_ix_dist: float | None = None,
            min_ix_samples: int = 2,
            skew_angle: float = np.pi / 6, # to eliminate vertical lines which can break Bentley-Ottmann
            max_trans: float = 0.4, # to eliminate collinear lines
            min_seg_length: float | None = 1,
    ):
        self.min_path_dist = min_path_dist
        self.ix_radius = ix_radius; self.ix_diameter = self.ix_radius * 2
        self.min_ix_dist = ix_radius if min_ix_dist is None else min_ix_dist
        self.min_ix_samples = min_ix_samples
        self.min_seg_length_sq = min_seg_length ** 2 if min_seg_length is not None else None # if not None then we'll do path length filtering
        self.rotate_mat = np.array([
            [ np.cos(skew_angle), -np.sin(skew_angle) ],
            [ np.sin(skew_angle),  np.cos(skew_angle) ]
        ])
        self.unrotate_mat = np.array([
            [ np.cos(-skew_angle), -np.sin(-skew_angle) ],
            [ np.sin(-skew_angle),  np.cos(-skew_angle) ]
        ])
        self.max_trans = max_trans

        self.paths: dict[int, list[tuple[tuple[float, float], tuple[float, float]]]] = dict()
        
        self.ix_points: dict[str, tuple[float, float]] = dict()
        self.ix_objects: dict[str, Intersection] = dict()

        self.last_cmd: dict[str, bool] = dict() # this is pretty much for logging only

    def rotate(self, x: float, y: float) -> tuple[float, float]:
        return tuple((np.array([x, y])).tolist())

    def set_path(self, idx: int, path: np.ndarray):
        if self.min_seg_length_sq is not None: # filter path
            filtered_path = []

            for i, point in enumerate(path):
                if len(filtered_path) > 0: # always add first waypoint in
                    if np.sum((point - filtered_path[-1])**2) < self.min_seg_length_sq: # shorter than min distance
                        if i == len(path) - 1: # last waypoint gets priority and will be added instead
                            filtered_path[-1] = point
                        continue
                filtered_path.append(point)

            path = np.array(filtered_path)
        
        if len(path) > 0:
            trans = (np.random.rand(2) * 2 - 1) * self.max_trans
            path += trans

            path = path @ self.rotate_mat # apply rotation
        
        path = path.tolist()
        points = [
            (p[0], p[1])
            for p in path
        ]
        self.paths[idx] = [
            (points[i], points[i + 1])
            for i in range(len(points) - 1)
        ]
        print(f'Robot {idx} published path with {len(self.paths[idx])} segments')
        # print(self.paths[idx])
        self.find_intersections()
    
    def find_intersections(self):
        segments = []
        robot_seg_idx: dict[int, tuple[int, int]] = dict() # robot: (start, count)
        for robot_name in self.paths:
            path = self.paths[robot_name]
            robot_seg_idx[robot_name] = (len(segments), len(path))
            segments.extend(path) # add segments

        def robot_from_seg_idx(idx):
            for robot_name in robot_seg_idx:
                start, count = robot_seg_idx[robot_name]
                if idx >= start and idx < start + count:
                    return robot_name
            
            print(f'cannot get robot corresponding to segment index {idx}')
            return None

        # save collision points
        intersections: list[tuple[float, float]] = []
        for (point, segment_idxs) in isect_segments_include_segments(segments, threshold=self.min_path_dist):
            colliding_robots = set([robot_from_seg_idx(i) for i in segment_idxs])
            if len(colliding_robots) > 1: # count number of robots in collision zone
                # self.get_logger().info(f'intersection at {point}: {colliding_robots}')
                intersections.append(point)

        intersections = self.cluster_intersections(intersections) # filter intersections
        self.replace_intersections(intersections)
    
    def cluster_intersections(self, intersections: list[tuple[float, float]]) -> list[tuple[float, float]]: # DBSCAN clustering
        if len(intersections) < 2: return intersections # no intersections to cluster

        algo = DBSCAN(eps=self.min_ix_dist, min_samples=self.min_ix_samples)
        labels = algo.fit_predict(intersections).tolist()

        output: list[tuple[float, float]] = []

        # group intersections by labels
        unique_labels = set(labels)
        for label in unique_labels:
            label_ixs = [
                intersections[i]
                for i in range(len(intersections))
                if labels[i] == label
            ] # intersections with this label
            # self.get_logger().info(f'label {label}: {label_ixs}')
            if label == -1: # noise - we'll still include it anyway
                output.extend(label_ixs)
            else: # not noise - take mean of their positions
                mean_pos = np.mean(label_ixs, axis=0).tolist()
                # self.get_logger().info(f'mean position: {mean_pos} (positions: {[ix.position for ix in label_ixs]})')
                output.append(tuple(mean_pos))
        
        # unrotate
        output = (np.array(output) @ self.unrotate_mat).tolist()
        output = [tuple(p) for p in output] # convert back to list of tuple

        return output

    def replace_intersections(self, intersections: list[tuple[float, float]]):
        if len(intersections) == 0: # no intersections
            self.ix_points = dict()
            return
        
        if len(self.ix_points) == 0: # no intersections yet
            for i, point in enumerate(intersections):
                self.ix_points[str(uuid.uuid4())] = point
            return
        
        old_ix_keys, old_ix = zip(*self.ix_points.items())
        self.ix_points = dict()
        nearest_dist, nearest_idx = NearestNeighbors(n_neighbors=1).fit(old_ix).kneighbors(intersections)
        nearest_idx = nearest_idx.astype(np.int64).flatten().tolist()
        for i, dist in enumerate(nearest_dist.flatten().tolist()):
            point = intersections[i]
            if dist <= self.ix_diameter: # if intersection is close enough to existing intersection
                self.ix_points[old_ix_keys[nearest_idx[i]]] = point
            else: # create new intersection
                self.ix_points[str(uuid.uuid4())] = point

        self.ix_objects = dict()
        for key, point in self.ix_points.items():
            ix_object = Intersection(
                position=(point[0], point[1]),
                diameter=self.ix_diameter
            )
            self.ix_objects[key] = ix_object

    def draw_intersections(self, ax: plt.Axes):
        # print('Number of intersections:', len(self.ix_objects))
        for ix in self.ix_objects.values():
            ix.draw(ax)

    def set_pose(self, robot: int, x: float, y: float) -> bool: # return whether the robot can move
        if robot not in self.last_cmd:
            self.last_cmd[robot] = True # moving by default
        
        position = (x, y) # robot position

        # check through all intersections
        stop_ixes: list[str] = []
        for ix_id, ix in self.ix_objects.items():
            if ix.distance_sq(position) < ix.radius_sq: # entering
                # self.get_logger().info(f'robot {robot_name} is in intersection {ix_id}')
                if not ix.can_enter(robot):
                    stop_ixes.append(ix_id)
                else:
                    ix.enter(robot)
            else: # leaving
                ix.leave(robot)
        
        move = len(stop_ixes) == 0
        if self.last_cmd[robot] != move:
            print(f'commanding robot {robot} to ' + ('move' if move else f'STOP (against intersection(s) {stop_ixes})')) # avoid polluting logs
            # if self.telemetry:
            #     self.telemetry_pub.publish(String(data=f'{self.get_clock().now().nanoseconds}:central_nav:{robot_name},{move}')) # telemetry format: (nanosec):central_nav:(robot name),(True if commanded to move, else False)

        self.last_cmd[robot] = move
        return move