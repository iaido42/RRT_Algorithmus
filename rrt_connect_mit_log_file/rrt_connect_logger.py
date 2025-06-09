import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import datetime
import logging


# --- Klassen Node und Obstacle (unverändert) ---
class Node:
    def __init__(self, x, y, parent=None):
        self.x = x;
        self.y = y;
        self.parent = parent


class Obstacle:
    def __init__(self, x, y, width, height, name=""):
        self.x = x;
        self.y = y;
        self.width = width;
        self.height = height;
        self.name = name


# --- Hauptklasse für den RRT-Connect-Algorithmus ---
class RRTConnect:
    def __init__(self, start_pos, goal_pos, obstacles, bounds, step_size=5.0, max_iter=5000,
                 log_file="rrt_connect_protocol.log"):
        self.start_node = Node(start_pos[0], start_pos[1])
        self.goal_node = Node(goal_pos[0], goal_pos[1])
        self.tree_a = [self.start_node];
        self.tree_b = [self.goal_node]
        self.obstacles = obstacles;
        self.bounds = bounds;
        self.step_size = step_size;
        self.max_iter = max_iter
        self.path = [];
        self._setup_logger(log_file)

    def _setup_logger(self, log_file):
        self.logger = logging.getLogger('RRT_Connect_Logger')
        if not self.logger.hasHandlers():
            self.logger.setLevel(logging.INFO)
            formatter = logging.Formatter('%(message)s')
            file_handler = logging.FileHandler(log_file, mode='w');
            file_handler.setFormatter(formatter)
            console_handler = logging.StreamHandler();
            console_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler);
            self.logger.addHandler(console_handler)

    def run(self):
        self.logger.info("[SYSTEM] RRT-Connect Protokoll gestartet.")
        self.logger.info(
            f"[CONFIG] Start: ({self.start_node.x:.1f}, {self.start_node.y:.1f}), Ziel: ({self.goal_node.x:.1f}, {self.goal_node.y:.1f})")
        self.logger.info("-" * 60)

        active_tree, passive_tree = self.tree_a, self.tree_b
        for i in range(self.max_iter):
            iter_str = f"[Iter. {i + 1:04d}]"
            active_tree_name = 'A' if active_tree is self.tree_a else 'B'
            rand_point = self._get_random_point()
            self.logger.info(
                f"{iter_str} EXPAND_{active_tree_name}: Erweitere Baum {active_tree_name} -> ({rand_point[0]:.1f}, {rand_point[1]:.1f})")
            new_node = self._expand_tree(active_tree, rand_point)

            if new_node:
                self.logger.info(
                    f"{iter_str} CONNECT: Versuche Verbindung von neuem Knoten aus Baum {active_tree_name} zum anderen Baum.")
                connection_result = self._connect_trees(new_node, passive_tree)
                if connection_result:
                    self.logger.info(f"    -> {iter_str} SUCCESS: Bäume nach {i + 1} Iterationen verbunden!")
                    node_from_active_tree, node_from_passive_tree = connection_result
                    if active_tree is self.tree_a:
                        self._reconstruct_path(node_from_active_tree, node_from_passive_tree)
                    else:
                        self._reconstruct_path(node_from_passive_tree, node_from_active_tree)
                    self._validate_path()
                    return self.path

            active_tree, passive_tree = passive_tree, active_tree

        self.logger.info(f"[SYSTEM] FEHLER: Kein Pfad nach {self.max_iter} Iterationen gefunden.")
        return None

    def _expand_tree(self, tree, rand_point):
        nearest_node = self._get_nearest_node(tree, rand_point)
        new_node = self._steer(nearest_node, rand_point)
        if self._is_path_free(nearest_node, new_node):
            new_node.parent = nearest_node;
            tree.append(new_node)
            return new_node
        return None

    def _connect_trees(self, new_node, target_tree):
        current_node = new_node
        while True:
            nearest_in_target = self._get_nearest_node(target_tree, (current_node.x, current_node.y))
            steered_node = self._steer(current_node, (nearest_in_target.x, nearest_in_target.y))
            if not self._is_path_free(current_node, steered_node): return None
            steered_node.parent = current_node
            if new_node in self.tree_a:
                self.tree_a.append(steered_node)
            else:
                self.tree_b.append(steered_node)
            current_node = steered_node
            dist_to_target = np.hypot(current_node.x - nearest_in_target.x, current_node.y - nearest_in_target.y)
            if dist_to_target < self.step_size:
                if self._is_path_free(current_node, nearest_in_target):
                    return current_node, nearest_in_target
                else:
                    return None
        return None

    def _reconstruct_path(self, node_from_start_tree, node_from_goal_tree):
        # KORREKTUR: Dies ist die entscheidende Aenderung.
        path_from_start = []
        node = node_from_start_tree
        while node is not None:
            path_from_start.append(node)
            node = node.parent

        path_from_goal = []
        node = node_from_goal_tree
        while node is not None:
            path_from_goal.append(node)
            node = node.parent

        # Nur der Start-Pfad muss umgedreht werden.
        path_from_start.reverse()

        # Der Ziel-Pfad ist bereits in der korrekten Reihenfolge (Verbindung -> Ziel).
        # FEHLERHAFTE ZEILE ENTFERNT: path_from_goal.reverse()

        # Kombiniere die Pfade
        self.path = path_from_start + path_from_goal

    def _validate_path(self):
        self.logger.info("[SYSTEM] Validiere den gefundenen Pfad...")
        if not self.path: self.logger.error("    -> VALIDIERUNG FEHLGESCHLAGEN: Pfad ist leer."); return
        start_ok = np.allclose([self.path[0].x, self.path[0].y], [self.start_node.x, self.start_node.y])
        end_ok = np.allclose([self.path[-1].x, self.path[-1].y], [self.goal_node.x, self.goal_node.y])
        if not start_ok: self.logger.warning(f"    -> WARNUNG: Pfad startet nicht am Startpunkt!")
        if not end_ok: self.logger.warning(f"    -> WARNUNG: Pfad endet nicht am Zielpunkt!")
        max_dist = 0;
        is_collision_free = True
        for i in range(len(self.path) - 1):
            n1 = self.path[i];
            n2 = self.path[i + 1]
            dist = np.hypot(n1.x - n2.x, n1.y - n2.y)
            if dist > max_dist: max_dist = dist
            if not self._is_path_free(n1, n2):
                self.logger.error(f"    -> VALIDIERUNG FEHLGESCHLAGEN: Kollision im Segment {i} -> {i + 1}")
                is_collision_free = False
        self.logger.info(f"    -> Maximale Distanz zwischen Segmenten: {max_dist:.2f} (Schrittweite: {self.step_size})")
        if is_collision_free and start_ok and end_ok:
            self.logger.info("    -> VALIDIERUNG ERFOLGREICH: Pfad ist verbunden und kollisionsfrei.")
        else:
            self.logger.error("    -> VALIDIERUNG FEHLGESCHLAGEN: Bitte Details oben prüfen.")

    def _get_random_point(self):
        if np.random.rand() > 0.95: return (self.goal_node.x, self.goal_node.y)
        return (np.random.uniform(0, self.bounds[0]), np.random.uniform(0, self.bounds[1]))

    def _get_nearest_node(self, nodes, point):
        distances = [np.hypot(node.x - point[0], node.y - point[1]) for node in nodes]
        return nodes[np.argmin(distances)]

    def _steer(self, from_node, to_point):
        direction = np.array([to_point[0] - from_node.x, to_point[1] - from_node.y])
        distance = np.linalg.norm(direction)
        if distance <= self.step_size: return Node(to_point[0], to_point[1])
        direction = (direction / distance) * self.step_size
        return Node(from_node.x + direction[0], from_node.y + direction[1])

    def _is_path_free(self, from_node, to_node):
        for obs in self.obstacles:
            if self._check_line_obstacle_collision(from_node, to_node, obs): return False
        return True

    def _check_line_obstacle_collision(self, n1, n2, obs):
        p1 = np.array([n1.x, n1.y]);
        p2 = np.array([n2.x, n2.y]);
        r_min = np.array([obs.x, obs.y]);
        r_max = np.array([obs.x + obs.width, obs.y + obs.height])
        direction = p2 - p1
        if np.all(np.abs(direction) < 1e-9): return r_min[0] <= p1[0] <= r_max[0] and r_min[1] <= p1[1] <= r_max[1]
        t_near, t_far = -np.inf, np.inf
        for i in range(2):
            if np.abs(direction[i]) < 1e-9:
                if p1[i] < r_min[i] or p1[i] > r_max[i]: return False
            else:
                t1 = (r_min[i] - p1[i]) / direction[i];
                t2 = (r_max[i] - p1[i]) / direction[i]
                if t1 > t2: t1, t2 = t2, t1
                t_near = max(t_near, t1);
                t_far = min(t_far, t2)
                if t_near > t_far: return False
        return t_near <= 1 and t_far >= 0

    def draw_graph(self):
        plt.figure(figsize=(12, 12));
        ax = plt.gca()
        for obs in self.obstacles: ax.add_patch(
            patches.Rectangle((obs.x, obs.y), obs.width, obs.height, facecolor='black', alpha=0.7, zorder=2))
        for tree, color in [(self.tree_a, 'c-'), (self.tree_b, 'y-')]:
            for node in tree:
                if node.parent: plt.plot([node.x, node.parent.x], [node.y, node.parent.y], color, zorder=1)
        if self.path:
            px = [node.x for node in self.path];
            py = [node.y for node in self.path]
            plt.plot(px, py, 'm-', linewidth=2.5, zorder=3)
        plt.plot(self.start_node.x, self.start_node.y, 'go', markersize=10, label='Start', zorder=4)
        plt.plot(self.goal_node.x, self.goal_node.y, 'ro', markersize=10, label='Ziel', zorder=4)
        plt.xlim(0, self.bounds[0]);
        plt.ylim(0, self.bounds[1])
        plt.gca().set_aspect('equal', adjustable='box');
        plt.title('RRT-Connect Visualisierung (Final V5)')
        plt.legend();
        plt.grid(True);
        plt.show()


# --- Hauptprogramm ---
if __name__ == '__main__':
    start_position = (10, 10);
    goal_position = (90, 90);
    bounds = (100, 100)
    obstacles_list = [Obstacle(20, 10, 20, 40, name="R1"), Obstacle(60, 50, 20, 40, name="R2")]
    rrt_connect = RRTConnect(start_pos=start_position, goal_pos=goal_position, obstacles=obstacles_list, bounds=bounds,
                             step_size=5.0, max_iter=1000)
    path = rrt_connect.run()
    if path:
        rrt_connect.draw_graph()
