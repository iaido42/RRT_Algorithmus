import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import datetime
import logging
import math


# --- Klasse für einen Knoten, jetzt mit Kosten-Attribut ---
class Node:
    def __init__(self, x, y, parent=None, cost=0.0):
        self.x = x
        self.y = y
        self.parent = parent
        self.cost = cost  # Kosten vom Start bis zu diesem Knoten


# --- Klasse für Hindernisse (unverändert) ---
class Obstacle:
    def __init__(self, x, y, width, height, name=""):
        self.x = x;
        self.y = y;
        self.width = width;
        self.height = height;
        self.name = name


# --- Hauptklasse für den RRT*-Algorithmus ---
class RRTStar:
    def __init__(self, start_pos, goal_pos, obstacles, bounds, step_size=5.0, max_iter=2000, search_radius=20.0,
                 log_file="rrt_star_protocol.log"):
        self.start_node = Node(start_pos[0], start_pos[1])
        self.goal_node = Node(goal_pos[0], goal_pos[1])
        self.nodes = [self.start_node]
        self.obstacles = obstacles
        self.bounds = bounds
        self.step_size = step_size
        self.max_iter = max_iter
        self.search_radius = search_radius  # Wichtiger Parameter fuer RRT*
        self.path = []
        self._setup_logger(log_file)

    def _setup_logger(self, log_file):
        self.logger = logging.getLogger('RRT_Star_Logger')
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
        self.logger.info("[SYSTEM] RRT* Protokoll gestartet.")
        self.logger.info(
            f"[CONFIG] Start: ({self.start_node.x:.1f}, {self.start_node.y:.1f}), Ziel: ({self.goal_node.x:.1f}, {self.goal_node.y:.1f})")
        self.logger.info(f"[CONFIG] Suchradius: {self.search_radius}, Schrittweite: {self.step_size}")
        self.logger.info("-" * 60)

        for i in range(self.max_iter):
            iter_str = f"[Iter. {i + 1:04d}]"

            # 1. SAMPLE & STEER (wie bei RRT)
            rand_point = self._get_random_point()
            nearest_node = self._get_nearest_node(rand_point)
            new_node = self._steer(nearest_node, rand_point)

            # 2. Kollisionsprüfung
            if not self._is_path_free(nearest_node, new_node):
                self.logger.info(f"{iter_str} KOLLISION: Pfad zu neuem Knoten blockiert.")
                continue

            # 3. CHOOSE PARENT: Finde den besten Elternteil in der Nachbarschaft
            near_nodes = self._find_near_nodes(new_node)
            best_parent, min_cost = self._choose_parent(new_node, near_nodes, nearest_node)
            new_node.parent = best_parent
            new_node.cost = min_cost
            self.nodes.append(new_node)
            self.logger.info(f"{iter_str} ADD_NODE: Füge Knoten mit Kosten {min_cost:.2f} hinzu.")

            # 4. REWIRE: Verdrahte den Baum neu, falls bessere Wege gefunden wurden
            self._rewire(new_node, near_nodes)

            # Ziel erreicht?
            if self._is_goal_reached(new_node):
                self.logger.info(f"    -> {iter_str} ZIEL ERREICHT!")
                self._reconstruct_path(new_node)
                self._validate_path()
                # Man kann hier stoppen oder weiterlaufen lassen, um den Pfad weiter zu optimieren
                # return self.path

        # Nach allen Iterationen den besten Pfad zum Ziel finden
        self.logger.info("[SYSTEM] Maximale Iterationen erreicht. Suche besten Pfad zum Ziel...")
        final_node = self._find_best_goal_candidate()
        if final_node:
            self._reconstruct_path(final_node)
            self._validate_path()
            return self.path

        self.logger.error("[SYSTEM] FEHLER: Kein Pfad zum Ziel gefunden.")
        return None

    def _choose_parent(self, new_node, near_nodes, nearest_node):
        """Wählt den kostengünstigsten Elternteil für new_node aus der Nachbarschaft."""
        best_parent = nearest_node
        min_cost = nearest_node.cost + self.distance(nearest_node, new_node)

        self.logger.info(f"    -> CHOOSE_PARENT: Start mit nächstem Knoten, Kosten: {min_cost:.2f}")

        for near_node in near_nodes:
            cost = near_node.cost + self.distance(near_node, new_node)
            if self._is_path_free(near_node, new_node) and cost < min_cost:
                min_cost = cost
                best_parent = near_node
                self.logger.info(f"        -> Finde besseren Elternteil! Neue Kosten: {min_cost:.2f}")

        return best_parent, min_cost

    def _rewire(self, new_node, near_nodes):
        """Prüft für alle Nachbarn, ob der Weg über new_node kürzer ist."""
        self.logger.info("    -> REWIRE: Prüfe Nachbarschaft...")
        for near_node in near_nodes:
            # Ignoriere den direkten Elternteil von new_node
            if near_node == new_node.parent:
                continue

            new_cost = new_node.cost + self.distance(new_node, near_node)
            if new_cost < near_node.cost and self._is_path_free(new_node, near_node):
                near_node.parent = new_node
                near_node.cost = new_cost
                self.logger.info(f"        -> VERDRAHTE NEU! Verbessere Kosten eines Nachbarn auf {new_cost:.2f}.")
                # In einer vollständigen Implementierung müssten die Kosten aller Kinder dieses Knotens aktualisiert werden.
                # Dies wird hier zur Vereinfachung weggelassen.

    def _find_near_nodes(self, new_node):
        """Findet alle Knoten innerhalb des Suchradius."""
        return [node for node in self.nodes if self.distance(node, new_node) < self.search_radius]

    def _reconstruct_path(self, final_node):
        path = []
        node = final_node
        while node is not None:
            path.append(node)
            node = node.parent
        self.path = path[::-1]  # Umdrehen, um vom Start zum Ziel zu laufen

    def _find_best_goal_candidate(self):
        """Findet den Knoten, der dem Ziel am nächsten ist und eine Verbindung hat."""
        goal_candidates = []
        for node in self.nodes:
            if self._is_goal_reached(node) and self._is_path_free(node, self.goal_node):
                goal_candidates.append(node)

        if not goal_candidates: return None

        # Wähle den Kandidaten mit den geringsten Gesamtkosten
        best_node = min(goal_candidates, key=lambda n: n.cost + self.distance(n, self.goal_node))
        self.goal_node.parent = best_node  # Finale Verbindung zum Ziel
        self.goal_node.cost = best_node.cost + self.distance(best_node, self.goal_node)
        self.nodes.append(self.goal_node)
        return self.goal_node

    # --- Die restlichen Hilfsfunktionen bleiben größtenteils gleich ---

    @staticmethod
    def distance(n1, n2):
        return np.hypot(n1.x - n2.x, n1.y - n2.y)

    def _get_random_point(self):
        if np.random.rand() > 0.95: return (self.goal_node.x, self.goal_node.y)
        return (np.random.uniform(0, self.bounds[0]), np.random.uniform(0, self.bounds[1]))

    def _get_nearest_node(self, point):
        distances = [self.distance(node, Node(point[0], point[1])) for node in self.nodes]
        return self.nodes[np.argmin(distances)]

    def _steer(self, from_node, to_point):
        direction = np.array([to_point[0] - from_node.x, to_point[1] - from_node.y])
        dist = np.linalg.norm(direction)
        if dist <= self.step_size: return Node(to_point[0], to_point[1])
        direction = (direction / dist) * self.step_size
        return Node(from_node.x + direction[0], from_node.y + direction[1])

    def _is_goal_reached(self, node):
        return self.distance(node, self.goal_node) < self.step_size

    def _is_path_free(self, from_node, to_node):
        for obs in self.obstacles:
            if self._check_line_obstacle_collision(from_node, to_node, obs): return False
        return True

    def _check_line_obstacle_collision(self, n1, n2, obs):
        p1 = np.array([n1.x, n1.y]);
        p2 = np.array([n2.x, n2.y]);
        r_min = np.array([obs.x, obs.y]);
        r_max = np.array([obs.x + obs.width, obs.y + obs.height])
        d = p2 - p1
        if np.all(np.abs(d) < 1e-9): return r_min[0] <= p1[0] <= r_max[0] and r_min[1] <= p1[1] <= r_max[1]
        t_n, t_f = -np.inf, np.inf
        for i in range(2):
            if np.abs(d[i]) < 1e-9:
                if p1[i] < r_min[i] or p1[i] > r_max[i]: return False
            else:
                t1 = (r_min[i] - p1[i]) / d[i];
                t2 = (r_max[i] - p1[i]) / d[i]
                if t1 > t2: t1, t2 = t2, t1
                t_n = max(t_n, t1);
                t_f = min(t_f, t2)
                if t_n > t_f: return False
        return t_n <= 1 and t_f >= 0

    def _validate_path(self):
        # Implementierung aus vorheriger Version übernommen
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
            n2 = self.path[i + 1];
            dist = self.distance(n1, n2)
            if dist > max_dist: max_dist = dist
            if not self._is_path_free(n1, n2):
                self.logger.error(f"    -> VALIDIERUNG FEHLGESCHLAGEN: Kollision im Segment {i}->{i + 1}")
                is_collision_free = False
        self.logger.info(f"    -> Maximale Distanz zwischen Segmenten: {max_dist:.2f}")
        if is_collision_free and start_ok and end_ok:
            self.logger.info("    -> VALIDIERUNG ERFOLGREICH.")
        else:
            self.logger.error("    -> VALIDIERUNG FEHLGESCHLAGEN.")

    def draw_graph(self):
        plt.figure(figsize=(12, 12));
        ax = plt.gca()
        for obs in self.obstacles: ax.add_patch(
            patches.Rectangle((obs.x, obs.y), obs.width, obs.height, facecolor='black', alpha=0.7, zorder=2))
        for node in self.nodes:
            if node.parent: plt.plot([node.x, node.parent.x], [node.y, node.parent.y], color='cyan', linewidth=0.5,
                                     zorder=1)
        if self.path:
            px = [node.x for node in self.path];
            py = [node.y for node in self.path]
            plt.plot(px, py, 'm-', linewidth=2.5, zorder=3)
        plt.plot(self.start_node.x, self.start_node.y, 'go', markersize=10, label='Start', zorder=4)
        plt.plot(self.goal_node.x, self.goal_node.y, 'ro', markersize=10, label='Ziel', zorder=4)
        plt.xlim(0, self.bounds[0]);
        plt.ylim(0, self.bounds[1])
        plt.gca().set_aspect('equal', adjustable='box');
        plt.title('RRT* Visualisierung')
        plt.legend();
        plt.grid(True);
        plt.show()


# --- Hauptprogramm ---
if __name__ == '__main__':
    start_position = (10, 10);
    goal_position = (90, 90);
    bounds = (100, 100)
    obstacles_list = [Obstacle(20, 10, 20, 40, name="R1"), Obstacle(60, 50, 20, 40, name="R2")]

    rrt_star = RRTStar(
        start_pos=start_position, goal_pos=goal_position,
        obstacles=obstacles_list, bounds=bounds,
        max_iter=2000,  # RRT* braucht oft mehr Iterationen
        step_size=8.0,
        search_radius=25.0  # Größerer Radius = mehr Rewiring, aber langsamer
    )

    path = rrt_star.run()
    if path:
        rrt_star.draw_graph()