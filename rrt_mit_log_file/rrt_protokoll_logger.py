import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import datetime
import logging  # 1. Modul importieren


# --- Klasse für einen einzelnen Knoten im Baum ---
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.id = -1


# --- Klasse zur Repraesentation eines Hindernisses ---
class Obstacle:
    def __init__(self, x, y, width, height, name=""):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.name = name

    def check_collision(self, point_x, point_y):
        """Prueft, ob ein Punkt innerhalb des Hindernisses liegt."""
        return self.x <= point_x <= self.x + self.width and \
            self.y <= point_y <= self.y + self.height


# --- Hauptklasse fuer den RRT-Algorithmus ---
class RRT:
    def __init__(self, start_pos, goal_pos, obstacles, bounds, step_size=8.0, max_iter=100,
                 log_file="rrt_protocol.log"):
        self.start_node = Node(start_pos[0], start_pos[1])
        self.start_node.id = 0
        self.goal_node = Node(goal_pos[0], goal_pos[1])
        self.nodes = [self.start_node]
        self.obstacles = obstacles
        self.bounds = bounds
        self.step_size = step_size
        self.max_iter = max_iter
        self.next_node_id = 1

        # 2. Logger aufsetzen
        self._setup_logger(log_file)

    def _setup_logger(self, log_file):
        """Konfiguriert den Logger fuer die Ausgabe in Konsole und Datei."""
        # Verhindert, dass Handler bei mehrmaliger Initialisierung dupliziert werden
        if hasattr(self, 'logger') and self.logger.hasHandlers():
            return

        self.logger = logging.getLogger('RRT_Logger')
        self.logger.setLevel(logging.INFO)

        # Formatter, der nur die reine Nachricht ausgibt (wie bei print)
        formatter = logging.Formatter('%(message)s')

        # Handler fuer die Datei (mode='w' ueberschreibt die alte Datei)
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(formatter)

        # Handler fuer die Konsole
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def run(self):
        """Fuehrt den RRT-Algorithmus aus und erzeugt das Protokoll."""

        # --- Protokoll-Header ---
        # 3. Alle print() durch self.logger.info() ersetzen
        self.logger.info("[SYSTEM] RRT-Protokoll gestartet.")
        self.logger.info(f"[SYSTEM] Datum/Zeit: {datetime.datetime.now().strftime('%d.%m.%Y %H:%M:%S CEST')}")
        self.logger.info(f"[CONFIG] Konfigurationsraum: Groesse [{self.bounds[0]}, {self.bounds[1]}]")
        self.logger.info(f"[CONFIG] Startknoten N0 an Position: ({self.start_node.x}, {self.start_node.y})")
        self.logger.info(f"[CONFIG] Zielposition: ({self.goal_node.x}, {self.goal_node.y})")
        self.logger.info(f"[CONFIG] Schrittweite (delta_q): {self.step_size}")
        for obs in self.obstacles:
            self.logger.info(
                f"[CONFIG] Hindernis '{obs.name}' definiert: von ({obs.x}, {obs.y}) bis ({obs.x + obs.width}, {obs.y + obs.height})")
        self.logger.info("-" * 60)

        for i in range(self.max_iter):
            iter_str = f"[Iter. {i + 1:03d}]"
            self.logger.info(f"{iter_str} START")

            rand_point = self._get_random_point()
            self.logger.info(
                f"{iter_str} SAMPLE: Generiere Zufallspunkt x_rand -> ({rand_point[0]:.1f}, {rand_point[1]:.1f})")

            nearest_node = self._get_nearest_node(rand_point)
            self.logger.info(
                f"{iter_str} NEAREST: -> Naechster Knoten ist N{nearest_node.id} ({nearest_node.x:.1f}, {nearest_node.y:.1f})")

            new_node = self._steer(nearest_node, rand_point)
            self.logger.info(f"{iter_str} STEER: -> Neue Position fuer x_new: ({new_node.x:.1f}, {new_node.y:.1f})")

            if not self._is_path_free(nearest_node, new_node):
                self.logger.info(f"{iter_str} COLLISION_CHECK: -> KOLLISION! Pfad kreuzt ein Hindernis.")
                self.logger.info(f"{iter_str} DISCARD_NODE: Verwerfe x_new. Keine Aenderung am Baum.")
                self.logger.info(f"{iter_str} END (Baumgroesse: {len(self.nodes)} Knoten)")
                self.logger.info("-" * 60)
                continue

            self.logger.info(f"{iter_str} COLLISION_CHECK: -> Pfad ist frei.")

            new_node.parent = nearest_node
            new_node.id = self.next_node_id
            self.next_node_id += 1
            self.nodes.append(new_node)
            self.logger.info(
                f"{iter_str} ADD_NODE: Fuege neuen Knoten N{new_node.id} hinzu. Parent: N{nearest_node.id}.")

            dist_to_goal = np.sqrt((new_node.x - self.goal_node.x) ** 2 + (new_node.y - self.goal_node.y) ** 2)
            if dist_to_goal <= self.step_size:
                self.goal_node.parent = new_node
                self.nodes.append(self.goal_node)
                self.logger.info(f"[SYSTEM] GOAL_CHECK: Neuer Knoten N{new_node.id} liegt nahe am Ziel!")
                self.logger.info("[SYSTEM] ZIEL ERREICHT! Pfad gefunden.")
                self.logger.info(f"{iter_str} END (Baumgroesse: {len(self.nodes)} Knoten)")
                self.logger.info("-" * 60)
                break

            self.logger.info(f"{iter_str} END (Baumgroesse: {len(self.nodes)} Knoten)")
            self.logger.info("-" * 60)
        else:
            self.logger.info("[SYSTEM] RRT beendet. Maximale Iterationen erreicht. Ziel nicht gefunden.")

        return self.nodes

    def _get_random_point(self):
        if np.random.rand() > 0.95:
            return (self.goal_node.x, self.goal_node.y)
        x = np.random.uniform(0, self.bounds[0])
        y = np.random.uniform(0, self.bounds[1])
        return (x, y)

    def _get_nearest_node(self, point):
        distances = [np.sqrt((node.x - point[0]) ** 2 + (node.y - point[1]) ** 2) for node in self.nodes]
        nearest_index = np.argmin(distances)
        return self.nodes[nearest_index]

    def _steer(self, from_node, to_point):
        direction = np.array([to_point[0] - from_node.x, to_point[1] - from_node.y])
        distance = np.linalg.norm(direction)
        if distance < self.step_size:
            return Node(to_point[0], to_point[1])
        direction = (direction / distance) * self.step_size
        new_pos = np.array([from_node.x + direction[0], from_node.y + direction[1]])
        return Node(new_pos[0], new_pos[1])

    def _is_path_free(self, from_node, to_node):
        for obs in self.obstacles:
            if obs.check_collision(to_node.x, to_node.y):
                return False
        return True

    def draw_graph(self):
        plt.figure(figsize=(12, 12))
        ax = plt.gca()
        for obs in self.obstacles:
            ax.add_patch(patches.Rectangle((obs.x, obs.y), obs.width, obs.height, facecolor='black', alpha=0.7))
        for node in self.nodes:
            if node.parent:
                plt.plot([node.x, node.parent.x], [node.y, node.parent.y], 'c-')
        path_node = self.nodes[-1]
        if path_node.parent:  # Nur zeichnen, wenn ein Pfad existiert
            while path_node.parent is not None:
                plt.plot([path_node.x, path_node.parent.x], [path_node.y, path_node.parent.y], 'm-', linewidth=2.5)
                path_node = path_node.parent
        plt.plot(self.start_node.x, self.start_node.y, 'go', markersize=10, label='Start')
        plt.plot(self.goal_node.x, self.goal_node.y, 'ro', markersize=10, label='Ziel')
        plt.xlim(0, self.bounds[0])
        plt.ylim(0, self.bounds[1])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title('RRT-Algorithmus Visualisierung')
        plt.legend()
        plt.grid(True)
        plt.show()


# --- Hauptprogramm ---
if __name__ == '__main__':
    start_position = (10, 10)
    goal_position = (95, 95)
    bounds = (100, 100)

    obstacles_list = [
        Obstacle(40, 20, 20, 60, name="Rect1"),
    ]

    rrt = RRT(
        start_pos=start_position,
        goal_pos=goal_position,
        obstacles=obstacles_list,
        bounds=bounds,
        max_iter=200,
        log_file="rrt_protocol.log"  # Dateiname fuer das Protokoll
    )

    nodes_found = rrt.run()
    rrt.draw_graph()
