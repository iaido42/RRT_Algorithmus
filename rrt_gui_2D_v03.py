# Importiere benötigte Module
import tkinter as tk  # GUI-Grundelemente
from tkinter import ttk, messagebox  # Erweiterte GUI-Elemente
import random  # Für Zufallszahlen
import math  # Mathematische Funktionen (z.B. Hypotenuse, Winkelberechnung)
import json  # Speichern/Laden von Konfigurationen
import os  # Dateisystemzugriff (z.B. Prüfen ob Datei existiert)
import pyautogui #  für die Automatisierung von GUI-Interaktionen und Bildschirmaufnahmen
# Dateiname für die gespeicherte Konfiguration
CONFIG_FILE = "rrt_config.json"

#================================================================
#
# Rapidly-exploring Random Trees (RRT)
# Ein moderner Ansatz zur Pfadplanung
# Seminararbeit im Modul Algorithmen und Datenstrukturen
# Autor:
# Wilfried Ornowski
# Studiengang: Informatik.Softwaresystem
# Semester: SoSe 2025
# Prüfer:
# Prof. Dr.-Ing. Martin Guddat
# Abgabedatum:
# 20. Mai 2025
# Campus Bocholt
# Fachbereich Wirtschaft und Informationstechnik
#-----------
# RRT Algorithmus Visualisierung in 2D mit automatischen Screenshots
# RRT Algorithm with visual 2D rendering
# with automatic Screenshots
#================================================================

# Standardkonfiguration, falls keine Datei vorhanden ist
DEFAULT_CONFIG = {
    "start_x": 50,
    "start_y": 50,
    "goal_x": 750,
    "goal_y": 550,
    "step_size": 20,
    "max_iter": 1000,
    "steps_per_frame": 1,
    "mode": "continuous"  # oder "step"
}

# Zeichenbereich und andere feste Werte
CANVAS_WIDTH = 800
CANVAS_HEIGHT = 600
NODE_RADIUS = 5
GOAL_RADIUS = 20
OBSTACLES = [
    (200, 150, 100, 200),
    (400, 100, 50, 300),
    (600, 250, 150, 150)
]

# Berechnet den euklidischen Abstand zwischen zwei Punkten
def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# Prüft, ob die Verbindung zwischen zwei Punkten ein Hindernis durchquert
def is_collision(p1, p2, obstacles):
    for (ox, oy, ow, oh) in obstacles:
        for i in range(11):  # Interpolation von 11 Zwischenpunkten
            u = i / 10
            x = p1[0] * (1 - u) + p2[0] * u
            y = p1[1] * (1 - u) + p2[1] * u
            if ox <= x <= ox + ow and oy <= y <= oy + oh:
                return True  # Punkt liegt innerhalb eines Hindernisses
    return False

# Gibt einen zufälligen Punkt im Zeichenbereich zurück
def get_random_point():
    return random.randint(0, CANVAS_WIDTH), random.randint(0, CANVAS_HEIGHT)

# Lädt Konfigurationswerte aus Datei oder nutzt Standardwerte
def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return DEFAULT_CONFIG.copy()

# Speichert aktuelle Konfigurationswerte in eine Datei
def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)

# Implementiert den RRT-Algorithmus (Rapidly-exploring Random Tree)
class RRT:
    def __init__(self, start, goal, obstacles, step_size):
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.step_size = step_size
        self.nodes = [start]  # Baumknoten
        self.parent = {start: None}  # Rückverfolgungspfad
        self.found_path = []  # Ergebnispfad, wenn Ziel erreicht wurde

    # Führt einen Schritt im RRT aus
    def step(self):
        rnd = get_random_point()  # Zielpunkt (zufällig)
        nearest = min(self.nodes, key=lambda p: dist(p, rnd))  # Nächstgelegener Knoten
        theta = math.atan2(rnd[1] - nearest[1], rnd[0] - nearest[0])  # Richtung zum Ziel

        # Erzeuge neuen Punkt in Richtung des Zielpunkts
        new_point = (
            int(nearest[0] + self.step_size * math.cos(theta)),
            int(nearest[1] + self.step_size * math.sin(theta))
        )

        # Prüfe, ob der Punkt außerhalb des erlaubten Bereichs liegt
        if not (0 <= new_point[0] <= CANVAS_WIDTH and 0 <= new_point[1] <= CANVAS_HEIGHT):
            return None, None

        # Prüfe, ob die Verbindung kollidiert
        if is_collision(nearest, new_point, self.obstacles):
            return None, None

        # Füge neuen Knoten hinzu
        self.nodes.append(new_point)
        self.parent[new_point] = nearest

        # Zielprüfung
        if dist(new_point, self.goal) < GOAL_RADIUS:
            self.found_path = self.retrace_path(new_point)
            return new_point, nearest

        return new_point, nearest

    # Rückverfolgungspfad vom Ziel zum Start
    def retrace_path(self, end_point):
        path = []
        while end_point:
            path.append(end_point)
            end_point = self.parent[end_point]
        return path[::-1]  # Umkehrung für Start→Ziel

# Die GUI-Klasse zur Visualisierung und Benutzerinteraktion
class RRTApp:
    def __init__(self, master):
        self.master = master
        self.master.title("RRT Visualisierung erweitert")
        self.config = load_config()
        self.entries = {}  # Input-Felder
        self.setup_ui()
        self.running = False
        self.iteration = 0

    # Erzeugt alle GUI-Elemente
    def setup_ui(self):
        controls = tk.Frame(self.master)
        controls.pack(side=tk.TOP, fill=tk.X)

        # Erzeugt Eingabefelder für alle Parameter
        for label, key in [
            ("Start X", "start_x"), ("Start Y", "start_y"),
            ("Goal X", "goal_x"), ("Goal Y", "goal_y"),
            ("Step Size", "step_size"),
            ("Max Iter", "max_iter"),
            ("Steps / Frame", "steps_per_frame")
        ]:
            frame = tk.Frame(controls)
            frame.pack(side=tk.LEFT, padx=4)
            tk.Label(frame, text=label).pack()
            e = tk.Entry(frame, width=6)
            e.insert(0, str(self.config.get(key)))
            e.pack()
            self.entries[key] = e

        # Auswahl des Ausführungsmodus
        self.mode_var = tk.StringVar(value=self.config.get("mode", "continuous"))
        mode_menu = ttk.Combobox(controls, textvariable=self.mode_var,
                                 values=["step", "continuous"], state="readonly", width=10)
        mode_menu.pack(side=tk.LEFT, padx=6)

        # Buttons zum Starten und Zurücksetzen
        tk.Button(controls, text="Start", command=self.start).pack(side=tk.LEFT, padx=6)
        tk.Button(controls, text="Reset", command=self.reset_defaults).pack(side=tk.LEFT, padx=6)

        # Zeichenbereich
        self.canvas = tk.Canvas(self.master, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg="white")
        self.canvas.pack(side=tk.LEFT)

        # Seitenbereich für Koordinatenausgabe
        side_panel = tk.Frame(self.master)
        side_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        self.coord_text = tk.Text(side_panel, width=40, height=30)
        self.coord_text.pack()
        self.coord_text.config(state=tk.DISABLED)
        self.toggle_btn = tk.Button(side_panel, text="Koordinaten ausblenden", command=self.toggle_coords)
        self.toggle_btn.pack(pady=5)

        self.show_coords = True
        self.master.bind("<space>", self.manual_step)  # Space-Taste für manuellen Schritt

    def take_screenshot(self, reason):
        """
        Erstellt einen Screenshot des aktuellen Fensters.

        Args:
            reason (str): Grund für den Screenshot, wird im Dateinamen verwendet.
        """
        x = self.master.winfo_rootx()
        y = self.master.winfo_rooty()
        w = self.master.winfo_width()
        h = self.master.winfo_height()

        # Erstellt einen Ordner für Screenshots, falls er nicht existiert
        os.makedirs("screenshots", exist_ok=True)

        dateiname = f"screenshots/screenshot_rrt_{reason}_{self.iteration}.png"
        screenshot = pyautogui.screenshot(region=(x, y, w, h))
        screenshot.save(dateiname)
        print(f"Screenshot '{dateiname}' gespeichert.")

    # Koordinatenausgabe ein-/ausblenden
    def toggle_coords(self):
        if self.show_coords:
            self.coord_text.pack_forget()
            self.toggle_btn.config(text="Koordinaten einblenden")
        else:
            self.coord_text.pack()
            self.toggle_btn.config(text="Koordinaten ausblenden")
        self.show_coords = not self.show_coords

    # Setzt alle Eingaben auf Standardwerte zurück
    def reset_defaults(self):
        for key in DEFAULT_CONFIG:
            if key in self.entries:
                self.entries[key].delete(0, tk.END)
                self.entries[key].insert(0, str(DEFAULT_CONFIG[key]))
        self.mode_var.set(DEFAULT_CONFIG["mode"])

    # Startet den Algorithmus
    def start(self):
        self.canvas.delete("all")
        self.iteration = 0
        self.running = True

        # Konfiguration aus Eingabefeldern lesen
        try:
            cfg = {key: int(entry.get()) for key, entry in self.entries.items()}
        except ValueError:
            messagebox.showerror("Fehler", "Bitte gültige Ganzzahlen eingeben.")
            return

        cfg["mode"] = self.mode_var.get()
        self.config = cfg

        self.start_point = (cfg["start_x"], cfg["start_y"])
        self.goal_point = (cfg["goal_x"], cfg["goal_y"])

        self.rrt = RRT(self.start_point, self.goal_point, OBSTACLES, cfg["step_size"])
        self.max_iter = cfg["max_iter"]
        self.steps_per_frame = cfg["steps_per_frame"]

        self.draw_static()
        if cfg["mode"] == "continuous":
            self.master.after(10, self.update)

    # Führt einen manuellen Schritt (bei Modus "step") aus
    def manual_step(self, event=None):
        if self.running and self.config["mode"] == "step":
            self.update_step()

    # Kontinuierlicher Ablauf (wird regelmäßig durch `after()` aufgerufen)
    def update(self):
        if not self.running:
            return
        for _ in range(self.steps_per_frame):
            self.update_step()
            if not self.running:
                break
        if self.running and self.config["mode"] == "continuous":
            self.master.after(5, self.update)

    # Ein Einzelschritt im RRT-Ablauf
    def update_step(self):
        self.iteration += 1


        if self.iteration in [50, 250, 500, 750, 1000]:
            print(f"Ein Einzelschritt i= {self.iteration}")  # Moderner f-String für die Ausgabe

            x = self.master.winfo_rootx()
            y = self.master.winfo_rooty()
            w = self.master.winfo_width()
            h = self.master.winfo_height()

            # Erstellen einen dynamischen Dateinamen
            dateiname = f"screenshot_rrt_{self.iteration}.png"

            screenshot = pyautogui.screenshot(region=(x, y, w, h))
            screenshot.save(dateiname)

            # Passe Bestätigungsmeldung dynamisch an
            print(f"Screenshot bei Schritt {self.iteration} als '{dateiname}' gespeichert.")

        point, parent = self.rrt.step()
       # self.iteration += 1
        if point and parent:
            self.canvas.create_line(parent[0], parent[1], point[0], point[1], fill="blue")
            self.show_point_info(point, parent)
        if self.rrt.found_path:
            self.draw_path(self.rrt.found_path)
            # Verzögere den Screenshot um 100 Millisekunden, um sicherzustellen, dass der Pfad gezeichnet ist
            self.master.after(3000,  lambda: self.take_screenshot("goal_reached") ) # Screenshot beim Erreichen des Ziels
            self.running = False
        elif self.iteration >= self.max_iter:
            messagebox.showinfo("Info", "Maximale Iterationen erreicht, kein Pfad gefunden.")
            self.running = False

    # Zeichnet statische Elemente (Hindernisse, Start/Ziel)
    def draw_static(self):
        for (x, y, w, h) in OBSTACLES:
            self.canvas.create_rectangle(x, y, x + w, y + h, fill="gray")
        self.draw_node(self.start_point, "green", "Start")
        self.draw_node(self.goal_point, "red", "Goal")

    # Zeichnet einen Knoten (Start, Ziel oder Baumknoten)
    def draw_node(self, pos, color, label):
        x, y = pos
        self.canvas.create_oval(x - NODE_RADIUS, y - NODE_RADIUS, x + NODE_RADIUS, y + NODE_RADIUS, fill=color)
        self.canvas.create_text(x, y - 10, text=label, fill=color)

    # Zeichnet den gefundenen Pfad
    def draw_path(self, path):
        for i in range(len(path) - 1):
            self.canvas.create_line(path[i][0], path[i][1], path[i + 1][0], path[i + 1][1], fill="orange", width=3)

    # Gibt Koordinaten eines neuen Punkts im Textfeld aus
    def show_point_info(self, current, parent):
        if not self.show_coords:
            return
        self.coord_text.config(state=tk.NORMAL)
        self.coord_text.insert(tk.END, f"Neu: {current}, Von: {parent}\n")
        self.coord_text.see(tk.END)
        self.coord_text.config(state=tk.DISABLED)

    # Bei Beenden speichern und schließen
    def on_close(self):
        save_config(self.config)
        self.master.destroy()

# Hauptprogrammstart
if __name__ == "__main__":
    root = tk.Tk()
    app = RRTApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
