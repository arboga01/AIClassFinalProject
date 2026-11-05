import sys
import json
import random
import time
import numpy as np
from PyQt6.QtCore import QRunnable, QThreadPool, QObject, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QMessageBox, QGroupBox, QFileDialog, QCheckBox 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

# random dataset to use as place hold

def generate_random_data(num_hubs=25, map_size=100.0):
    # generating a generic random dataset as placeholder
    data = []
    for i in range(num_hubs):
        data.append({
            "id": i,
            "label": f"Hub_{i}",
            "x": round(random.uniform(5.0, map_size - 5.0), 2),
            "y": round(random.uniform(5.0, map_size - 5.0), 2),
        })

        #connected distance and cost matrix
    matrix = []
    for i in range(num_hubs):
            row = []
            for j in range(num_hubs):
                if i == j:
                    row.append(0)
                else:
                    x1, y1 = data[i]["x"], data[i]["y"]
                    x2, y2 = data[j]["x"], data[j]["y"]
                    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                    row.append(round(distance, 2))
            matrix.append(row)

    return {num_hubs: num_hubs, "nodes" : data, "cost_matrix" : matrix}

#Thread safe code also where thealgorithm is running so it runs on a seperate thread where any other algorithms are to be put

class WorkSignals(QObject):
    progress = pyqtSignal(int, float, float,  list) #generation, GA_cost, Woac_cost, edges
    finished = pyqtSignal(float, float, list)

class GAWorkRunnable(QRunnable):
    def __init__(self, dataset, pop_size=200, generations=150, mutation_rate=0.15, use_woac=True):
        super().__init__()
        self.dataset = dataset
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.use_woac = use_woac
        self.signals = WorkSignals()
        self._is_running = True

    @pyqtSlot()
    def run(self):
        # Simulate a long-running genetic algorithm process
        nodes = self.dataset["nodes"]
        num_nodes = len(nodes)
        curr_ga_cost = 1000.0 + num_nodes * 20.0
        curr_woac_cost = curr_ga_cost * 0.95 if self.use_woac else curr_ga_cost

        for generation in range(1, self.generations + 1):
            if not self._is_running:
                break

            # Simulate some processing time
            time.sleep(0.05)

            # Simulate cost reduction over generations
            curr_ga_cost *= random.uniform(0.95, 0.99)
            if self.use_woac:
                curr_woac_cost *= random.uniform(0.95, 0.99)
            else:
                curr_woac_cost = curr_ga_cost
            #temporary random Spanning tree, replace with the official dataset that is used
            available_nodes = list(range(1, num_nodes))
            connected_nodes = [0]
            curr_tree_edges =  []
            while available_nodes:
                u = random.choice(connected_nodes)
                v = available_nodes.pop(random.randint(0, len(available_nodes) - 1))
                curr_tree_edges.append((u, v))
                connected_nodes.append(v)

            self.signals.progress.emit(generation, round(curr_ga_cost, 2), round(curr_woac_cost, 2), curr_tree_edges)
        self.signals.finished.emit(round(curr_ga_cost, 2), round(curr_woac_cost, 2), curr_tree_edges)

    def stop(self):
        self._is_running = False

    # main visualizer for minimum routing cost spanning tree, this is where the main GUI is built
class MRCSTVisualizer(QMainWindow):
    def __init__(self):
            super().__init__()
            self.setWindowTitle("Supply chain visualizer using MRCST")
            self.resize(1200, 750)

            # dataset and solver state
            self.dataset = generate_random_data(num_hubs=25)
            self.thread_pool = QThreadPool.globalInstance()
            self.active_worker = None

            self.history_generations = []
            self.history_ga_costs = []
            self.history_woac_costs = []

            #UI structure
            self.init_ui()
            self.plot_initial_network()

    def init_ui(self):
            main_widget = QWidget()
            self.setCentralWidget(main_widget)
            main_layout = QHBoxLayout(main_widget)

            #left control pannel
            control_panel = self.build_control_panel()
            main_layout.addWidget(control_panel, stretch=1)

            #Right visualization Area
            canvas_layout = QVBoxLayout()
            self.fig, (self.ax_map, self.ax_curve) = plt.subplots(1, 2, figsize=(12, 6))
            self.canvas = FigureCanvas(self.fig)
            canvas_layout.addWidget(self.canvas)

            main_layout.addLayout(canvas_layout, stretch=3)

    def build_control_panel(self):
            panel = QWidget()
            layout = QVBoxLayout(panel)

            #data management

            data_box = QGroupBox("Data Source")
            data_layout = QVBoxLayout(data_box)

            self.btn_load_json = QPushButton("load the dataset")
            self.btn_load_json.clicked.connect(self.load_json_file)
            data_layout.addWidget(self.btn_load_json)

            self.btn_gen_random = QPushButton("generate random hubs")
            self.btn_gen_random.clicked.connect(self.generate_new_random_data)
            data_layout.addWidget(self.btn_gen_random)

            layout.addWidget(data_box)

            #Algorithm Parameter groups
            param_box = QGroupBox("Ga and WoAC settings")
            param_layout = QVBoxLayout(param_box)

            param_layout.addWidget(QLabel("population Size:"))
            self.spin_pop = QSpinBox()
            self.spin_pop.setRange(20, 1000)
            self.spin_pop.setValue(200)
            param_layout.addWidget(self.spin_pop)

            param_layout.addWidget(QLabel("Generations:"))
            self.spin_gen = QSpinBox()
            self.spin_gen.setRange(10, 5000)
            self.spin_gen.setValue(150)
            param_layout.addWidget(self.spin_gen)

            param_layout.addWidget(QLabel("mutation rate:"))
            self.spin_mut = QDoubleSpinBox()
            self.spin_mut.setRange(0.01, 1.0)
            self.spin_mut.setSingleStep(0.05)
            self.spin_mut.setValue(0.15)
            param_layout.addWidget(self.spin_mut)

            self.chk_woac = QCheckBox("Use (WoAC) Wisdom of Artificial Crowds")
            self.chk_woac.setChecked(True)
            param_layout.addWidget(self.chk_woac)

            layout.addWidget(param_box)

            #Execution Control group
            exec_box = QGroupBox("Execution")
            exec_layout = QVBoxLayout(exec_box)

            self.btn_run = QPushButton("run solver")
            self.btn_run.setStyleSheet("background-color: white; font-weight: bold;")
            self.btn_run.clicked.connect(self.start_solver)
            exec_layout.addWidget(self.btn_run)

            self.btn_stop =  QPushButton("Stop")
            self.btn_stop.setEnabled(False)
            self.btn_stop.clicked.connect(self.stop_solver)
            exec_layout.addWidget(self.btn_stop)

            self.btn_export = QPushButton("Export report")
            self.btn_export.clicked.connect(self.export_snapshot)
            exec_layout.addWidget(self.btn_export)

            layout.addWidget(exec_box)

            #status readout
            self.lbl_status = QLabel("status: Idle")
            self.lbl_cost = QLabel("best Cost: ")
            layout.addWidget(self.lbl_status)
            layout.addWidget(self.lbl_cost)
            layout.addStretch()

            return panel

#plotting and rendering
    def get_node_label(self, node_id):
        """Helper to generate clear logistics roles based on Node ID."""
        if node_id == 0:
            return "Factory / Main Warehouse"
        elif 1 <= node_id <= 5:
            return f"Regional Hub {node_id}"
        else:
            return f"Retail Store {node_id}"
    def plot_initial_network(self):
        #render initial hub coordinates on canvas
        self.ax_map.clear()
        self.ax_curve.clear()

        nodes = self.dataset["nodes"]
        xs = [n["x"] for n in nodes]
        ys = [n["y"] for n in nodes]

        node_colors  =['gold' if n['id'] == 0 else 'navy' for n in nodes]
        node_sizes = [250 if n['id'] == 0 else 120 for n in nodes]

        self.ax_map.scatter(xs, ys, c=node_colors, s=node_sizes, zorder=3, edgecolors='black')

    
        for n in nodes:
            label = self.get_node_label(n['id'])
            self.ax_map.annotate(f" {label}", (n['x'], n['y']), fontsize=8, fontweight='bold')
        self.ax_map.set_title("Supply chain hub spatial network")
        self.ax_map.set_xlabel("X coordinate (km)")
        self.ax_map.set_ylabel("Y coordinate (km)")
        self.ax_map.grid(True, linestyle='--', alpha=0.5)

        self.ax_curve.set_title("MRCST convergence curves")
        self.ax_curve.set_xlabel("Generations")
        self.ax_curve.set_ylabel("Cost")
        self.ax_curve.grid(True, linestyle='--', alpha=0.5)

        self.fig.tight_layout()
        self.canvas.draw()

    def update_live_render(self, generation, ga_cost, woac_cost, edges):
        #update the live render of the MRCST and convergence curve
        self.lbl_status.setText(f"Status: running (generation {generation}/ {self.spin_gen.value()})")
        display_cost = woac_cost if self.chk_woac.isChecked() else ga_cost
        self.lbl_cost.setText(f"Best Cost: {display_cost}")

        self.history_generations.append(generation)
        self.history_ga_costs.append(ga_cost)
        self.history_woac_costs.append(woac_cost)

        #redraw spatial tree network
        self.ax_map.clear()
        nodes = self.dataset["nodes"]
        xs = [n["x"] for n in nodes]
        ys = [n["y"] for n in nodes]
        node_colors = ['gold' if n['id'] == 0 else 'navy' for n in nodes]
        node_sizes = [250 if n['id'] == 0 else 120 for n in nodes]

        self.ax_map.scatter(xs, ys, c=node_colors, s=node_sizes, zorder=4, edgecolors='black')
#draw supply chain hub nodes
        for n in nodes:
            short_label = "Factory" if n['id'] == 0 else (f"Hub {n['id']}" if 1 <= n['id'] <= 5 else f"Store {n['id']}")
            self.ax_map.annotate(f" {short_label}", (n['x'], n['y']), fontsize=8, fontweight='bold')

    #draw current spanning tree edges
        for u, v in edges:
            x1, y1 = nodes[u]["x"], nodes[u]["y"]
            x2, y2 = nodes[v]["x"], nodes[v]["y"]
            self.ax_map.plot([x1, x2], [y1, y2], c='orange', linewidth=2, zorder=2)

        self.ax_map.set_title("Supply chain hub spatial network")
        self.ax_map.grid(True, linestyle='--', alpha=0.5)

    #update convergence curve
        self.ax_curve.clear()
        self.ax_curve.plot(self.history_generations, self.history_ga_costs, c='green', label='Standard GA', linewidth=2)
        if self.chk_woac.isChecked():
            self.ax_curve.plot(self.history_generations, self.history_woac_costs, c='blue', label='WoAC GA', linewidth=2)

        self.ax_curve.set_title("MRCST convergence curve")
        self.ax_curve.set_xlabel("Generations")
        self.ax_curve.set_ylabel("Cost")
        self.ax_curve.grid(True, linestyle='--', alpha=0.5)

        self.fig.tight_layout()
        self.canvas.draw()

#UI event handlers
    def clear_history(self):
        self.history_generations.clear()
        self.history_ga_costs.clear()
        self.history_woac_costs.clear()
    def generate_new_random_data(self):
        self.dataset = generate_random_data(num_hubs=25)
        self.clear_history()
        self.plot_initial_network()
        self.lbl_status.setText("Status: generated random hubs")

    def load_json_file(self):
        file_path, _ =QFileDialog.getOpenFileName(self, "open dataset", "", "JSON files (*.json);;All files (*.*)")
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    self.dataset = json.load(f)
                self.clear_history()
                self.plot_initial_network()
                self.lbl_status.setText(f"Status: loaded dataset from {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load dataset: {str(e)}")
    def start_solver(self):
        self.clear_history()

        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_load_json.setEnabled(False)
        self.btn_gen_random.setEnabled(False)

    #create runnable worker
        self.active_worker = GAWorkRunnable(
            dataset=self.dataset,
            pop_size=self.spin_pop.value(),
            generations=self.spin_gen.value(),
            mutation_rate=self.spin_mut.value(),
            use_woac=self.chk_woac.isChecked()
        )
    # connect thread safe signals
        self.active_worker.signals.progress.connect(self.update_live_render)
        self.active_worker.signals.finished.connect(self.solver_finished)

    #submit task to Qt's Thread pool
        self.thread_pool.start(self.active_worker)

    def stop_solver(self):
        if self.active_worker:
            self.active_worker.stop()
            self.lbl_status.setText("Status: stopping solver...")
    def solver_finished(self, ga_cost, woac_cost, edges):
        final_cost = woac_cost if self.chk_woac.isChecked() else ga_cost
        self.lbl_status.setText("Status: solver finished")
        self.lbl_cost.setText(f"Best Cost: {final_cost}")

        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_load_json.setEnabled(True)
        self.btn_gen_random.setEnabled(True)

    #final render of the MRCST
        self.ax_map.clear()
        nodes = self.dataset["nodes"]
        xs = [n["x"] for n in nodes]
        ys = [n["y"] for n in nodes]

        node_colors = ['gold' if n['id'] == 0 else 'navy' for n in nodes]
        node_sizes = [250 if n['id'] == 0 else 120 for n in nodes]

        self.ax_map.scatter(xs, ys, c=node_colors, s=node_sizes, zorder=4, edgecolors='black')
        for n in nodes:
            label = self.get_node_label(n['id'])
            self.ax_map.annotate(f" {label}", (n['x'], n['y']), fontsize=8, fontweight='bold')

        for u, v in edges:
            x1, y1 = nodes[u]["x"], nodes[u]["y"]
            x2, y2 = nodes[v]["x"], nodes[v]["y"]
            self.ax_map.plot([x1, x2], [y1, y2], c='orange', linewidth=2, zorder=2)

        self.ax_map.set_title("Supply chain hub spatial network (Final MRCST)")
        self.ax_map.grid(True, linestyle='--', alpha=0.5)

        self.fig.tight_layout()
        self.canvas.draw()

    def export_snapshot(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Export report", "", "PNG files (*.png);;All files (*.*)")
        if file_path:
            try:
                self.fig.savefig(file_path)
                QMessageBox.information(self, "Export Successful", f"Report exported to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export report: {str(e)}")
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MRCSTVisualizer()
    window.show()
    sys.exit(app.exec())
#https://www.pythonguis.com/tutorials/multithreading-pyqt6-applications-qthreadpool/


#https://www.geeksforgeeks.org/python/python-designing-gui-applications-using-pyqt/





