import json
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

class PathsChosen():
    def __init__(self):
        # self.halfway_targets("halfway_exploration_log_ClosestFrontier.json", "Closest Frontier")
        # self.halfway_targets("halfway_exploration_log_WaveFront.json", "Wave Front")
        # self.halfway_targets_frontier("halfway_exploration_log_WaveFront.json", "Wave Front")
        self.load_and_plot_area_layout("AreaLayout.JSON")

    def load_and_plot_area_layout(self, json_file_path):
        # 1. Load the JSON file
        with open(json_file_path, 'r') as f:
            data = json.load(f)

        plt.figure(figsize=(16, 6))

        # Get grid dimensions from fullarea
        width = data['fullarea']['width']
        height = data['fullarea']['height']

        # 2. Plot all walls as black line segments
        for wall in data.get('walls', []):
            start = wall['start']
            end = wall['end']
            plt.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color='black',
                linewidth=2,
                solid_capstyle='round',
            )

        # 3. Plot doors as orange points to visualize openings/choke points
        doors = data.get('doors', [])
        if doors:
            door_x = [d[0] for d in doors]
            door_y = [d[1] for d in doors]
            plt.scatter(
                door_x, door_y, color='orange', s=30, zorder=5, label='Doors / Openings'
            )

        # 4. Format the plot for a Top-Left Origin
        plt.xlim(-2, width + 2)
        plt.ylim(
            height + 2, -2
        )  # Flips the Y-axis so 0 is at the top, matching your JSON grid
        plt.gca().set_aspect('equal', adjustable='box')

        plt.title('Area Layout Map (Top-Left Origin)')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend(loc='upper right')
        plt.grid(True, linestyle=":", alpha=0.3)
        plt.tight_layout()
        output_filename = f"AreaLayout.png"
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')

    def halfway_targets_frontier(
    self,
    log_filename,
    algorithm_name,
    layout_filename="AreaLayout.JSON",  # Adjust filename if yours is named differently
    ):
        try:
            with open(log_filename, "r") as f:
                lines = f.readlines()
                if not lines:
                    print("Log file is empty!")
                    return
                data = json.loads(lines[-1])
        except FileNotFoundError:
            print(f"Error: Log file '{log_filename}' not found.")
            return

        step = data["step"]
        agents = data["agents"]
        scanned_grid = np.array(data["scanned_area"])
        grid_height, grid_width = scanned_grid.shape

        # 2. Load the area layout JSON for ground-truth environment details
        layout_walls = []
        layout_doors = []
        layout_targets = []
        try:
            with open(layout_filename, "r") as f:
                layout_data = json.load(f)
                layout_walls = layout_data.get("walls", [])
                layout_doors = layout_data.get("doors", [])
                # layout_targets = layout_data.get("targets", [])
        except FileNotFoundError:
            print(f"Warning: Layout file '{layout_filename}' not found.")

        fig, ax = plt.subplots(figsize=(16, 7))

        # 3. Normalize scanned grid: Any non-zero value becomes a uniform grey (0.7), 0 becomes NaN (transparent)
        uniform_scanned = np.where(scanned_grid != 0, 0.7, np.nan)
        ax.imshow(
            uniform_scanned,
            cmap="Greys",
            vmin=0,
            vmax=1,
            origin="upper",
            alpha=0.4,
        )

        # 4. Create and overlay Agent Frontier Grid (RGBA image for precise cell coloring)
        color_names = ["red", "blue", "green", "purple"]
        agent_rgba_colors = [
            mcolors.to_rgba(c, alpha=0.7) for c in color_names
        ]  # slightly transparent colored cells

        frontier_overlay = np.zeros((grid_height, grid_width, 4))
        for i, agent in enumerate(agents):
            rgba = agent_rgba_colors[i % len(color_names)]
            frontier_pts = agent.get("frontier", [])
            for pt in frontier_pts:
                fx, fy = pt[0], pt[1]
            if 0 <= fy < grid_height and 0 <= fx < grid_width:
                frontier_overlay[fy, fx] = rgba

        ax.imshow(frontier_overlay, origin="upper")

        # 5. Plot Ground-Truth Walls (Line segments)
        for wall in layout_walls:
            start = wall["start"]
            end = wall["end"]
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color="black",
                linewidth=2,
                alpha=0.8,
                label="Walls" if wall["id"] == 1 else "",
            )

        # 6. Plot Doors as Gaps (White cutouts with borders over the walls)
        if layout_doors:
            doors_arr = np.array(layout_doors)
            ax.scatter(
                doors_arr[:, 0],
                doors_arr[:, 1],
                c="white",
                marker="s",
                s=25,
                alpha=1.0,
                edgecolors="black",
                linewidths=0.5,
                label="Doors (Gaps)",
            )

        # 7. Plot Fixed Environment Targets (from layout file)
        # for t in layout_targets:
        #     pos = t["position"]
        #     ax.scatter(
        #         pos[0],
        #         pos[1],
        #         c="gold",
        #         marker="*",
        #         s=140,
        #         edgecolors="black",
        #         label="Global Targets" if t["id"] == 1 else "",
        #     )
        #     ax.text(
        #         pos[0] + 1,
        #         pos[1] + 1,
        #         f"T{t['id']}",
        #         color="darkorange",
        #         fontsize=9,
        #         weight="bold",
        #     )

        # 8. Plot agents, paths, and dynamic targets (ID - 4)
        for i, agent in enumerate(agents):
            color = color_names[i % len(color_names)]
            raw_id = agent["agent_id"]
            display_id = raw_id - 4

            pos = agent["position"]
            dest = agent["destination"]
            path = agent["planned_path"]
            frontier_pts = agent.get("frontier", [])

            # Add a dummy scatter element solely for clean legend tracking of the agent's frontier cells
            if frontier_pts:
                ax.scatter(
                    [],
                    [],
                    c=color,
                    marker="s",
                    s=50,
                    label=f"Agent {display_id} Frontier",
                )

            agent_target_pos = None
            if path:
                path_arr = np.array(path)
                ax.plot(
                    path_arr[:, 0],
                    path_arr[:, 1],
                    color=color,
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.8,
                    label=f"Agent {display_id} Path",
                )
                agent_target_pos = path[-1]
            elif dest and len(dest) == 2:
                agent_target_pos = dest

            # Agent Position
            ax.scatter(
                pos[0],
                pos[1],
                c=color,
                marker="o",
                s=80,
                edgecolors="black",
                label=f"Agent {display_id} Pos",
            )
            ax.text(
                pos[0] + 1,
                pos[1] - 1,
                f"A{display_id}",
                color=color,
                fontsize=10,
                weight="bold",
            )

            # Current Agent Path Destination (marked with X)
            if agent_target_pos is not None:
                ax.scatter(
                    agent_target_pos[0],
                    agent_target_pos[1],
                    c=color,
                    marker="X",
                    s=100,
                    edgecolors="black",
                )

        # Set limits and invert Y-axis so [0,0] is precisely at the top-left corner
        ax.set_xlim(0, grid_width)
        ax.set_ylim(grid_height, 0)

        ax.set_title(
            f"Multi-Agent Exploration State & Frontier Grid Overlay (Step: {step})"
        )
        ax.set_xlabel("X Coordinate (Grid Units)")
        ax.set_ylabel("Y Coordinate (Grid Units)")

        # Clean up legend clutter
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(
            by_label.values(),
            by_label.keys(),
            loc="upper right",
            bbox_to_anchor=(1.18, 1),
        )

        plt.grid(True, linestyle=":", alpha=0.3)
        plt.tight_layout()
        output_filename = f"{algorithm_name}_halfway_target.png"
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
                              

    def halfway_targets(
        self,
        log_filename,
        algorithm_name,
        layout_filename="AreaLayout.JSON",  # Adjust filename if yours is named differently
    ):
        # 1. Load the simulation log snapshot
        try:
            with open(log_filename, "r") as f:
                lines = f.readlines()
                if not lines:
                    print("Log file is empty!")
                    return
            data = json.loads(lines[-1])
        except FileNotFoundError:
            print(f"Error: Log file '{log_filename}' not found.")
            return

        step = data["step"]
        agents = data["agents"]
        scanned_grid = np.array(data["scanned_area"])
        grid_height, grid_width = scanned_grid.shape

        # 2. Load the area layout JSON for ground-truth environment details
        layout_walls = []
        layout_doors = []
        try:
            with open(layout_filename, "r") as f:
                layout_data = json.load(f)
                layout_walls = layout_data.get("walls", [])
                layout_doors = layout_data.get("doors", [])
        except FileNotFoundError:
            print(f"Warning: Layout file '{layout_filename}' not found.")

        fig, ax = plt.subplots(figsize=(16, 7))

        # 3. Plot the scanned grid area with origin='upper' (Top-Left [0,0])
        uniform_scanned = np.where(scanned_grid != 0, 0.7, np.nan)
        ax.imshow(
            uniform_scanned,
            cmap="Greys",
            vmin=0,
            vmax=1,
            origin="upper",
            alpha=0.4,
        )

        # 4. Plot Ground-Truth Walls (Line segments)
        for wall in layout_walls:
            start = wall["start"]
            end = wall["end"]
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color="black",
                linewidth=2,
                alpha=0.8,
                label="Walls" if wall["id"] == 1 else "",
            )

        # 5. Plot Doors as Gaps (White cutouts with borders over the walls)
        if layout_doors:
            doors_arr = np.array(layout_doors)
            ax.scatter(
                doors_arr[:, 0],
                doors_arr[:, 1],
                c="white",
                marker="s",
                s=25,
                alpha=1.0,
                edgecolors="black",
                linewidths=0.5,
                label="Doors (Gaps)",
            )

        colors = ["red", "blue", "green", "purple"]

        # 7. Plot agents, paths, and their calculated dynamic targets (ID - 4)
        for i, agent in enumerate(agents):
            color = colors[i % len(colors)]
            raw_id = agent["agent_id"]
            display_id = raw_id - 4

            pos = agent["position"]
            dest = agent["destination"]
            path = agent["planned_path"]

            agent_target_pos = None
            if path:
                path_arr = np.array(path)
                ax.plot(
                    path_arr[:, 0],
                    path_arr[:, 1],
                    color=color,
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.8,
                    label=f"Agent {display_id} Path",
                )
                agent_target_pos = path[-1]
            elif dest and len(dest) == 2:
                agent_target_pos = dest

            # Agent Position
            ax.scatter(
                pos[0],
                pos[1],
                c=color,
                marker="o",
                s=80,
                edgecolors="black",
                label=f"Agent {display_id} Pos",
            )
            ax.text(
                pos[0] + 1,
                pos[1] - 1,
                f"A{display_id}",
                color=color,
                fontsize=10,
                weight="bold",
            )

            # Current Agent Path Destination (marked with X)
            if agent_target_pos is not None:
                ax.scatter(
                    agent_target_pos[0],
                    agent_target_pos[1],
                    c=color,
                    marker="X",
                    s=100,
                    edgecolors="black",
                )

        # Set limits and invert Y-axis so [0,0] is precisely at the top-left corner
        ax.set_xlim(0, grid_width)
        ax.set_ylim(
            grid_height, 0
        )  # Inverted limits make Y=0 at the top and max height at the bottom

        ax.set_title(
            f"{algorithm_name} Halfway Targets"
        )
        ax.set_xlabel("X Coordinate (Grid Units)")
        ax.set_ylabel("Y Coordinate (Grid Units)")

        # Clean up legend clutter
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(
            by_label.values(),
            by_label.keys(),
            loc="upper right",
            bbox_to_anchor=(1.15, 1),
        )

        plt.grid(True, linestyle=":", alpha=0.3)
        plt.tight_layout()
        output_filename = f"{algorithm_name}_halfway_target.png"
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
                        
