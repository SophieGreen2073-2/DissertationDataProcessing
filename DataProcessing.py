import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.colors import BoundaryNorm
import os
import json

# Need some more stuff for path planning, paths taken etc.
# Will add logs soon to abstract then I can put this in here

class DataProcessing:
    def __init__(self):
        print("Process Data")

        # self.process_time_data()
        self.process_overlap_data()
        

    # Okay this needs changing
    # Need to have two graphs made, one for each sim and line/bar for the overlap across them all
    # Easier to copy sim config across each time than try to pull out of csv
    # Need to change the title to be more self explanatory
    def process_overlap_data(self):
        csv_start_filepath = '/home/student36/Dissertation/AbstractModel/DissertationAbtractModel/NewSavedData/dissertation_redundancy_record_'
        algorithm_names = ["Utility", "ClosestFrontier", "WaveFront"]
        agent_names = ["DJI", "Elios"]
        comms = ["no_comms", "comms"]
        num_uavs = [1, 2, 3, 4, 5]

        # ==========================================
        # LOAD LAYOUT FROM AreaLayout.JSON
        # ==========================================
        current_dir = os.path.dirname(__file__) if '__file__' in locals() else '.'
        json_path = os.path.join(current_dir, 'AreaLayout.JSON')
        
        if not os.path.exists(json_path):
            json_path = '/home/student36/Dissertation/AbstractModel/DissertationAbtractModel/AreaLayout.JSON'

        with open(json_path) as f:
            layout_data = json.load(f)

        HEIGHT = int(layout_data["fullarea"]["height"])
        WIDTH = int(layout_data["fullarea"]["width"])

        for algorithm in algorithm_names:
            for agent in agent_names:
                for comm in comms:
                    x_num = []
                    y_redundancy = []
                    for num in num_uavs:
                        csv_file_path = csv_start_filepath + f"{algorithm}_{comm}_{agent}_{str(num)}.csv"
                                                
                        # Read CSV without a fixed column count (handles varying row lengths)
                        with open(csv_file_path, 'r') as f:
                            lines = f.readlines()

                        # Loop through each run/row in the file
                        for row_idx, line in enumerate(lines):
                            row = line.strip().split(',')
                            if not row or row[0] == '':
                                continue
                                
                            # num_uavs = int(row[0])
                            array_length = HEIGHT * WIDTH * num
                            array_data = np.array(row[1 : 1 + array_length], dtype=float)
                            overlap_area = array_data.reshape((HEIGHT, WIDTH, num))
                            
                            # uav_params_raw = row[1 + array_length :]
                            
                            # top_speed = uav_params_raw[2]
                            # lidar_dist = int(float(uav_params_raw[5]))
                            # battery_life = int(float(uav_params_raw[6]))
                            # accel = uav_params_raw[7]
                            
                            # ==========================================
                            # REDUNDANCY METRICS
                            # ==========================================
                            total_scans = np.sum(overlap_area, axis=2)
                            unique_uavs_per_cell = np.sum((overlap_area > 0).astype(int), axis=2)
                            
                            mapped_cells = np.count_nonzero(total_scans)
                            cross_uav_cells = np.count_nonzero(unique_uavs_per_cell > 1)
                            
                            global_redundancy_ratio = np.sum(total_scans) / mapped_cells if mapped_cells > 0 else 0
                            cross_uav_percent = (cross_uav_cells / mapped_cells) * 100 if mapped_cells > 0 else 0

                            x_num.append(num)
                            y_redundancy.append(cross_uav_percent)
                            
                            # ==========================================
                            # PLOTTING & DYNAMIC SAVING (Single Plot)
                            # ==========================================
                            fig, ax = plt.subplots(figsize=(14, 5))

                            comm_string = "Comms Enabled" if comm == "comms" else "Comms Not Enabled"
                            
                            title_text = (
                                f"Cross-UAV Overlap Analysis ({num} UAVs)\n"
                                f"(Algorithm: {algorithm}, Drone: {agent}, comms: {comm_string})\n"
                                f"Shared Coverage: {cross_uav_percent:.1f}%"
                                # f"Global Redundancy Ratio: {global_redundancy_ratio:.2f}x | Shared Coverage: {cross_uav_percent:.1f}%"
                            )
                            ax.set_title(title_text, fontsize=10, fontweight='bold', pad=12)
                            
                            # ==========================================
                            # DISCRETE COLOR MAPPING
                            # ==========================================
                            max_uavs_in_grid = max(int(np.max(unique_uavs_per_cell)), num)
                            
                            cmap = plt.get_cmap('YlOrRd', max_uavs_in_grid + 1)
                            bounds = np.arange(-0.5, max_uavs_in_grid + 1.5, 1)
                            norm = BoundaryNorm(bounds, cmap.N)
                            
                            im = ax.imshow(unique_uavs_per_cell, cmap=cmap, norm=norm, aspect='equal')
                            
                            # ==========================================
                            # OVERLAY WALLS & DOORS IN BLACK FROM JSON
                            # ==========================================
                            doors_set = set(tuple(d) for d in layout_data["doors"])
                            for wall in layout_data["walls"]:
                                x_start, x_end = min(wall["start"][0], wall["end"][0]), max(wall["start"][0], wall["end"][0])
                                y_start, y_end = min(wall["start"][1], wall["end"][1]), max(wall["start"][1], wall["end"][1])
                                
                                if x_start == x_end:  # Vertical wall
                                    x = x_start
                                    ymin, ymax = y_start, y_end
                                    segment_doors = sorted([y for (dx, y) in doors_set if dx == x and ymin <= y <= ymax])
                                    
                                    current_y = ymin
                                    for dy in segment_doors:
                                        if dy > current_y:
                                            ax.plot([x, x], [current_y, dy], color='black', linewidth=1.5)
                                        current_y = dy + 1
                                    if current_y <= ymax:
                                        ax.plot([x, x], [current_y, ymax], color='black', linewidth=1.5)
                                        
                                elif y_start == y_end:  # Horizontal wall
                                    y = y_start
                                    xmin, xmax = x_start, x_end
                                    segment_doors = sorted([x for (x, dy) in doors_set if dy == y and xmin <= x <= xmax])
                                    
                                    current_x = xmin
                                    for dx in segment_doors:
                                        if dx > current_x:
                                            ax.plot([current_x, dx], [y, y], color='black', linewidth=1.5)
                                        current_x = dx + 1
                                    if current_x <= xmax:
                                        ax.plot([current_x, xmax], [y, y], color='black', linewidth=1.5)

                            # ==========================================
                            # HIGHLIGHT START POSITION [1, 1]
                            # ==========================================
                            ax.scatter([1], [1], color='cyan', marker='X', s=120, zorder=5, edgecolor='black', linewidth=1)

                            # ==========================================
                            # BUILD NORMAL DISCRETE LEGEND
                            # ==========================================
                            legend_handles = []
                            for i in range(max_uavs_in_grid + 1):
                                color = cmap(norm(i))
                                label = f"{i} UAV{'s' if i != 1 else ''}"
                                legend_handles.append(mpatches.Patch(color=color, label=label, edgecolor='gray', linewidth=0.5))

                            start_handle = mlines.Line2D([], [], color='cyan', marker='X', linestyle='None',
                                                        markersize=8, markeredgecolor='black', markeredgewidth=1,
                                                        label='Start Position [1, 1]')
                            legend_handles.append(start_handle)

                            ax.legend(handles=legend_handles, bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8, frameon=True)

                            ax.set_xlim(0, WIDTH)
                            ax.set_ylim(HEIGHT, 0)  # Ensures proper top-down grid alignment matching array bounds
                            ax.set_xlabel("X (Grid Columns)", fontsize=9)
                            ax.set_ylabel("Y (Grid Rows)", fontsize=9)

                            plt.tight_layout()
                            
                            # Save chart uniquely per run (bbox_inches='tight' ensures external legend isn't clipped)
                            output_filename = f"OverlapScanningGraphs/UAV_overlap_grid_{algorithm}_{agent}_{comms}_{num}.png"
                            plt.savefig(output_filename, dpi=300, bbox_inches='tight')
                            # plt.close()
                            # plt.show()
                            
                            # print(f"Processed Run {row_idx+1} ({num_uavs} UAVs) -> Saved as: {output_filename}")

                    # 2. Configure the plot
                    plt.figure(figsize=(8, 5))
                    plt.plot(x_num, y_redundancy, marker='o', linewidth=2, color='#1f77b4', label='Simulation Time')

                    comm_string = "Comms Enabled" if comm == "comms" else "Comms Not Enabled"

                    title_text = (
                        f"Cross-UAV Overlap vs. Number of UAVs Abstract\n"
                        f"(Algorithm: {algorithm}, Drone: {agent}, comm: {comm_string})"
                    )
                    plt.title(title_text, fontsize=11, fontweight='bold', pad=12)
                    plt.xlabel("Number of UAVs", fontsize=10)
                    plt.ylabel("Time Elapsed (minutes)", fontsize=10)
                    plt.xticks(num_uavs)
                    plt.grid(True, linestyle='--', alpha=0.6)

                    # Annotate each point with its time elapsed
                    for x, y in zip(x_num, y_redundancy):
                        plt.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)

                    plt.tight_layout()

                    # 3. Save the plot using the dynamic parameter filename
                    output_filename = f"OverlapScanningGraphs/Cross_UAV_Percent_{algorithm}_{agent}_{comms}.png"
                    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
                    # plt.show()
                        
                        

    def process_time_data(self):
        csv_start_filepath = '/home/student36/Dissertation/AbstractModel/DissertationAbtractModel/NewSavedData/dissertation_time_record'
        algorithm_names = ["Utility", "ClosestFrontier", "WaveFront"]
        agent_names = ["DJI", "Elios"]
        comms = ["no_comms", "comms"]
        num_uavs = [1, 2, 3, 4, 5]

        for algorithm in algorithm_names:
            for agent in agent_names:
                for comm in comms:
                    # data_frames = []
                    # data_frame_params = []
                    x_nums = []
                    y_times = []
                    for num in num_uavs:
                        csv_file_path = csv_start_filepath + f"_{algorithm}_{comm}_{agent}_{str(num)}.csv"
                        df = pd.read_csv(csv_file_path, header=None)
                        seconds = float(df.iloc[0, 1])
                        mins = round(seconds / 60, 1)
                        # data_frame_params.append([mins, num])
                        x_nums.append(num)
                        y_times.append(mins)

                    # 2. Configure the plot
                    plt.figure(figsize=(8, 5))
                    plt.plot(x_nums, y_times, marker='o', linewidth=2, color='#1f77b4', label='Simulation Time')

                    comm_string = "Comms Enabled" if comm == "comms" else "Comms Not Enabled"

                    title_text = (
                        f"Time Elapsed vs. Number of UAVs Abstract\n"
                        f"(Algorithm: {algorithm}, Drone: {agent}, comm: {comm_string})"
                    )
                    plt.title(title_text, fontsize=11, fontweight='bold', pad=12)
                    plt.xlabel("Number of UAVs", fontsize=10)
                    plt.ylabel("Time Elapsed (minutes)", fontsize=10)
                    plt.xticks(num_uavs)
                    plt.grid(True, linestyle='--', alpha=0.6)

                    # Annotate each point with its time elapsed
                    for x, y in zip(x_nums, y_times):
                        plt.annotate(f"{y}m", (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)

                    plt.tight_layout()

                    # 3. Save the plot using the dynamic parameter filename
                    output_filename = f"TimeElapsedGraphs/Time_Elapsed_{algorithm}_{agent}_{comm}.png"
                    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
                    # plt.show()

                    # print(f"Graph successfully saved as: {output_filename}")

    def generate_overlap_visualizations(num_uavs, overlap_area, uav_params):
        """
        Generates:
        1. The main overlap map with the requested metadata title.
        2. An overlap frequency distribution graph.
        """
        
        # 1. Extract required metadata safely from uav_params
        battery_life = uav_params.get("BatteryLife", "Unknown")
        # Determine communications status (either explicitly or from nesting)
        comms_status = "Enabled" if "Communications" in uav_params and uav_params["Communications"] else "Disabled"
        
        # 2. Calculate Overlap Percentage 
        # (Assuming overlap_area contains cell counts or frequency counts where > 1 means redundant coverage)
        total_cells = overlap_area.size
        overlapped_cells = np.sum(overlap_area > 1)
        overlap_percentage = (overlapped_cells / total_cells) * 100 if total_cells > 0 else 0.0

        # --- GRAPH 1: Spatial Overlap Map with Custom Title ---
        plt.figure(figsize=(8, 6))
        plt.imshow(overlap_area, cmap='viridis', origin='lower')
        plt.colorbar(label='Coverage Count per Cell')
        
        # Construct the requested title dynamically
        title_string = (
            f"Redundancy Map | UAVs: {num_uavs} | Battery: {battery_life}s | "
            f"Comms: {comms_status} | Overlap: {overlap_percentage:.2f}%"
        )
        plt.title(title_string, fontsize=10, wrap=True)
        plt.xlabel("X Grid Index")
        plt.ylabel("Y Grid Index")
        plt.tight_layout()
        plt.show()

        # --- GRAPH 2: Overlap Frequency Distribution Graph ---
        plt.figure(figsize=(8, 5))
        # Flatten array to count occurrences of coverage frequencies (0 visits, 1 visit, 2+ visits, etc.)
        unique_vals, counts = np.unique(overlap_area, return_counts=True)
        
        plt.bar(unique_vals, counts, color='teal', edgecolor='black', alpha=0.7)
        plt.title(f"Overlap Frequency Distribution (UAVs: {num_uavs})", fontsize=12)
        plt.xlabel("Number of UAV Visits per Cell (Coverage Frequency)", fontsize=10)
        plt.ylabel("Number of Grid Cells", fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()