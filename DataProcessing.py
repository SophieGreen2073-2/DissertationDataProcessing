import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DataProcessing:
    def __init__(self):
        print("Process Data")

        # self.process_time_data()
        self.process_overlap_data()

    def process_overlap_data(self):
        csv_file_path = '/home/student36/Dissertation/AbstractModel/DissertationAbtractModel/dissertation_redundancy_record.csv'

        # Fixed Grid Dimensions
        HEIGHT = 55
        WIDTH = 195

        # Read CSV without a fixed column count (handles varying row lengths)
        with open(csv_file_path, 'r') as f:
            lines = f.readlines()

        # Loop through each run/row in the file
        for row_idx, line in enumerate(lines):
            # Parse line into raw values
            row = line.strip().split(',')
            if not row or row[0] == '':
                continue
                
            # 1. Read num_uavs dynamically from index 0
            num_uavs = int(row[0])
            
            # 2. Compute exact array length for THIS specific row
            array_length = HEIGHT * WIDTH * num_uavs
            
            # 3. Slice and reshape the 3D array dynamically
            array_data = np.array(row[1 : 1 + array_length], dtype=float)
            overlap_area = array_data.reshape((HEIGHT, WIDTH, num_uavs))
            
            # 4. Extract uav_params from the remaining elements at the end
            uav_params_raw = row[1 + array_length :]
            
            # Extract parameters (adjust indices if your JSON structure differs slightly)
            # [StartPosition, ReleaseDelay, TopSpeed, DangerSpeed, StartSpeed, LIDARDistance, BatteryLife, Acceleration, WallDangerZone, ChargeTime]
            top_speed = uav_params_raw[2]
            lidar_dist = int(float(uav_params_raw[5]))
            battery_life = int(float(uav_params_raw[6]))
            accel = uav_params_raw[7]
            
            # ==========================================
            # REDUNDANCY METRICS
            # ==========================================
            
            total_scans = np.sum(overlap_area, axis=2)
            unique_uavs_per_cell = np.sum((overlap_area > 0).astype(int), axis=2)
            
            mapped_cells = np.count_nonzero(total_scans)
            cross_uav_cells = np.count_nonzero(unique_uavs_per_cell > 1)
            
            global_redundancy_ratio = np.sum(total_scans) / mapped_cells if mapped_cells > 0 else 0
            cross_uav_percent = (cross_uav_cells / mapped_cells) * 100 if mapped_cells > 0 else 0
            
            # ==========================================
            # PLOTTING & DYNAMIC SAVING
            # ==========================================
            
            fig, axes = plt.subplots(2, 1, figsize=(12, 7))
            
            title_text = (
                f"Scan Redundancy Analysis ({num_uavs} UAVs)\n"
                f"(Battery Life: {battery_life}s, LIDAR: {lidar_dist}m, Speed: {top_speed}m/s, Accel: {accel}m/s²)\n"
                f"Global Redundancy Ratio: {global_redundancy_ratio:.2f}x | Shared Coverage: {cross_uav_percent:.1f}%"
            )
            fig.suptitle(title_text, fontsize=11, fontweight='bold', y=0.98)
            
            # Subplot 1: Total Scan Heatmap
            im1 = axes[0].imshow(total_scans, cmap='YlOrRd', aspect='equal')
            axes[0].set_title("Total Scan Count Density per Cell", fontsize=10)
            fig.colorbar(im1, ax=axes[0], label='Scans')
            
            # Subplot 2: Cross-UAV Overlap
            im2 = axes[1].imshow(unique_uavs_per_cell, cmap='viridis', aspect='equal')
            axes[1].set_title("Cross-UAV Overlap (Number of Unique UAVs per Cell)", fontsize=10)
            fig.colorbar(im2, ax=axes[1], label='Unique UAVs')
            
            plt.tight_layout(rect=[0, 0, 1, 0.93])
            
            # Save chart uniquely per run
            output_filename = f"redundancy_run{row_idx+1}_uavs{num_uavs}_bat{battery_life}_lidar{lidar_dist}.png"
            plt.savefig(output_filename, dpi=300)
            plt.close()  # Close plot to free memory during loop
            
            print(f"Processed Run {row_idx+1} ({num_uavs} UAVs) -> Saved as: {output_filename}")

    def process_time_data(self):
        # 1. Load the CSV file directly
        csv_file_path = '/home/student36/Dissertation/AbstractModel/DissertationAbtractModel/dissertation_time_record.csv'

        # If your CSV doesn't have headers, pass header=None and assign names:
        # names=['NumUAVS', 'TimeElapsed', 'StartPosition', 'ReleaseDelay', 'TopSpeed', 
        #        'DangerSpeed', 'StartSpeed', 'LIDARDistance', 'BatteryLife', 'Acceleration', 
        #        'WallDangerZone', 'ChargeTime']
        df = pd.read_csv(csv_file_path)

        # Ensure proper numeric data types
        df['NumUAVS'] = df['NumUAVS'].astype(int)
        df['TimeElapsed'] = df['TimeElapsed'].astype(float)

        # Extract constant parameter values from the first run for the title and filename
        battery_life = int(df['BatteryLife'].iloc[0])
        lidar_dist = int(df['LIDARDistance'].iloc[0])
        top_speed = df['TopSpeed'].iloc[0]
        accel = df['Acceleration'].iloc[0]

        # 2. Configure the plot
        plt.figure(figsize=(8, 5))
        plt.plot(df['NumUAVS'], df['TimeElapsed'], marker='o', linewidth=2, color='#1f77b4', label='Simulation Time')

        title_text = (
            f"Time Elapsed vs. Number of UAVs\n"
            f"(Battery Life: {battery_life}s, LIDAR: {lidar_dist}m, Speed: {top_speed}m/s, Accel: {accel}m/s²)"
        )
        plt.title(title_text, fontsize=11, fontweight='bold', pad=12)
        plt.xlabel("Number of UAVs", fontsize=10)
        plt.ylabel("Time Elapsed (seconds)", fontsize=10)
        plt.xticks(df['NumUAVS'])
        plt.grid(True, linestyle='--', alpha=0.6)

        # Annotate each point with its time elapsed
        for x, y in zip(df['NumUAVS'], df['TimeElapsed']):
            plt.annotate(f"{y:.1f}s", (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)

        plt.tight_layout()

        # 3. Save the plot using the dynamic parameter filename
        output_filename = f"sim_results_bat{battery_life}_lidar{lidar_dist}_speed{top_speed}.png"
        plt.savefig(output_filename, dpi=300)
        plt.show()

        print(f"Graph successfully saved as: {output_filename}")