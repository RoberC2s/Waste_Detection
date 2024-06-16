import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load the CSV file into a DataFrame
file_path = 'results_to_compare.csv'  # Replace with your actual file path
df = pd.read_csv(file_path)

# Step 2: Convert the 'timestamp' column to datetime format
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Step 3: Get unique main object IDs without the fractional part
main_ids = sorted(set(int(id) for id in df['object ID'].unique() if id.is_integer()))

# Step 4: Define true values
true_x = 20
true_y = 25
true_z = -6.5
true_spd = 0
# Step 5: Create subplots for x, y, z positions, and speeds over time for each pair of object IDs
for main_id in main_ids:
    # Define the IDs to plot
    obj_ids = [main_id, main_id + 0.1, main_id + 0.2, main_id + 0.3]
    
    # Create a figure with six subplots in a 2x3 grid
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    
    for obj_id in obj_ids:
        # Extract data for the current ID
        obj_data = df[df['object ID'] == obj_id]
        
        # Calculate RMSE for each parameter
        def calculate_rmse(data, true_value, column):
            return np.sqrt(((data[column] - true_value) ** 2).mean())
        
        # Calculate and print RMSE values
        rmse_x = calculate_rmse(obj_data, true_x, 'x')
        rmse_y = calculate_rmse(obj_data, true_y, 'y')
        rmse_z = calculate_rmse(obj_data, true_z, 'z')
        rmse_spd_X = calculate_rmse(obj_data, true_spd, 'spd_X')
        rmse_spd_Y = calculate_rmse(obj_data, true_spd, 'spd_Y')
        rmse_spd_Z = calculate_rmse(obj_data, true_spd, 'spd_Z')
        
        print(f'Object ID {obj_id}:')
        print(f'  RMSE X Position: {rmse_x:.2f}')
        print(f'  RMSE Y Position: {rmse_y:.2f}')
        print(f'  RMSE Z Position: {rmse_z:.2f}')
        print(f'  RMSE Speed X: {rmse_spd_X:.2f}')
        print(f'  RMSE Speed Y: {rmse_spd_Y:.2f}')
        print(f'  RMSE Speed Z: {rmse_spd_Z:.2f}')
        print()
        
        # Plot x position over time for the current ID
        axs[0, 0].plot(obj_data['timestamp'], obj_data['x'], label=f'Object {obj_id}')
        
        # Plot x speed over time for the current ID
        axs[1, 0].plot(obj_data['timestamp'], obj_data['spd_X'], label=f'Object {obj_id}')
        
        # Plot y position over time for the current ID
        axs[0, 1].plot(obj_data['timestamp'], obj_data['y'], label=f'Object {obj_id}')
        
        # Plot y speed over time for the current ID
        axs[1, 1].plot(obj_data['timestamp'], obj_data['spd_Y'], label=f'Object {obj_id}')
        
        # Plot z position over time for the current ID
        axs[0, 2].plot(obj_data['timestamp'], obj_data['z'], label=f'Object {obj_id}')
        
        # Plot z speed over time for the current ID
        axs[1, 2].plot(obj_data['timestamp'], obj_data['spd_Z'], label=f'Object {obj_id}')
    
    # Set titles and labels for each subplot
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].set_ylabel('X Position')
    axs[0, 0].set_title(f'X Position Over Time for Object {main_id}')
    axs[0, 0].grid(True)
    axs[0, 0].legend()
    
    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel('Speed X')
    axs[1, 0].set_title(f'Speed X Over Time for Object {main_id}')
    axs[1, 0].grid(True)
    axs[1, 0].legend()
    
    axs[0, 1].set_xlabel('Time')
    axs[0, 1].set_ylabel('Y Position')
    axs[0, 1].set_title(f'Y Position Over Time for Object {main_id}')
    axs[0, 1].grid(True)
    axs[0, 1].legend()
    
    axs[1, 1].set_xlabel('Time')
    axs[1, 1].set_ylabel('Speed Y')
    axs[1, 1].set_title(f'Speed Y Over Time for Object {main_id}')
    axs[1, 1].grid(True)
    axs[1, 1].legend()
    
    axs[0, 2].set_xlabel('Time')
    axs[0, 2].set_ylabel('Z Position')
    axs[0, 2].set_title(f'Z Position Over Time for Object {main_id}')
    axs[0, 2].grid(True)
    axs[0, 2].legend()
    
    axs[1, 2].set_xlabel('Time')
    axs[1, 2].set_ylabel('Speed Z')
    axs[1, 2].set_title(f'Speed Z Over Time for Object {main_id}')
    axs[1, 2].grid(True)
    axs[1, 2].legend()
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()
