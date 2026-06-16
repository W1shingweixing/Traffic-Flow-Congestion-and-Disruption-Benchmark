import numpy as np
import pandas as pd
import pickle

# # Load accident list (CSV)
df_incidents = pd.read_csv("match_incidents.csv")

# Load the traffic NumPy main table ( [Time, Node, Feature])
traffic_matrix = np.load("XTraffic\p03_done.npy") 

# Get the total number of time steps in the traffic matrix (i.e., the length of the first dimension of the matrix)
total_time_steps = traffic_matrix.shape[0]

station_array = np.load("XTraffic/node_order.npy") 
station_to_idx = {int(sid): idx for idx, sid in enumerate(station_array)}

# Set the start time for the data
START_DATETIME = "2023-03-01 00:00:00" 
DATA_FREQ = "5min"

# Automatically generate a set of continuous time series in memory
timestamps = pd.date_range(start=START_DATETIME, periods=total_time_steps, freq='5min')

# Create a dictionary of [timestamp string -> matrix row number]
time_to_idx = {ts.strftime('%Y-%m-%d %H:%M:%S'): idx for idx, ts in enumerate(timestamps)}


# matching
all_event_samples = []

# Define the time window: 5 minutes per step, 12 steps for 1 hour ago, 36 steps for 3 hours later
PRE_STEPS = 12  
POST_STEPS = 36 
WEEK_STEPS = 2016

for index, row in df_incidents.iterrows():
    sid = int(row['station_id'])
    dt_event = pd.to_datetime(row['dt'])
    
    dt_event_rounded = dt_event.floor(DATA_FREQ)
    dt_str = dt_event_rounded.strftime('%Y-%m-%d %H:%M:%S')
    
    # Convert numeric index
    node_idx = station_to_idx.get(sid)
    time_idx = time_to_idx.get(dt_str)
    
    if node_idx is None or time_idx is None:
        continue
        
    # Calculate the starting and ending time row numbers of the slice
    start_time_idx = time_idx - PRE_STEPS
    end_time_idx = time_idx + POST_STEPS
    
    # Border
    if start_time_idx < 0 or end_time_idx > traffic_matrix.shape[0]:
        continue
        
    # extract 4 hours of data from the NumPy matrix
    # traffic_matrix[time range, corresponding nodes, 0 (assuming 0 is the traffic feature)]
    factual_traffic = traffic_matrix[start_time_idx:end_time_idx, node_idx, 0]
    factual_speed   = traffic_matrix[start_time_idx:end_time_idx, node_idx, 1] 
    
    # Looking for data from the week before
    cf_time_idx = time_idx - WEEK_STEPS
    start_cf_idx = cf_time_idx - PRE_STEPS
    end_cf_idx = cf_time_idx + POST_STEPS
    
    if start_cf_idx >= 0:
        cf_traffic = traffic_matrix[start_cf_idx:end_cf_idx, node_idx, 0]
        cf_speed   = traffic_matrix[start_cf_idx:end_cf_idx, node_idx, 1]
    else:
        cf_traffic = None
        cf_speed = None
        
    # Save results
    all_event_samples.append({
        'incident_id': row['incident_id'],
        'station_id': sid,
        'node_idx': node_idx,
        'event_time_raw': row['dt'],        # Record the original accident time
        'event_time_aligned': dt_str,       
        'factual_traffic': factual_traffic.copy(), 
        'factual_speed': factual_speed.copy(),
        'counterfactual_traffic': cf_traffic.copy() if cf_traffic is not None else None,
        'counterfactual_speed': cf_speed.copy() if cf_speed is not None else None
    })

print(f"Successfully extracted {len(all_event_samples)} event sample arrays")
with open("traffident_counterfactual_samples_mar.pkl", "wb") as f:
    pickle.dump(all_event_samples, f)
print("The sample has been successfully saved.")