### 1. Main Spatiotemporal Traffic Matrix: `p01_done.npy`

* **About this file:** This file contains traffic states captured by 16,972 roadside inductive loop detectors in California across 31 consecutive days.
  
* **Meanings of the 3 features in the last dimension `[ , , 3]`:**
* **Dimension 0 (Feature 0): Volume** — The total number of vehicles passing through the detector within a 5-minute interval.
* **Dimension 1 (Feature 1): Occupancy** — Ranging between `[0, 1]`, this represents the percentage of time the lane is occupied by vehicles, reflecting the density of traffic congestion.
* **Dimension 2 (Feature 2): Speed** — The average traveling speed of vehicles passing through the section (measured in miles per hour, mph).

### 2. Spatial Reference: `XTraffic/node_order.npy`

* **About this file:** A 1-dimensional array that sequentially stores the actual official IDs of the 16,972 sensors (e.g., `3038101`).
* **Meaning of Fields/Indices:**
* The **Index** of the array corresponds to the `Node_Idx` (the second dimension) of the `p01_done.npy` matrix.
* The **Value** of the array represents the actual `station_id`.

### 3. Spatially Aligned Incident Table: `match_incidents.csv`

* **About this file:** A traffic anomaly and incident log that has been spatially filtered and precisely aligned to the postmile locations of nearby sensors.
* **Meanings of Core Fields:**
* `station_id`: The plain-text ID of the sensor where the incident occurred.
* `incident_id`: The official, unique tracking ID of the traffic event (used for cross-referencing the original incident database to query root causes, number of blocked lanes, and event duration).
* `dt`: The exact timestamp of the incident occurrence (e.g., `06/19/2023 00:20:00`).
* `dis`: The physical distance between the incident location and the sensor (`0.0` means the event occurred directly over the sensor).


### 4. Core Experimental Sample Set: `traffident_counterfactual_samples.pkl`

* **About this file:** It is structured as a list, where each element is a dictionary containing the following fields:
* `incident_id` / `station_id` / `node_idx`: Association identifiers linking the incident to its corresponding sensor and matrix position.
* `event_time_raw`: The original, exact time of the incident occurrence.
* `event_time_aligned`: The standardized timestamp after rounding down the incident time to the nearest 5-minute interval to match the main traffic table.
* `factual_traffic` / `factual_speed`: Shaped as `(48,)`. The **factual (with incident)** traffic volume and speed time-series recorded from 1 hour before to 3 hours after the incident.
* `counterfactual_traffic` / `counterfactual_speed`: Shaped as `(48,)`. The normal baseline traffic volume and speed time-series from the **same time slot of the previous week** when no incident occurred (serving as the counterfactual control group).
* 