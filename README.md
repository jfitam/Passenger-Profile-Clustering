# Customer Segmentation for Train Services

This project performs **customer segmentation using K-Means clustering** based on travel behavior and purchase patterns in a railway service.

The objective is to identify groups of customers with similar characteristics in order to better understand demand patterns, customer profiles, and potential business strategies.

## Overview

Customers are clustered according to multiple behavioral features derived from historical ticket purchases and travel records, including:

- travel frequency
- ticket price behavior
- booking anticipation
- travel time preferences
- route usage
- class of service (economy / business)
- purchase channel (web, app, ticket machines)
- seasonality (Ramadan, Hajj, regular periods)
- demographic attributes

Each resulting cluster represents a **distinct type of passenger behavior**.

## Methodology

### 1. Feature Engineering

Customer-level features are aggregated from raw transaction data. Examples include:

- `num_travels` — total number of trips
- `average_price` — mean ticket price
- `price_stddev` — price variability
- `avg_advance_days` — booking anticipation
- `avg_group_size` — average train occupancy for the traveled service
- `unique_routes` — number of different routes used

Percent features capture travel patterns such as:

- route combinations (`travels_mak_mad`, `travels_jed_mad`, etc.)
- time-of-day preferences (`travels_morning`, `travels_evening`, etc.)
- travel days (`travels_weekday`, `travels_fri`, etc.)
- class usage (`travels_economy`, `travels_business`)
- purchase channels (`purchases_web`, `purchases_app`, etc.)

Seasonal indicators are also included:

- `travels_ramadan`
- `travels_hajj`
- `travels_no_peak_season`

### 2. Clustering

Customers are segmented using the **K-Means clustering algorithm** applied to the standardized feature space.

The number of clusters is selected initally based on exploratory analysis, using the elbow method. The result gathered, after cross checked with other teams, was deemed insufficient, so the number of cluster was selected combining the total distance to the center and their expertise.

### 3. Cluster Profiling

After clustering, each group is analyzed by computing descriptive statistics for the underlying features.

Typical output includes:

- count of customers per cluster
- mean and distribution of behavioral features
- most common nationalities within the cluster

Example summary statistics per cluster:

| Feature | Mean | Description |
|-|-|-|
| num_travels | 1.03 | average number of trips |
| average_price | 195.77 | mean ticket price |
| avg_advance_days | 40.1 | days booked in advance |
| travels_weekday | 0.58 | proportion of weekday trips |
| travels_hajj | 0.76 | share of trips during Hajj season |

## Cluster Interpretation

Clusters typically represent passenger profiles such as:

- **Occasional travelers**
- **Early planners**
- **Seasonal pilgrims**
- **Route-specific commuters**

Interpretation is supported by:

- aggregated feature statistics
- dominant routes and travel times
- top passenger nationalities

## Outputs

The project produces:

- cluster assignment for each customer
- descriptive statistics for each cluster
- nationality distribution per cluster
- summary tables for behavioral features

These outputs can be used for:

- demand forecasting
- targeted marketing
- pricing strategy
- service planning

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib / Seaborn

## Future Improvements

Possible extensions include:

- dimensionality reduction (PCA, UMAP) for visualization
- cluster stability analysis
- integration with downstream demand models
