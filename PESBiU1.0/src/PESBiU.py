# Assuming a CSV file structure with additional columns for consumed energy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data_path = 'TuesdayLSTM.csv'
data = pd.read_csv(data_path)

# Static capacities for each BBU
C = [25.0, 25.0, 25.0]  # Example capacities

def predict_traffic_level(total_traffic, T_total):
    if total_traffic < 0.3 * T_total:
        return 'L'
    elif 0.3 * T_total <= total_traffic <= 0.7 * T_total:
        return 'M'
    else:
        return 'H'

def select_bb_us_for_sleep(traffic_levels, bbu_traffics):
    sorted_bb_us = sorted(range(len(bbu_traffics)), key=lambda k: bbu_traffics[k])
    if traffic_levels == 'L':
        return sorted_bb_us[:2]  # Select the two BBUs with the lowest traffic
    elif traffic_levels == 'M':
        return sorted_bb_us[:1]  # Select the BBU with the lowest traffic
    else:
        return []
    
    
    
def update_bbu_statuses(bb_us_for_sleep, successful_redistribution):
    statuses = ['active'] * len(C)
    for i, redistributed in zip(bb_us_for_sleep, successful_redistribution):
        if redistributed:
            statuses[i] = 'sleep'
    return statuses

def redistribute_traffic(bb_us_for_sleep, bbu_traffics, C):
    high_thresholds = [0.7 * c for c in C]
    redistributed_traffics = bbu_traffics.copy()  # Make a copy of the original traffic values
    covering_bb_us = [i for i in range(len(C)) if i not in bb_us_for_sleep]
    available_capacities = {i: high_thresholds[i] - redistributed_traffics[i] for i in covering_bb_us}
    covering_sorted_by_capacity = sorted(available_capacities, key=available_capacities.get)
    successful_redistribution = [False] * len(bb_us_for_sleep)

    for idx, sleep_bbu in enumerate(bb_us_for_sleep):
        traffic_to_redistribute = redistributed_traffics[sleep_bbu]
        original_traffic = redistributed_traffics[:]  # Store the original traffic distribution

        for covering_bbu in covering_sorted_by_capacity:
            if traffic_to_redistribute <= 0:
                break
            potential_redistribution = min(traffic_to_redistribute, available_capacities[covering_bbu])
            redistributed_traffics[covering_bbu] += potential_redistribution
            traffic_to_redistribute -= potential_redistribution
            available_capacities[covering_bbu] -= potential_redistribution

        if traffic_to_redistribute > 0:
            # Redistribution failed, revert to original traffic values for this attempt
            print(f"Cannot redistribute traffic from BBU {sleep_bbu} without exceeding high thresholds.")
            redistributed_traffics = original_traffic  # Revert to original traffic distribution
            continue

        redistributed_traffics[sleep_bbu] = 0  # Successfully redistributed
        successful_redistribution[idx] = True

    return redistributed_traffics, successful_redistribution




def calculate_consumed_energy(bbu_statuses, row):
    consumed_energy_values = [row['bbu1_consumedenergy'], row['bbu2_consumedenergy'], row['bbu3_consumedenergy']]
    adjusted_energy_consumption = [
        energy if status == 'active' else 0 
        for status, energy in zip(bbu_statuses, consumed_energy_values)
    ]
    return adjusted_energy_consumption

def calculate_additional_energy(bbu_traffics_before, bbu_traffics_after, consumed_energy, C):
    for i in range(len(C)):
        traffic_change = bbu_traffics_after[i] - bbu_traffics_before[i]
        if traffic_change > 0:
            percent_increase = (traffic_change / C[i]) * 100
            additional_energy = 0.05 * percent_increase
            consumed_energy[i] += additional_energy
    return consumed_energy

# Initialize a list to hold the results
results_list = []



for index, row in data.iterrows():
    T_total = sum(C)
    T_pred = row['total_traffic']
    bbu_traffics_before = [row['bbu1_traffic'], row['bbu2_traffic'], row['bbu3_traffic']]
    traffic_level = predict_traffic_level(T_pred, T_total)
    bb_us_for_sleep = select_bb_us_for_sleep(traffic_level, bbu_traffics_before)
    
    bbu_traffics_after, successful_redistribution = redistribute_traffic(bb_us_for_sleep, bbu_traffics_before, C)
    bbu_statuses = update_bbu_statuses(bb_us_for_sleep, successful_redistribution)
    
    consumed_energy = calculate_consumed_energy(bbu_statuses, row)
    consumed_energy = calculate_additional_energy(bbu_traffics_before, bbu_traffics_after, consumed_energy, C)
    
    result = {
        'time_interval': row['time_interval'],
        'bbu_statuses': bbu_statuses,
        'traffic_after_redistribution': bbu_traffics_after,
        'adjusted_consumed_energy': consumed_energy
    }
    
    results_list.append(result)
    print(result)



# Assuming results_list is already populated as per the previous example
time_intervals = [result['time_interval'] for result in results_list]
bbu_statuses = [[status for status in result['bbu_statuses']] for result in results_list]
traffic_after_redistribution = [result['traffic_after_redistribution'] for result in results_list]
adjusted_consumed_energy = [sum(result['adjusted_consumed_energy']) for result in results_list] # Sum of energy for all BBUs per interval

original_consumed_energy_list = []

for index, row in data.iterrows():
    
    # Calculate original consumed energy for the current row
    original_energy = row['bbu1_consumedenergy'] + row['bbu2_consumedenergy'] + row['bbu3_consumedenergy']
    original_consumed_energy_list.append(original_energy)
    

# Assuming the preparation of data as described previously
time_intervals = [result['time_interval'] for result in results_list]  # Time intervals
total_traffic = [sum(traffic) for traffic in traffic_after_redistribution]  # Total traffic after redistribution
adjusted_consumed_energy = [sum(result['adjusted_consumed_energy']) for result in results_list]  # Adjusted consumed energy


original_consumed_energy = [energy * 0.9 for energy in adjusted_consumed_energy]  

active_count = [sum(1 for status in interval if status == 'active') for interval in bbu_statuses]  # Active BBU counts

fig, ax1 = plt.subplots(figsize=(12, 7))

# Bar plot for active BBU count
ax1.bar(time_intervals, active_count, color='gray', alpha=0.3, label='Active BBUs')
ax1.set_xlabel('Time Interval', fontsize=15)
ax1.set_ylabel('Active BBUs #', color='black', fontsize=15)
ax1.tick_params(axis='y', labelcolor='black')

# Determine the max value of active_count to set the upper limit of y-axis ticks

max_active_count = max(active_count)
ax1.set_yticks(range(0, max_active_count + 1, 1))  # Adjust the step as necessary
# Rotate x-axis labels to be horizontal

plt.xticks(rotation=90)  # Set rotation to 0 for horizontal labels

# Instantiate a second axes that shares the same x-axis
ax2 = ax1.twinx()  

ax2.set_ylabel('Total Traffic [GB] / Consumed Energy [Wh]', fontsize=15)

# Line plot for total traffic
line1, = ax2.plot(time_intervals, total_traffic, color='blue', marker='o', label='Total Traffic', linewidth=2)
# Line plot for adjusted consumed energy
line2, = ax2.plot(time_intervals, adjusted_consumed_energy, color='red', marker='^', label='Adjusted Consumed Energy', linewidth=2)
# Line plot for original consumed energy
line3, = ax2.plot(time_intervals, original_consumed_energy_list, color='green', marker='x', label='Original Consumed Energy', linestyle='--', linewidth=2)


# Create a legend that combines both plots
lines = [line1, line2, line3]
labels = [l.get_label() for l in lines]
ax2.legend(lines, labels, loc='lower right', fontsize=15)

plt.title('BBU Activity, Total Traffic, and Consumed Energy Over Tuesday using LSTM', fontsize=16)
plt.show()



def calculate_energy_savings(original_energy, adjusted_energy):
    
    # Ensure both lists have the same length
    assert len(original_energy) == len(adjusted_energy), "The lists must be of the same length."
    
    # Calculate total original and adjusted energy
    total_original_energy = sum(original_energy)
    total_adjusted_energy = sum(adjusted_energy)
    
    # Calculate total energy saved
    total_energy_saved = total_original_energy - total_adjusted_energy
    
    # Calculate percentage savings
    if total_original_energy > 0:  # Avoid division by zero
        percentage_savings = (total_energy_saved / total_original_energy) * 100
    else:
        percentage_savings = 0  # If no energy was consumed originally, set savings to 0%
    
    return total_energy_saved, percentage_savings

# Assuming original_consumed_energy_list and adjusted_consumed_energy have been correctly calculated
total_energy_saved, percentage_savings = calculate_energy_savings(original_consumed_energy_list, adjusted_consumed_energy)

print(f"Total Energy Saved: {total_energy_saved:.2f} Wh")
print(f"Percentage Savings: {percentage_savings:.2f}%")