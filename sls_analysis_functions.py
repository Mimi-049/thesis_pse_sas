import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import glob
import tkinter as tk
import os
from tkinter import Listbox, MULTIPLE, END, Button, Label, Frame

plt.rcParams["font.family"] = "Times New Roman"

#### HELPER FUNCTIONS ####
def prompt_function_selection():
    print("\nSelect a function to run:")
    print("1. plot_series_from_file")
    print("2. plot_series_from_file_subplots")
    print("3. plot_histogram_for_D32_ranges")
    print("4. compare_samples_by_bar_passes")
    print("5. compare_samples_by_bar_passes_averages")
    print("6. compare_effect_of_D2O_on_D32")
    print("7. compare_effect_of_centrifuging_and_water")
    print("8. filter and plot")
    choice = input("Enter the number of the function you want to run: ")
    return choice

def prompt_for_dates(dates, wpi_values):
    def on_select():
        selected_indices = listbox.curselection()
        selected_dates.extend([sorted_dates[i] for i in selected_indices])
        root.quit()
        root.destroy()

    sorted_dates = sorted(set(pd.to_datetime(dates)))
    root = tk.Tk()
    root.title(f"Select Dates for WPI values: {', '.join(wpi_values)}")

    max_date_length = max(len(str(date)) for date in sorted_dates)
    listbox_width = max_date_length + 2

    frame = Frame(root, padx=10, pady=10)
    frame.pack(padx=10, pady=10)

    label = Label(frame, text=f"Select Dates for WPI values: {', '.join(wpi_values)}", font=("Times New Roman", 12), bg=None, fg='black')
    label.pack(pady=5)

    listbox = Listbox(frame, selectmode=MULTIPLE, width=listbox_width, height=15, font=("Times New Roman", 10), bg='white', fg='black')
    for date in sorted_dates:
        listbox.insert(END, date.strftime('%Y-%m-%d'))
    listbox.pack(pady=5)

    button = Button(frame, text="Select", command=on_select, font=("Times New Roman", 10), bg='white', fg='black')
    button.pack(pady=5)

    root.update_idletasks()
    window_width = root.winfo_width() + 10
    window_height = root.winfo_height()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    position_top = int(screen_height / 2 - window_height / 2)
    position_right = int(screen_width / 2 - window_width / 2)
    root.geometry(f'{window_width}x{window_height}+{position_right}+{position_top}')

    selected_dates = []
    root.mainloop()
    return [date.strftime('%Y-%m-%d') for date in selected_dates]

def create_bar_passes_column(data):
    '''
    Create a new column 'bar_passes' in the dataset by combining the 'bar' and 'passes' columns.
    '''
    data['bar_passes'] = data.apply(lambda row: '{}b{}'.format(row['bar'], row['passes']), axis=1)
    return data

#### MAIN FUNCTIONS ####
def plot_series_from_file(WPI_value):
    '''
    Plot different series from the separated data file.
    '''
    # Read the data from the file
    data = pd.read_csv('Mastersizer/Separated_data/{}.txt'.format(WPI_value), header=0, delimiter=",")
    
    # Extract necessary columns
    date = data['date']
    D32 = data['D32']
    bar = data['bar']
    passes = data['passes']
    indicator = data['indicator'].astype(str) 
    age = data['age']
    water = data['water'] 

    # Sort the data by bar and passes
    data = data.sort_values(by=['bar', 'passes'])

    # Create a new column for bar+passes
    data['bar_passes'] = data.apply(lambda row: '{}b{}'.format(row['bar'], row['passes']), axis=1)

    # Filter the data based on selected dates
    data['date'] = pd.to_datetime(data['date'])
    selected_dates = prompt_for_dates(data['date'], WPI_value)
    data = data[data['date'].dt.strftime('%Y-%m-%d').isin(selected_dates)]

    # Group the data by date and the second indicator value
    grouped = data.groupby(['date', indicator.str[1]])
    
    # Plot the data
    plt.figure(figsize=(8, 5))
    
    for (date, series), group in grouped:
        x = group['bar_passes']
        y = group['D32']
        e = group['error_D32']
        plt.errorbar(x, y, e, marker='o', linestyle='-', label=f'Date: {date}, S{series}')
    
    plt.title(f'D32 for {WPI_value}wt% WPI')
    plt.xlabel('Bar + Passes')
    plt.ylabel(r'$D[3,2]$ [$\mu$m]')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    x_min, x_max = plt.xlim()
    plt.axhline(y=3.5, color='green', linestyle='dashed')
    plt.axhline(y=4.5, color='green', linestyle='dashed')
    plt.fill_between([x_min, x_max], 3.5, 4.5, color='lightgreen')
    plt.xlim(x_min, x_max)
    plt.savefig('Mastersizer/Homogeniser_Settings/Separated_{}wt%WPI.png'.format(WPI_value), dpi = 250)
    plt.show()

def plot_series_from_file_subplots(WPI_values):
    '''
    Plot different series from the separated data files in subplots.
    '''
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, WPI_value in enumerate(WPI_values):
        # Read the data from the file
        data = pd.read_csv('Mastersizer/Separated_data/{}.txt'.format(WPI_value), header=0, delimiter=",")
        
        # Extract necessary columns
        date = data['date']
        D32 = data['D32']
        bar = data['bar']
        passes = data['passes']
        indicator = data['indicator'].astype(str)  
        age = data['age']
        water = data['water'] 

        # Sort the data by bar and passes
        data = data.sort_values(by=['bar', 'passes'])

        # Create a new column for bar+passes
        data['bar_passes'] = data.apply(lambda row: '{}b{}'.format(row['bar'], row['passes']), axis=1)

        # Filter the data based on selected dates
        data['date'] = pd.to_datetime(data['date'])
        selected_dates = prompt_for_dates(data['date'], wpi_values=[WPI_value])
        data = data[data['date'].dt.strftime('%Y-%m-%d').isin(selected_dates)]

        # Group the data by date and the second indicator value
        grouped = data.groupby(['date', indicator.str[1]])
        
        # Plot the data
        ax = axes[i]
        for (date, series), group in grouped:
            x = group['bar_passes']
            y = group['D32']
            e = group['error_D32']
            ax.errorbar(x, y, e, marker='o', linestyle='-', label=f'Date: {date}, S{series}')
        
        ax.set_title(f'D32 for {WPI_value}wt% WPI')
        ax.set_xlabel('Bar + Passes')
        ax.set_ylabel(r'$D[3,2]$ [$\mu$m]')
        ax.set_xticks(range(len(x)))
        ax.set_xticklabels(x, rotation=45, ha='right')
        ax.legend()
        ax.grid(True)

        # Add the green region spanning the entire x-axis
        ax.axhline(y=3.5, color='green', linestyle='dashed')
        ax.axhline(y=4.5, color='green', linestyle='dashed')
        ax.fill_between(range(len(x)), 3.5, 4.5, color='lightgreen', alpha=0.5)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, wspace=0.1)
    plt.savefig('Mastersizer/Homogeniser_Settings/Separated_WPI_values.png', dpi=250)
    plt.show()

def plot_histogram_for_D32_ranges():
    '''
    Plot histograms showing how often the D[3,2] value is between 3.5 and 4.5 um (good)
    and how often it is outside of this range (not good) for specific bar+passes values.
    Save the good and not good sample information to separate files.
    Distinguish between fresh and frozen samples in the histogram.
    '''
    # Define the bar+passes values to consider
    bar_passes_values = ['10b1', '5b1', '5b2', '5b3']

    # Read all separated data files
    all_data = []
    for file in glob.glob('Mastersizer/Separated_data/*.txt'):
        data = pd.read_csv(file, header=0, delimiter=",")
        all_data.append(data)
    all_data = pd.concat(all_data, ignore_index=True)

    # Filter the data based on selected dates
    all_data['date'] = pd.to_datetime(all_data['date'])
    selected_dates = prompt_for_dates(all_data['date'], wpi_values=[])
    all_data = all_data[all_data['date'].dt.strftime('%Y-%m-%d').isin(selected_dates)]

    # Filter the data to include only the specified bar+passes values
    all_data['bar_passes'] = all_data.apply(lambda row: '{}b{}'.format(row['bar'], row['passes']), axis=1)
    filtered_data = all_data[all_data['bar_passes'].isin(bar_passes_values)]

    # Count how often the D[3,2] value is between 3.5 and 4.5 um (good) and outside this range (not good)
    good_data = filtered_data[(filtered_data['D32'] >= 3.5) & (filtered_data['D32'] <= 4.5)]
    not_good_data = filtered_data[(filtered_data['D32'] < 3.5) | (filtered_data['D32'] > 4.5)]

    # Separate counts for fresh and frozen samples
    good_fresh_counts = good_data[good_data['age'] == 'fresh'].groupby('bar_passes').size()
    good_frozen_counts = good_data[good_data['age'] == 'frozen'].groupby('bar_passes').size()
    not_good_fresh_counts = not_good_data[not_good_data['age'] == 'fresh'].groupby('bar_passes').size()
    not_good_frozen_counts = not_good_data[not_good_data['age'] == 'frozen'].groupby('bar_passes').size()

    # Ensure all specified bar+passes values are included in the counts
    good_fresh_counts = good_fresh_counts.reindex(bar_passes_values, fill_value=0)
    good_frozen_counts = good_frozen_counts.reindex(bar_passes_values, fill_value=0)
    not_good_fresh_counts = not_good_fresh_counts.reindex(bar_passes_values, fill_value=0)
    not_good_frozen_counts = not_good_frozen_counts.reindex(bar_passes_values, fill_value=0)
    
    # Save the good and not good sample information to separate files
    good_data.to_csv('Mastersizer/Homogeniser_Settings/Histograms/good_samples.csv', index=False)
    not_good_data.to_csv('Mastersizer/Homogeniser_Settings/Histograms/not_good_samples.csv', index=False)

    # Plot the histograms
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.35
    index = np.arange(len(bar_passes_values))

    bars1 = ax.bar(index, good_fresh_counts, bar_width, label='Good Fresh (3.5 <= D[3,2] <= 4.5)', color='lightgreen')
    bars2 = ax.bar(index, good_frozen_counts, bar_width, bottom=good_fresh_counts, label='Good Frozen (3.5 <= D[3,2] <= 4.5)', color='green')
    bars3 = ax.bar(index + bar_width, not_good_fresh_counts, bar_width, label='Not Good Fresh (D[3,2] < 3.5 or D[3,2] > 4.5)', color='lightcoral')
    bars4 = ax.bar(index + bar_width, not_good_frozen_counts, bar_width, bottom=not_good_fresh_counts, label='Not Good Frozen (D[3,2] < 3.5 or D[3,2] > 4.5)', color='red')

    ax.set_xlabel('Bar + Passes')
    ax.set_ylabel('Counts')
    ax.set_title('Histogram of D[3,2] Values for Specific Bar + Passes Settings')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(bar_passes_values)
    ax.legend()

    # Add counts on top of the bars
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                if bar in bars2 or bar in bars4:
                    bottom_height = bar.get_y()
                    ax.annotate('{}'.format(height),
                                xy=(bar.get_x() + bar.get_width() / 2, bottom_height + height / 2),
                                xytext=(0, 0),  # No offset
                                textcoords="offset points",
                                ha='center', va='center')
                else:
                    ax.annotate('{}'.format(height),
                                xy=(bar.get_x() + bar.get_width() / 2, height / 2),
                                xytext=(0, 0),  # No offset
                                textcoords="offset points",
                                ha='center', va='center')

    plt.tight_layout()
    plt.savefig('Mastersizer/Homogeniser_Settings/Histograms/D32_histogram.png', dpi=250)
    plt.show()

def compare_samples_by_bar_passes(WPI_values):
    '''
    Compare different samples on the same bar+passes settings.
    The x-axis has the WPI_value and the y-axis has the D[3,2] value.
    Use all files in the separated_data folder. The samples need to be selected on date again.
    The first level of separation is whether the samples are fresh or frozen, the second level their bar settings.
    A separate line for 5b1 samples and 5b2 samples (only 5b1 and 5b2 need to be considered).
    Samples of the same age (fresh/frozen) and same bar+passes setting are considered the same series.
    As a third level of separation consider the indicator numbers again, where the first one is the batch,
    the second one the series (separate on this one), and the third one the measurement indicator.
    '''
    # Define the bar+passes values to consider
    bar_passes_values = ['5b1', '5b2']

    # Read all separated data files
    all_data = []
    for file in glob.glob('Mastersizer/Separated_data/*.txt'):
        data = pd.read_csv(file, header=0, delimiter=",")
        all_data.append(data)
    all_data = pd.concat(all_data, ignore_index=True)

    # Filter the data based on selected dates
    all_data['date'] = pd.to_datetime(all_data['date'])
    selected_dates = prompt_for_dates(all_data['date'], wpi_values=[])
    all_data = all_data[all_data['date'].dt.strftime('%Y-%m-%d').isin(selected_dates)]

    # Filter the data to include only the specified bar+passes values
    all_data['bar_passes'] = all_data.apply(lambda row: '{}b{}'.format(row['bar'], row['passes']), axis=1)
    filtered_data = all_data[all_data['bar_passes'].isin(bar_passes_values)]

    # Separate the data by age (fresh/frozen)
    fresh_data = filtered_data[filtered_data['age'] == 'fresh']
    frozen_data = filtered_data[filtered_data['age'] == 'frozen']

    # Plot the data
    fig, axes = plt.subplots(2, 2, figsize=(18, 9))
    axes = axes.flatten()

    plot_titles = ['Fresh 5b1', 'Fresh 5b2', 'Frozen 5b1', 'Frozen 5b2']
    data_sets = [(fresh_data, '5b1'), (fresh_data, '5b2'), (frozen_data, '5b1'), (frozen_data, '5b2')]

    y_limits = [[], []]  # To store y-limits for each row
    
    for ax, (age_data, bar_passes), title in zip(axes, data_sets, plot_titles):
        bar_passes_data = age_data[age_data['bar_passes'] == bar_passes]
        grouped = bar_passes_data.groupby(['indicator'])

        for (indicator, group) in grouped:
            x = group['protein_wt']
            y = group['peak_width']
            e = group['error_peak_width']
            ax.errorbar(x, y, e, marker='o', linestyle='None')
            # ax.scatter(x, y, zorder=3)
            y_limits[0 if 'Fresh' in title else 1].extend(y)

        ax.set_title(f'Peak Width for {title} Samples')
        ax.set_xlabel('WPI wt%')
        ax.set_ylabel(r'$D$ [$\mu$m]')
        ax.set_xticks(WPI_values)
        ax.set_xticklabels(WPI_values)
        ax.grid(True)

    # Set the same y-axis limits for each row
    for i in range(2):
        min_y = min(y_limits[i])
        max_y = max(y_limits[i])
        axes[i * 2].set_ylim(min_y-0.2, max_y+0.2)
        axes[i * 2 + 1].set_ylim(min_y-0.2, max_y+0.2)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.1)
    plt.savefig('Mastersizer/Homogeniser_Settings/Comparison_WPI_values_peak_width.png', dpi=250)
    plt.show()

def compare_samples_by_bar_passes_averages(WPI_values):
    '''
    Compare different samples on the same bar+passes settings.
    The x-axis has the WPI_value and the y-axis has the D[3,2] value.
    Use all files in the separated_data folder. The samples need to be selected on date again.
    The first level of separation is whether the samples are fresh or frozen, the second level their bar settings.
    A separate line for 5b1 samples and 5b2 samples (only 5b1 and 5b2 need to be considered).
    Samples of the same age (fresh/frozen) and same bar+passes setting are considered the same series.
    As a third level of separation consider the indicator numbers again, where the first one is the batch,
    the second one the series (separate on this one), and the third one the measurement indicator.
    '''
    # Define the bar+passes values to consider
    bar_passes_values = ['5b1', '5b2']

    # Read all separated data files
    all_data = []
    for file in glob.glob('Mastersizer/Separated_data/*.txt'):
        data = pd.read_csv(file, header=0, delimiter=",")
        all_data.append(data)
    all_data = pd.concat(all_data, ignore_index=True)
    # print('All data', all_data)

    # Filter the data based on selected dates
    all_data['date'] = pd.to_datetime(all_data['date'])
    selected_dates = prompt_for_dates(all_data['date'], wpi_values=[])
    all_data = all_data[all_data['date'].dt.strftime('%Y-%m-%d').isin(selected_dates)]

    # Filter the data to include only the specified bar+passes values
    all_data['bar_passes'] = all_data.apply(lambda row: '{}b{}'.format(row['bar'], row['passes']), axis=1)
    filtered_data = all_data[all_data['bar_passes'].isin(bar_passes_values)]
    # print('Filtered', filtered_data)

    # Separate the data by age (fresh/frozen)
    fresh_data = filtered_data[filtered_data['age'] == 'fresh']
    frozen_data = filtered_data[filtered_data['age'] == 'frozen']

    # Calculate the average D[3,2] value per WPI value for each dataset
    averages = {'Fresh 5b1': [], 'Fresh 5b2': [], 'Frozen 5b1': [], 'Frozen 5b2': []}
    averages_error = {'Fresh 5b1': [], 'Fresh 5b2': [], 'Frozen 5b1': [], 'Frozen 5b2': []}
    data_sets = [(fresh_data, '5b1', 'Fresh 5b1'), (fresh_data, '5b2', 'Fresh 5b2'), 
                 (frozen_data, '5b1', 'Frozen 5b1'), (frozen_data, '5b2', 'Frozen 5b2')]

    for age_data, bar_passes, label in data_sets:
        bar_passes_data = age_data[age_data['bar_passes'] == bar_passes]
        for wpi_value in WPI_values:
            wpi_data = bar_passes_data[bar_passes_data['protein_wt'] == wpi_value]
            avg_d32 = wpi_data['peak_width'].mean()
            # print('WPI data', wpi_data['D32'])
            avg_d32_err = wpi_data['peak_width'].std()
            # print(len(wpi_data['D32']))
            # avg_d32_err.replace(np.nan, 0, inplace=True)
            if len(wpi_data['peak_width']) < 2:
                avg_d32_err = wpi_data['error_peak_width'].mean()
            print(avg_d32, avg_d32_err)
            averages[label].append(avg_d32)
            averages_error[label].append(avg_d32_err)

    # Plot the averages
    plt.figure(figsize=(10, 6))
    for label, avg_values in averages.items():
        avg_d32_err = averages_error[label]
        plt.errorbar(WPI_values, avg_values, avg_d32_err,  marker='o', linestyle='-', label=label)

    plt.title('Average Peak Width per WPI Value')
    plt.xlabel('WPI Value')
    plt.ylabel(r'Average $D$ [$\mu$m]')
    plt.xticks(WPI_values)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Mastersizer/Homogeniser_Settings/Average_Peak_Width_per_WPI_value.png', dpi=250)
    plt.show()

def compare_effect_of_D2O_on_D32():
    '''
    Compare the effect of D2O on the D[3,2] value for the 5wt% protein samples.
    The x-axis has H2O and D2O, and the y-axis has the D[3,2] value.
    Differentiate the samples based on their bar+passes settings.
    Only include fresh samples.
    '''
    # Read the data from the 5wt% protein file
    data = pd.read_csv('Mastersizer/Separated_data/5,00.txt', header=0, delimiter=",")
    
    # Filter the data based on selected dates
    data['date'] = pd.to_datetime(data['date'])
    selected_dates = prompt_for_dates(data['date'], wpi_values=[])
    data = data[data['date'].dt.strftime('%Y-%m-%d').isin(selected_dates)]

    # Create the bar_passes column
    data = create_bar_passes_column(data)
    
    # Separate the data by water type (H2O and D2O)
    h2o_data = data[data['water'] == 'H2O']
    d2o_data = data[data['water'] == 'D2O']

    # Only consider fresh samples
    fresh_h2o_data = h2o_data[h2o_data['age'] == 'fresh']
    fresh_d2o_data = d2o_data[d2o_data['age'] == 'fresh']

    # Define the bar+passes values to consider
    bar_passes_values = ['5b1', '5b2']

    # Plot the data
    plt.figure(figsize=(8, 5))

    data_sets = [
        (fresh_h2o_data, 'H2O'), 
        (fresh_d2o_data, 'D2O')
    ]

    for age_data, water_type in data_sets:
        for bar_passes in bar_passes_values:
            bar_passes_data = age_data[age_data['bar_passes'] == bar_passes]
            x = bar_passes_data['water'].replace({'H2O': 0, 'D2O': 1})
            y = bar_passes_data['D32']
            e = bar_passes_data['error_D32']
            plt.errorbar(x, y, e, label=f'{bar_passes}', zorder=3, linestyle='None', marker='o')

    plt.title('Effect of D2O on D32 for 5wt% Protein Samples (Fresh)')
    plt.xlabel('Water Type')
    plt.ylabel(r'$D[3,2]$ [$\mu$m]')
    plt.xlim(-0.5, 1.5)
    plt.xticks([0, 1], [r'H$_2$O', r'D$_2$O'])
    plt.grid(True, zorder=0)
    plt.legend(title='Bar + Passes', loc='upper center')
    plt.tight_layout()
    plt.savefig('Mastersizer/Homogeniser_Settings/Effect_of_D2O_on_D32.png', dpi=250)
    plt.show()

def compare_effect_of_centrifuging_and_water():
    '''
    Compare the effect of centrifuging and adding H2O or D2O on the D[3,2] value for the 5wt% protein samples.
    The x-axis has different centrifuging parameters and the y-axis has the D[3,2] value.
    Differentiate the samples based on the water type (H2O or D2O).
    Only include fresh samples and 5b1 bar+passes samples.
    '''
    # Read the data from the 5wt% protein file
    data = pd.read_csv('Mastersizer/Separated_data/5,00.txt', header=0, delimiter=",")
    
    # Filter the data based on selected dates
    data['date'] = pd.to_datetime(data['date'])
    selected_dates = prompt_for_dates(data['date'], wpi_values=[])
    data = data[data['date'].dt.strftime('%Y-%m-%d').isin(selected_dates)]

    # Create the bar_passes column
    data = create_bar_passes_column(data)
    
    # Filter the data to include only fresh samples and 5b1 bar+passes samples
    data = data[(data['age'] == 'fresh') & (data['bar_passes'] == '5b1')]
    
    # Separate the data by water type (H2O and D2O)
    h2o_data = data[data['water'] == 'H2O']
    d2o_data = data[data['water'] == 'D2O']

    # Separate the data based on the presence of centrifuging parameters
    homogenised_data = data[data['rpm'].isna() & data['mins'].isna() & data['acceleration'].isna()]
    centrifuged_data = data.dropna(subset=['rpm', 'mins', 'acceleration'])

    # Replace 's' in the acceleration column with '8'
    centrifuged_data['acceleration'] = centrifuged_data['acceleration'].str.replace('s', '8', regex=False)

    # Replace " None" with NaN
    centrifuged_data['rpm'].replace("None", np.nan, inplace=True)
    centrifuged_data['mins'].replace("None", np.nan, inplace=True)
    centrifuged_data['acceleration'].replace("None", np.nan, inplace=True)

    # Convert rpm, mins, and acceleration to numeric types
    centrifuged_data['rpm'] = pd.to_numeric(centrifuged_data['rpm'], errors='coerce')
    centrifuged_data['mins'] = pd.to_numeric(centrifuged_data['mins'], errors='coerce')
    centrifuged_data['acceleration'] = pd.to_numeric(centrifuged_data['acceleration'], errors='coerce')

    # Sort the centrifuged data based on rpm, mins, and acceleration
    centrifuged_data = centrifuged_data.sort_values(by=['rpm', 'mins', 'acceleration'])

    # Combine homogenised and centrifuged data
    combined_data = pd.concat([homogenised_data, centrifuged_data])

    # Collect all unique labels from both H2O and D2O data
    all_labels = set()
    for water_type in ['H2O', 'D2O']:
        water_data = combined_data[combined_data['water'] == water_type]
        centrifuged_data = water_data.dropna(subset=['rpm', 'mins', 'acceleration'])
        all_labels.update([f'{rpm} rpm, {mins} mins, {accel} g' for rpm, mins, accel in zip(centrifuged_data['rpm'].astype(str), centrifuged_data['mins'].astype(str), centrifuged_data['acceleration'].astype(str))])

    # Sort all labels and add 'Homogenised' at the beginning
    all_labels = ['Homogenised'] + sorted(all_labels)

    # Plot the data
    plt.figure(figsize=(12, 6))

    markers = {'H2O': 'o', 'D2O': 's'}
    colors = {'H2O': 'blue', 'D2O': 'red'}
    labels = {'H2O': 'H2O', 'D2O': 'D2O'}

    for water_type, marker in markers.items():
        water_data = combined_data[combined_data['water'] == water_type]
        homogenised_data = water_data[water_data['rpm'].isna() & water_data['mins'].isna() & water_data['acceleration'].isna()]
        centrifuged_data = water_data.dropna(subset=['rpm', 'mins', 'acceleration'])
        
        # Plot homogenised data
        x_homogenised = [0] * len(homogenised_data)
        y_homogenised = homogenised_data['D32']
        e_homogenised = homogenised_data['error_D32']
        plt.errorbar(x_homogenised, y_homogenised, e_homogenised, marker=marker, color=colors[water_type], zorder=3, linestyle = 'None')
        
        # Plot centrifuged data
        x_centrifuged = [all_labels.index(f'{rpm} rpm, {mins} mins, {accel} g') for rpm, mins, accel in zip(centrifuged_data['rpm'].astype(str), centrifuged_data['mins'].astype(str), centrifuged_data['acceleration'].astype(str))]
        y_centrifuged = centrifuged_data['D32']
        e_centrifuged = centrifuged_data['error_D32']
        plt.errorbar(x_centrifuged, y_centrifuged, e_centrifuged, linestyle='None', marker=marker, color=colors[water_type], label=labels[water_type], zorder=3)

    # Set x-axis labels
    plt.xticks(range(len(all_labels)), all_labels, rotation=45, ha='right')

    plt.title('Effect of Centrifuging and Water Type on D32 for 5wt% Protein Samples (Fresh, 5b1)')
    plt.xlabel('Centrifuging Parameters')
    plt.ylabel(r'$D[3,2]$ [$\mu$m]')
    plt.grid(True, zorder=0)
    plt.legend(title='Water Type', loc='upper center')
    plt.tight_layout()
    plt.savefig('Mastersizer/Homogeniser_Settings/Effect_of_Centrifuging_and_Water_on_D32_2801.png', dpi=250)
    plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import Listbox, MULTIPLE, END, Button, Label, Frame

def filter_and_plot_x_vs_I(file_path, date=None, WPI_value=None, bar=None, passes=None, age=None):
    data = pd.read_csv(file_path, header=0, delimiter=',', decimal='.')
    
    # Display an overview of all possibilities
    overview = data.iloc[:, :16]
    print("Overview of all possibilities:")
    print(overview)

    # Create a popup window for filtering
    selected_filters = {header: [] for header in data.columns[:16]}

    def apply_filters(header):
        selected_filters[header] = [listboxes[header].get(i) for i in listboxes[header].curselection()]
        print(f"Selected {header}: {selected_filters[header]}")
        update_filtered_data()

    def select_all(header):
        listboxes[header].select_set(0, END)
        apply_filters(header)

    def update_filtered_data():
        filtered_data = data
        for header, selected_values in selected_filters.items():
            if selected_values:
                if header in ['mins', 'rpm', 'oil_wt', 'protein_wt']:
                    filtered_data = filtered_data[filtered_data[header].astype(float).isin([float(val) for val in selected_values])]
                else:
                    filtered_data = filtered_data[filtered_data[header].astype(str).isin(selected_values)]
        
        # Show the final overview of the remaining filtered data
        print("Filtered Data Overview:")
        print(filtered_data.iloc[:, :16])

    def show_overview():
        filtered_data = data
        for header, selected_values in selected_filters.items():
            if selected_values:
                if header in ['mins', 'rpm', 'oil_wt', 'protein_wt']:
                    filtered_data = filtered_data[filtered_data[header].astype(float).isin([float(val) for val in selected_values])]
                else:
                    filtered_data = filtered_data[filtered_data[header].astype(str).isin(selected_values)]
        
        # Show the final overview of the remaining filtered data
        print("Filtered Data Overview:")
        print(filtered_data.iloc[:, :16])

        # Create a new popup window for showing the overview
        overview_root = tk.Tk()
        overview_root.title("Overview of Filtered Data")

        overview_frame = Frame(overview_root, padx=10, pady=10)
        overview_frame.pack(padx=10, pady=10)

        # Create a frame for the headers
        header_frame = Frame(overview_frame)
        header_frame.pack()

        # Add headers as column headers
        headers = ['Sample No'] + [header for header in filtered_data.columns[:16] if header not in ['D32', 'error_D32', 'peak_width', 'error_peak_width']]
        for col, header in enumerate(headers):
            Label(header_frame, text=header, font=("Times New Roman", 10, "bold")).grid(row=0, column=col, padx=5, pady=5)

        # Create a listbox for the sample specifications
        overview_listbox = Listbox(overview_frame, selectmode=MULTIPLE, width=150, height=20)
        for index, row in filtered_data.iterrows():
            row_data = [f"Sample {index}"] + [str(row[header]) for header in filtered_data.columns[:16] if header not in ['D32', 'error_D32', 'peak_width', 'error_peak_width']]
            overview_listbox.insert(END, ' | '.join(row_data))
        overview_listbox.pack(pady=5)

        def plot_selected_samples():
            plt.figure(figsize=(10,5))
            plt.rcParams["font.family"] = "Times New Roman"
            selected_indices = [int(overview_listbox.get(i).split()[1]) for i in overview_listbox.curselection()]
            selected_samples = data.iloc[selected_indices]
            print("Selected Samples:")
            print(selected_samples)

            # Extract x and y values
            x_values = data.columns[17:171].astype(float)
            label_headers = data.columns[:16]
            for index, row in selected_samples.iterrows():
                y_values = row[17:171].astype(float)
                label = ', '.join([f"{header}: {row[header]}" for header in label_headers if len(selected_filters[header]) > 1])
                plt.plot(x_values, y_values, label=f"{label}")

            # Set title based on single selected options
            title_parts = [f"{header}: {selected_filters[header][0]}" for header in label_headers if len(selected_filters[header]) == 1]
            title = ' | '.join(title_parts)
            save_parts = [f"{header}-{selected_filters[header][0]}" for header in label_headers if len(selected_filters[header]) == 1]
            
            # Include unique values for multiple selected filters
            for header in label_headers:
                if len(selected_filters[header]) > 1:
                    unique_values = ','.join(sorted(set(selected_samples[header].astype(str))))
                    save_parts.append(f"{header}-[{unique_values}]")
            
            save_title = '_'.join(save_parts).replace('.', ',')
            plt.title(title)

            plt.xlabel(r'$D$ [$\mu$m]')
            plt.ylabel(r'$I$')
            plt.xscale('log')
            plt.xlim(min(x_values), max(x_values))
            plt.legend(loc='center left', bbox_to_anchor=(1.01, 0.5))
            plt.tight_layout()
            plt.grid()
            plt.savefig('Mastersizer/Measurement_Comparison/{}.png'.format(save_title), dpi=250)
            plt.show()

            overview_root.quit()
            overview_root.destroy()

        Button(overview_frame, text="Plot Data", command=plot_selected_samples).pack(pady=5)

        overview_root.update_idletasks()
        window_width = min(overview_root.winfo_width(), overview_root.winfo_screenwidth() - 20)
        window_height = min(overview_root.winfo_height(), overview_root.winfo_screenheight() - 20)
        screen_width = overview_root.winfo_screenwidth()
        screen_height = overview_root.winfo_screenheight()
        position_top = int(screen_height / 2 - window_height / 2)
        position_right = int(screen_width / 2 - window_width / 2)
        overview_root.geometry(f'{window_width}x{window_height}+{position_right}+{position_top}')

        overview_root.mainloop()

    def refresh_filters():
        for header in selected_filters:
            selected_filters[header] = []
        for listbox in listboxes.values():
            listbox.selection_clear(0, END)
        print("Filters refreshed")

    def plot_data():
        filtered_data = data
        for header, selected_values in selected_filters.items():
            if selected_values:
                if header in ['mins', 'rpm', 'oil_wt', 'protein_wt']:
                    filtered_data = filtered_data[filtered_data[header].astype(float).isin([float(val) for val in selected_values])]
                else:
                    filtered_data = filtered_data[filtered_data[header].astype(str).isin(selected_values)]

        # Show the final overview of the remaining filtered data
        print("Filtered Data Overview:")
        print(filtered_data.iloc[:, :16])

        # Extract x and y values
        x_values = data.columns[17:171].astype(float)
        for index, row in filtered_data.iterrows():
            y_values = row[17:171].astype(float)
            label = ', '.join([f"{header}: {row[header]}" for header in selected_filters if selected_filters[header]])
            plt.plot(x_values, y_values, label=f"Sample {index} ({label})")
        
        plt.xlabel('X-axis values')
        plt.ylabel('Y-axis values')
        plt.xscale('log')
        plt.legend()
        plt.show()

        root.quit()
        root.destroy()

    root = tk.Tk()
    root.title("Filter Data")

    frame = Frame(root, padx=10, pady=10)
    frame.pack(padx=10, pady=10)

    # Exclude columns 1, 2, 14, and 15 from filtering
    headers = [col for i, col in enumerate(data.columns[:16]) if i not in [1, 2, 14, 15]]
    listboxes = {}

    for i, header in enumerate(headers):
        row = i % 2
        col = i // 2
        subframe = Frame(frame, padx=5, pady=5)
        subframe.grid(row=row, column=col, padx=5, pady=5)
        
        Label(subframe, text=f"Select {header}:", font=("Times New Roman", 12)).pack(pady=5)
        listbox = Listbox(subframe, selectmode=MULTIPLE, width=20, height=10)
        
        # Handle mixed data types when sorting unique values
        unique_values = data[header].dropna().unique()
        unique_values = sorted(unique_values, key=lambda x: (str(type(x)), x))
        
        for value in unique_values:
            listbox.insert(END, value)
        listbox.pack(pady=5)
        listboxes[header] = listbox

        Button(subframe, text=f"Apply {header} Filter", command=lambda h=header: apply_filters(h)).pack(pady=5)
        Button(subframe, text=f"Select All {header}", command=lambda h=header: select_all(h)).pack(pady=5)

    Button(frame, text="Show Overview", command=show_overview).grid(row=2, columnspan=len(headers) // 2, pady=10)
    Button(frame, text="Plot Data", command=plot_data).grid(row=3, columnspan=len(headers) // 2, pady=10)
    Button(frame, text="Refresh Filters", command=refresh_filters).grid(row=4, columnspan=len(headers) // 2, pady=10)

    root.update_idletasks()
    window_width = min(root.winfo_width(), root.winfo_screenwidth() - 20)
    window_height = min(root.winfo_height(), root.winfo_screenheight() - 20)
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    position_top = int(screen_height / 2 - window_height / 2)
    position_right = int(screen_width / 2 - window_width / 2)
    root.geometry(f'{window_width}x{window_height}+{position_right}+{position_top}')

    root.mainloop()

