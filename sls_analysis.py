import sls_csv_reduction as mc
import sls_analysis_functions as mwa
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


'''
Functions in ms_functions.py:
- select_files()
- get_values(name)
- make_files(WPI_values)
- write_data(WPI_value, date, D3_2, bar_passes, oil, sample_number, fresh_frozen, water)
- plot_whole_graph(data)
- plot_avg_D3_2(D3_2, names)
- plot_whole_graph(data)
- plot_splitted(WPI_values)
'''

# Prompt the user to select files
selected_files = mc.select_files()

### REDUCE WPI DATA
# Define the WPI values	
WPI_values = np.array(['0,25', '0,50', '0,75', '1,50', '2,00', '5,00'])
WPI_values_float = np.array([0.25, 0.50, 0.75, 1.50, 2.00, 5.00])

# Create the files to write to, only needed when you want to make designated files based on WPI concentration
# mc.make_files(WPI_values) #Protein is WPI

# Extract the data per protein percentage value
if selected_files:
    for selected_file in selected_files:
        print(f"Selected file: {selected_file}")

        ### Load in the selected csv file
        file_path = os.path.join("Mastersizer/Data_Mastersizer_in_CSV", selected_file)
        data = pd.read_csv(file_path, header=0, delimiter=',', decimal='.')
        names = data['Sample Name']
        D3_2 = data['D [3,2]']

        ### Plot the data to get a first glance at the data and retrieve peak locations
        peak_width, peak_width_error = mc.plot_whole_graph(data)
        intensity = pd.read_csv('Mastersizer/intensity.txt', header = None)

        ### Extract the data and save to splitted txt files
        # mc.split_data_on_wt(names, D3_2, WPI_values, peak_width, peak_width_error, intensity)

        ### Plot the average D3_2 values with the error from the whole graphs
        # mc.plot_avg_D3_2(D3_2, names)
else:
    print("No files selected.")

### ANALYSE WPI DATA
choice = mwa.prompt_function_selection()

if choice == '1':
    for i in WPI_values:
        mwa.plot_series_from_file(i)
elif choice == '2':
    mwa.plot_series_from_file_subplots(WPI_values)
elif choice == '3':
    mwa.plot_histogram_for_D32_ranges()
elif choice == '4':
    mwa.compare_samples_by_bar_passess(WPI_values_float)
elif choice == '5':
    mwa.compare_samples_by_bar_passes_averages(WPI_values_float)
elif choice == '6':
    mwa.compare_effect_of_D2O_on_D32()
elif choice == '7':
    mwa.compare_effect_of_centrifuging_and_water()
elif choice == '8':
    file_path = 'Mastersizer/x_vs_I.txt'  # Update with the actual file path
    mwa.filter_and_plot_x_vs_I(file_path)
else:
    print("Invalid choice. Please run the script again and select a valid option.")
