import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('seaborn')

# Define a function to read in the World Bank data
import pandas as pd

def read_worldbank_data(filename):
    # Read in the World Bank formatted data
    df = pd.read_csv(filename, skiprows=4)

    # Remove any unnecessary columns
    df.drop(['Country Code', 'Indicator Code'], axis=1, inplace=True)

    # Remove any rows with missing values
    df.dropna(inplace=True)

    # Transpose the dataframe with years as columns
    df_years = df.set_index('Country Name').transpose()

    # Transpose the dataframe with countries as columns
    df_countries = df.set_index('Indicator Name')

    return df_years, df_countries

filename = "datasets/API_19_DS2_en_csv_v2_5361599.csv"
df_years, df_countries = read_worldbank_data(filename)

print("Dataframe with years as columns:")
print(df_years.head())

print("\nDataframe with countries as columns:")
print(df_countries.head())


# Load the global temperature dataset and store it in a dataframe
orig_temp_data = pd.read_csv('datasets/GlobalTemperatures.csv')
global_temp_df = orig_temp_data.copy()
global_temp_df.head()

# Choose the required columns
global_temp_df = global_temp_df[['dt', 'LandAndOceanAverageTemperature']]

# Set the date column as a DateTimeIndex and sort it
global_temp_df['dt'] = pd.to_datetime(global_temp_df['dt'])
global_temp_df.set_index('dt', inplace = True)
global_temp_df.sort_index(axis = 0, inplace = True)

# Resample annually and rename index & columns
global_temp_df = global_temp_df.resample('A').mean()
global_temp_df.rename(columns = {'LandAndOceanAverageTemperature': 'AnnualAverageTemp'}, inplace = True)
global_temp_df.index.rename('Year', inplace = True)
global_temp_df.index = global_temp_df.index.year

# Check the number of missing values and the corresponding years
print(global_temp_df.isnull().sum())
print(global_temp_df[global_temp_df['AnnualAverageTemp'].isnull()].index)
global_temp_df.dropna(inplace = True)

# Calculate the global baseline temperature
global_ref_temp = global_temp_df.loc['1951':'1980'].mean()['AnnualAverageTemp'] 

# Create the temperature anomaly column
global_temp_df['Temperature Anomaly'] = global_temp_df['AnnualAverageTemp'] - global_ref_temp
global_temp_df.drop(['AnnualAverageTemp'], axis = 1, inplace = True)
global_temp_df.head()

# Plot the temperature anomaly
plt.figure(figsize = (15, 8))
plt.plot(global_temp_df['Temperature Anomaly'], 'r')
plt.title('The Warming Earth\nGlobal temperature anomalies (annual) for 1850-2015', fontsize = 19)
plt.xlabel('Year', fontsize = 15)
plt.ylabel('Temperature anomaly (degree Celsius)', fontsize = 15)
plt.show()

# Load the natural disaster dataset and store it in a dataframe


# Read in the original data
orig_disaster_data = pd.read_csv('datasets/number-of-natural-disaster-events.csv')

# Make a copy of the data
nat_disaster_df = orig_disaster_data.copy()

# Remove the 'Code' column
nat_disaster_df.drop(['Code'], axis=1, inplace=True)

# Check the different types of 'Entity' values
nat_disaster_df['Entity'].unique()

# Pivot the dataframe
nat_disaster_df = nat_disaster_df.pivot(index='Year', columns='Entity', values='Number of reported natural disasters (reported disasters)')

# Remove the 'Impact' column
nat_disaster_df.drop(['Impact'], axis=1, inplace=True)

# Handle missing values and rename columns
nat_disaster_df.fillna(value=0, inplace=True)
nat_disaster_df = nat_disaster_df.add_suffix(' (Occurrence)')

# Plot the types of natural disasters occurrences for 1950-2018
colors = ['#4169e1', '#b22222', '#228b22', '#ff4500', '#9370db', '#8b4513', '#c71585', '#ffd700', 'olive']

nat_disaster_df.drop(['All natural disasters (Occurrence)'], axis=1).loc[1950:].plot.bar(width=0.8, stacked=True, color=colors, figsize=(15, 8))

plt.title('Global occurrences of natural disasters for 1950-2018', fontsize=19)
plt.xlabel('Year', fontsize=15)
plt.ylabel('Occurrence', fontsize=15)
plt.legend(loc=2, prop={'size': 12})

plt.show()


fig, ax = plt.subplots(figsize=(14, 8))
ax2 = ax.twinx()

line1 = ax.plot(nat_disaster_df.loc[:2015, 'All natural disasters (Occurrence)'], '-ro', markersize=4, label='All natural disasters (Occurrence)')
line2 = ax2.plot(global_temp_df.loc[1900:, 'Temperature Anomaly'], 'b-', label='Temperature Anomaly')

lines = line1 + line2
labels = [l.get_label() for l in lines]

plt.title('All natural disaster occurrences and temperature anomaly for 1900-2015', fontsize=19)
ax.set_xlabel('Year', fontsize=15)
ax.set_ylabel('Occurrence', fontsize=15, color='r')
ax2.set_ylabel('Temperature anomaly (degree Celsius)', fontsize=15, color='b')
ax.legend(lines, labels, loc=0, prop={'size': 12})

plt.show()


# Read the dataset
orig_econ_data = pd.read_csv('datasets/economic-damage-from-natural-disasters.csv')
econ_dmg_df = orig_econ_data.copy()

# Remove the 'Code' column
econ_dmg_df.drop(['Code'], axis=1, inplace=True)

# Pivot the dataframe
econ_dmg_df = econ_dmg_df.pivot(index='Year', columns='Entity', values='Total economic damage from natural disasters (US$)')

# Remove the irrelevant columns, rename the other columns and handle the missing values
econ_dmg_df.drop(['Impact'], axis=1, inplace=True)
econ_dmg_df.fillna(value=0, inplace=True)
econ_dmg_df = econ_dmg_df.add_suffix(' (Economic Damage)')

# Define the color palette
colors = ['#E69F00', '#56B4E9', '#F0E442', '#009E73', '#D55E00', '#CC79A7']

# Plot the economic damage categorized by the type of natural disasters for 1950-2018
((econ_dmg_df.drop(['All natural disasters (Economic Damage)'], axis=1).loc[1950:]) / 1e9).plot.bar(width=0.8, stacked=True, color=colors, figsize=(15, 8))

plt.title('Economic damage by type of natural disaster for 1950-2018', fontsize=19)


plt.show()
# Combine dataframes
combined_df = global_temp_df.join([nat_disaster_df, econ_dmg_df], how='inner')

# Compute correlation table
correlation_table = combined_df.corr()

# Compute correlation with respect to temperature anomaly
correlations = combined_df.corr()['Temperature Anomaly']

# Plot correlation heatmap
fig, ax = plt.subplots(figsize=(10, 10))

labels = [name for name in correlation_table.columns]

cax = ax.matshow(correlation_table, cmap='RdYlGn')

cbar = fig.colorbar(cax, shrink=0.82)

ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))

ax.set_xticklabels(labels, fontsize=12, rotation='vertical')
ax.set_yticklabels(labels, fontsize=12)

ax.grid(False)

plt.show()




