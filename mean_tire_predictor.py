"""
Brian Horner
Project For: CS677
Start Date: August 12, 2021
Last Revision Date: August 18, 2021
Project Name: Formula 1 Machine Learning Tires Used Predictor
Description: This program takes data from three csv files and puts them into
dataframes. We mangle the data in order to pull features together from all
three. The program then tries multiple models in order to predict the mean
number of tires used for each race in the 2017 Formula 1 season.
"""

# Module Imports : Classifier Models
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

"""
Datasets from diana on kaggle 
She used this dataset to classify race finish status of drivers of each race 
of a Formula 1 season
Kaggle profile - https://www.kaggle.com/coolcat
Kaggle Project Part 1 - https://www.kaggle.com/coolcat/f1-create-dataset-data-exploration-with-altair?scriptVersionId=8404883
Kaggle Project Part 2 - https://www.kaggle.com/coolcat/f1-binary-classification-of-race-finish-status
"""

# Datasets imports
tire_allocation_df = pd.read_csv('Pirelli_Tyre_Categories.csv')
weather_df = pd.read_csv('Weather_SafetyCar.csv')
tire_clusters = pd.read_csv('tyre_strategy_clusters.csv')

def predictor(models, dataframe):
    """Function that takes a list of models and a dataframe in order to
    return best model accuracy, best model itself and that models predicted
    values for mean tires used."""
    best_accuracy = 0
    for model in models:
        """Data splitting"""
        train_data = dataframe.loc[(dataframe['year'] < 2017)].copy(
            deep=True)
        test_data = dataframe.loc[(dataframe['year'] == 2017)].copy(
        deep=True)

        X_test = test_data.drop(['race', 'mean_tires_used'], axis=1)
        X_train = train_data.drop(['race', 'mean_tires_used'], axis=1)

        Y_train = train_data['mean_tires_used'].values
        Y_test = test_data['mean_tires_used'].values

        """Model fitting and predicting"""
        current_model = model
        # Fitting model and predicting
        current_fit = current_model.fit(X=X_train, y=Y_train)
        y_predict = current_fit.predict(X_test)
        # Doing comparisons with previous models
        current_acc = accuracy_score(Y_test, y_predict)
        if current_acc > best_accuracy:
            best_accuracy = current_acc
            best_predictions = y_predict
        else:
            pass
    # Returning best model, best accuracy, best models predictions a
    return model, best_accuracy, best_predictions


def table_printer(print_list, header_list, title, margin=20):
    """Formats the predictions, true values and races for table printing."""
    print(title)
    # Adding header
    print_list.insert(0, header_list)
    for index, stuff in enumerate(print_list):
        # Adding a | in front of each value of the lists in print list
        row = '|'.join(str(value).ljust(margin) for value in stuff)
        # Printing the row for the list in print list
        print(row)
        # Adding a line between the header and the data rows
        if index == 0:
            print('-' * len(row))
    print('\n')


lab = LabelEncoder()

# Getting dummy variables for categorical data
tire_allocation_df = pd.get_dummies(tire_allocation_df, columns=['Super Soft', 'Soft', 'Medium', 'Hard', 'Ultra Soft'])
# Converting weather with LabelEncoder
weather_df['weather'] = lab.fit_transform(weather_df['weather'])

# Sorting the two dataframes to be able to take SC Laps and weather data
# without incorrect assignment
tire_allocation_df.sort_values(['year', 'name'], ascending=[True, True],
                               inplace=True)
tire_allocation_df.index = pd.RangeIndex(len(tire_allocation_df.index))

weather_df.sort_values(['year', 'name'], ascending=[True, True], inplace=True)
weather_df.index = pd.RangeIndex(len(weather_df.index))

# Grabbing weather and SC Laps data
tire_allocation_df['SC Laps'] = weather_df['SC Laps']
tire_allocation_df['weather'] = weather_df['weather']

tire_clusters.sort_values(['year', 'name'], ascending=[True, True],
                          inplace=True)
tire_clusters.index = pd.RangeIndex(len(tire_clusters.index))

# Unique values for years and names to iterate and compare
tire_years = tire_allocation_df.year.unique()
tire_races = tire_allocation_df.name.unique()


master_list = []
count = 1
for name in tire_races:
    for year in tire_years:
        temp_list = []
        # Isolating correct year and race
        temp_series = tire_clusters[(tire_clusters.name == name) &
                                    (tire_clusters.year == year)]
        # Getting mean of all drivers tires used in the race
        mean = round(temp_series['clusters'].mean(), 3)
        master_list.append([year, name, mean])
# Using master list to make new dataframe
mean_clusters = pd.DataFrame(master_list, columns=['year', 'name',
                                                   'mean_tire_changes'])
mean_clusters.sort_values(['year', 'name'], ascending=[True, True],
                          inplace=True)

# Isolating year and name for comparison
unique_tire = tire_allocation_df[['year', 'name']].copy(deep=True)
unique_clusters = mean_clusters[['year', 'name']].copy(deep=True)

# Converting year to a string for concatonation with name
unique_tire['year'] = unique_tire['year'].astype(str)
unique_clusters['year'] = unique_clusters['year'].astype(str)

# Combining race and year for easier comparison
# Did this to find year/races that are not in both dataframes
tire_allocation_df['race'] = unique_tire['year'] + " " + unique_tire['name']
mean_clusters['race'] = unique_clusters['year'] + " " + unique_clusters['name']

# Getting the races that are not found in both datasets and deleting
rows_to_delete = pd.concat([tire_allocation_df['race'], mean_clusters[
    'race']]).drop_duplicates(keep=False).index.tolist()

# Dropping 2015 Azebaijan & 2015-2017 German GP
mean_clusters = mean_clusters.drop(rows_to_delete)
mean_clusters.index = pd.RangeIndex(len(mean_clusters.index))

# Dropping 2016 German GP from tire data - not found in clusters
tire_allocation_df = tire_allocation_df.drop(29)
tire_clusters.index = pd.RangeIndex(len(tire_clusters.index))

# Assigning mean_tire_changes to master dataframe
tire_allocation_df['mean_tires_used'] = mean_clusters.loc[:,
                                        ['mean_tire_changes']]


# Getting rid of NAN in SC Laps data - NAN = 0
tire_allocation_df['SC Laps'] = tire_allocation_df['SC Laps'].fillna(0)

# Encoding race name
tire_allocation_df['name'] = lab.fit_transform(tire_allocation_df['name'])

# Converting mean tires used to an integer for rounding and predicting purposes
tire_allocation_df['mean_tires_used'] = tire_allocation_df[
                                             'mean_tires_used'].round().astype(int)


k_neighbors = len(tire_allocation_df['mean_tires_used'].unique())

# Models that are going to be used
# Random states used to avoid changes when running code multiple times
models = [DecisionTreeClassifier(random_state=3390),
          svm.SVC(kernel='linear', random_state=3390),
          svm.SVC(kernel='rbf', random_state=3390),
          svm.SVC(kernel='poly', degree=2, random_state=3390),
          KNeighborsClassifier(n_neighbors=k_neighbors),
          RandomForestClassifier(random_state=3390)
          ]

# Calling predictor in order to find the best model to use
best_model = predictor(models, tire_allocation_df)

print(f"The best model for predicting is {best_model[0]} and its "
      f"accuracy is {best_model[1]}.\n")

# Cleaning up dataframe for printing
tire_allocation_df['name'] = lab.inverse_transform(tire_allocation_df['name'])

predicted_dataframe = tire_allocation_df.loc[(tire_allocation_df['year'] == 2017)].copy(
        deep=True)
predicted_dataframe['True Tires Used'] = predicted_dataframe['mean_tires_used']

predicted_dataframe = predicted_dataframe.loc[:, ['name', 'mean_tires_used']]
predicted_dataframe['Predicted Pit Stops'] = best_model[2]

total_stats = []

# Iterating through predicted to find correctly predicted races
number_of_correct = 0
for index, row in predicted_dataframe.iterrows():
    storage_list = [row['name'], row['mean_tires_used'], row['Predicted Pit Stops']]
    total_stats.append(storage_list)
    if row['mean_tires_used'] == row['Predicted Pit Stops']:
        number_of_correct += 1
    else:
        pass

# Getting rid of Grand Prix on the end of each race name
for list in total_stats:
    if list[0].count(" ") == 3:
        name1, name2, trash1, trash2 = list[0].split(" ")
        list[0] = name1 + " " + name2
    else:
        name, trash, trash2 = list[0].split(" ")
        list[0] = name

total_races_predicted = (len(predicted_dataframe))

print(f"Out of the {total_races_predicted} 2017 Formula 1 races in which "
      f"the model attempted to predict the mean "
      f"tires used {number_of_correct} were correct.\n")


# Calling table printer for correct races, actual mean tires, and predicted
predicted_table_title = "--- Summary of True and Predicted Mean Number of " \
                        "Tires Used and Race Names ---\n"
predicted_table_header = ['Race', 'True Mean Tires', 'Predicted Mean Tires']
table_printer(total_stats, title=predicted_table_title,
              header_list=predicted_table_header)
