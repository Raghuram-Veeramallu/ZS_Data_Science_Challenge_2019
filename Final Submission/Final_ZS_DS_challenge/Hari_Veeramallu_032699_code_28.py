# Basic Libraries for data handling
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.preprocessing import LabelEncoder, StandardScaler, normalize
from datetime import date, datetime
from dateutil.parser import parse
# Libraries to implement Machine Learning Algorithms
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, ElasticNet, Lasso
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor

# Read data to be preprocessed and analysed.
def read_data_from_csv():
    data = pd.read_csv('Cristano_Ronaldo_Final_v1/Data.csv')
    # Removing unwanted columns
    data.drop('Unnamed: 0', axis = 1, inplace = True)
    # Replacing the shot_id_number as it contains Nan
    data['shot_id_number'] = range(1, len(data)+1)
    return data

# Separate the Home/Away columns into two separate columns
def separate_home_away_col(lines):
    # Home Plays
    away = []
    # Away Plays
    home = []
    for l in lines:
        l = str(l)
        # If the venue is not given then place None in both of them
        if (l == 'nan'):
            away.append(None)
            home.append(None)
        else:
            tokens = l.split(' ')
            if (tokens[1] == '@'):
                # Played away
                away.append(tokens[2])
                home.append(None)
            else:
                # Played at home against the team..
                away.append(None)
                home.append(tokens[2])
    return away, home

# Separate the latitudes and longitudes into two separate columns
def separate_lat_long(lines):
    lat = []
    lon = []
    for l in lines:
        l = str(l)
        if (l == 'nan'):
            # If latitudes and longitudes are not given then place None in place
            lat.append(None)
            lon.append(None)
        else:
            # Else split the latitudes and longitudes into two separate columns
            tokens = l.split(',')
            lat.append(tokens[0])
            lon.append(tokens[1])
    return lat,lon

# calculate the number of days before today the match was played
def calc_days(g_date):
    if (type(g_date) == str):
        # Split the data into mm dd and yyyy
        tokens = g_date.split('-')
        # Converting the given date into date format for ease of operations
        given_date = date(int(tokens[0]), int(tokens[1]), int(tokens[2]))
        today = date.today()
        # Difference between the days
        diff = today - given_date
        # Calculate the days between them
        return diff.days
    else:
        # Return None if the date is not mentioned
        return None

# Function for preprocessing the data
def preprocess_data(data):
    # Fill the None values of the game_season column using fillna
    data.game_season = data.game_season.fillna(method = 'ffill')
    # Converting the string data into Categories using LabelEncoder
    change_cols = ['area_of_shot','shot_basics','range_of_shot','team_name', 'game_season', 'match_id']
    for i in range (0,len(change_cols)):
        le = LabelEncoder()
        le.fit(data[change_cols[i]].tolist())
        new_col = le.transform(data[change_cols[i]].tolist())
        data[change_cols[i]] = new_col

    # Split the home/away column into two separate columns and label encoding them
    away, home = separate_home_away_col(list(data['home/away']))
    data['home'] = home
    data['away'] = away
    data.drop('home/away', axis = 1, inplace = True)
    data[['home', 'away']] = data[['home', 'away']].fillna(0)
    le_ha = LabelEncoder()
    le_ha.fit((data['home'].tolist() + data['away'].tolist()))
    new_col = le_ha.transform(data['home'].tolist())
    data['home'] = new_col
    new_col = le_ha.transform(data['away'].tolist())
    data['away'] = new_col   

    # Split the lat/lng into two separate columns
    lat, lon = separate_lat_long(list(data['lat/lng']))
    data['lat'] = lat
    data['lng'] = lon
    data.drop('lat/lng', axis = 1, inplace = True)

    # Combining the type_of_shot and type_of_combined_shot together as they both converse each other
    data['type_of_shot'] = data['type_of_shot'].fillna(data['type_of_combined_shot'])
    data.drop('type_of_combined_shot', axis = 1, inplace = True)
    data['type_of_shot'] = data['type_of_shot'].apply(lambda x: x.split('-')[1])

    # The date is converted into number of days before the game was played
    data.date_of_game = data.date_of_game.apply(lambda x: calc_days(x))

    return data

def scale_data(data):
    # Fill the null values
    data.loc[:, data.columns != 'is_goal'] = data.loc[:, data.columns != 'is_goal'].fillna(method = 'ffill')
    # Scaling the data for better predictions
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(data.loc[:, data.columns != 'is_goal'])
    scaled_df = pd.DataFrame(scaled_df, columns=data.loc[:, data.columns != 'is_goal'].columns)
    scaled_df['is_goal'] = data['is_goal']
    scaled_df['shot_id_number'] = data['shot_id_number']
    scaled_df = scaled_df.set_index('shot_id_number')
    return scaled_df


# Select all the important features based on the correlation matrix
def select_important_features(data):
    new_data = data[['match_event_id','location_y','power_of_shot','distance_of_shot', 'area_of_shot', 'shot_basics','range_of_shot','distance_of_shot.1','is_goal']]
    return new_data

# Split the train and test data according to the is_goal parameter
def split_data(new_data):
    # If is_goal is present then train data, else test data
    test = new_data[data['is_goal'].isnull()]
    train = new_data[data['is_goal'].notnull()]
    return train, test

# Train the model using Support Vector Machine
def model_train(train, test):
    lgr = LogisticRegression()
    lgr.fit(train.iloc[:,:-1], train['is_goal'])
    #y_pred = lgr.predict(test.iloc[:,:-1])
    lp = lgr.predict_proba(test.iloc[:,:-1])
    y_pred = [x[0] for x in lp]
    return y_pred

# Convert the datafram into a csv file for submission
def convert_csv(y_pred):
    temp = [[str(int(x)) for x in test.index], [x for x in list(y_pred)]]
    df = pd.DataFrame(temp).transpose()
    df.columns = ['shot_id_number', 'is_goal']
    df.set_index('shot_id_number')
    df.to_csv('Hari_Veeramallu_032699_prediction_28.csv', index = False)

# Main function of the python file to do all the processes
if __name__ == '__main__':
    data = read_data_from_csv()
    data = preprocess_data(data)
    data = scale_data(data)
    new_data = select_important_features(data)
    train, test = split_data(new_data)
    y_pred = model_train(train, test)
    convert_csv(y_pred)
