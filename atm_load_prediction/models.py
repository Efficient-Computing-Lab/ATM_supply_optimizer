#Part of ATMO-MoRe ATM Load Predictor, a system that models ATM load and their need for resupply.
#Copyright (C) 2024  Evangelos Psomakelis
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU Affero General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#any later version.
#
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#GNU Affero General Public License for more details.
#
#You should have received a copy of the GNU Affero General Public License
#along with this program.  If not, see <https://www.gnu.org/licenses/>.

import pandas
from sklearn.linear_model import LinearRegression, LassoLars, SGDRegressor, BayesianRidge, \
    LogisticRegression, OrthogonalMatchingPursuit
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import datetime
import holidays

class LinearModel():
    """ A Simple linear model that is trained on supply timeseries and predicts the next need for supply """
    algorithms = [
        (OrthogonalMatchingPursuit,{}), # - 0 - {'mean_accuracy': -0.018479540356505224, 'r2': -0.018479540356505224, 'mean_absolute_error': 1451.3930144986389}
        (BayesianRidge,{}),# - 1 - {'mean_accuracy': 0.01745275792508246, 'r2': 0.01745275792508246, 'mean_absolute_error': 1432.8390195058812}
        (SGDRegressor,{'max_iter':1000, 'tol':1e-3}),# - 2 - {'mean_accuracy': -0.05118982345439753, 'r2': -0.05118982345439753, 'mean_absolute_error': 1543.8355415060707}
        (LassoLars,{'alpha':.1}),# - 3 - {'mean_accuracy': -0.13665600147935383, 'r2': -0.13665600147935383, 'mean_absolute_error': 0.8536458797961455}
        (LinearRegression,{}),# - 4 - {'mean_accuracy': -0.09793593082460776, 'r2': -0.09793593082460776, 'mean_absolute_error': 0.8403160340397787}
        (DecisionTreeRegressor,{})# - 5 - {'mean_accuracy': -0.8811254899217652, 'r2': -0.8811254899217652, 'mean_absolute_error': 0.8927514727246861}
    ]

    def __init__(self,dataset:pandas.DataFrame,target:str,features:list=[],categorical_features:list=[],algorithm:int=0):
        self.clf,self.params = LinearModel.algorithms[algorithm]
        self.model = self.clf(**self.params)
        self.dataset = dataset
        self.target = target
        self.features = features
        self.categorical_features = categorical_features
        self.encoders = {key:LabelEncoder() for key in categorical_features}
        self.average_change = None

    def train(self):
        """ Trains the model using the parameters and data provided during object creation """
        for key in self.categorical_features:
            self.encoders[key].fit(self.dataset[key])
            self.dataset[key] = self.encoders[key].transform(self.dataset[key])
        X = self.dataset[self.features+self.categorical_features]
        y = self.dataset[self.target]
        self.model.fit(X, y)
        self.average_change = float(y.mean())
    
    def predict(self,state:pandas.DataFrame):
        """ Predicts the target feature using the state provided """
        for key in self.categorical_features:
            state[key] = self.encoders[key].transform(state[key])
        X = state[self.features+self.categorical_features]
        return self.model.predict(X)
    
    def score(self,state:pandas.DataFrame,y_true:list):
        """ Scores the model using the default scorer """
        for key in self.categorical_features:
            state[key] = self.encoders[key].transform(state[key])
        X = state[self.features+self.categorical_features]
        return self.model.score(X,y_true)
    
    def apply(self,data:pandas.DataFrame,load_threshold:float=0.2):
        """ Performs a regression trying to estimate the amount of days before the ATM needs cash supply """
        greece_holidays = holidays.country_holidays('GR')
        time = data.index[0]
        value = float(data['value_t-1'].iloc[0])
        cur_data = data.copy()
        train_features_actual = self.features
        days = 0
        while value > load_threshold:
            prev_value = value
            value = float(self.predict(cur_data[train_features_actual]))
            time:datetime.datetime = time + datetime.timedelta(days=1)

            if value > prev_value:
                return (days,cur_data)
            elif value == prev_value:
                value = prev_value+self.average_change

            cur_data[f'day_of_week_t-1'] = time.weekday()
            cur_data[f'monday_t-1'] = time.weekday() == 0
            cur_data[f'tuesday_t-1'] = time.weekday() == 1
            cur_data[f'wednesday_t-1'] = time.weekday() == 2
            cur_data[f'thursday_t-1'] = time.weekday() == 3
            cur_data[f'friday_t-1'] = time.weekday() == 4
            cur_data[f'saturday_t-1'] = time.weekday() == 5
            cur_data[f'sunday_t-1'] = time.weekday() == 6
            cur_data[f'day_of_month_t-1'] = time.day
            cur_data[f'month_t-1'] = time.month
            cur_data[f'workday_t-1'] = time.weekday() < 5
            cur_data[f'holiday_t-1'] = int(datetime.datetime.timestamp(time)) in greece_holidays
            cur_data[f'value_t-1'] = value
            days += 1

            cur_data[train_features_actual] = cur_data[train_features_actual].astype('float32')

        return (days if days > 0 else 1,cur_data)

class ClassifierModel():
    """ A classification model that is trained on supply timeseries and predicts the next need for supply """
    algorithms = [
        (LogisticRegression,{'max_iter':1000}), # - 0 - {'mean_accuracy': 0.33909028375935474, 'r2': -0.928734351821944, 'mean_absolute_error': 0.9819655168282421}
        (KNeighborsClassifier,{'n_neighbors':3}), # - 1 - {'mean_accuracy': 0.4065950231059898, 'r2': -0.8235700086767245, 'mean_absolute_error': 0.8753908367432857}
        (SVC,{'kernel':"linear", 'C':0.025, 'random_state':42}), # - 2 - {'mean_accuracy': 0.426470866752057, 'r2': -0.17555850693897526, 'mean_absolute_error': 0.7743700175748767}
        (SVC,{'gamma':2, 'C':1, 'random_state':42}), # - 3 - {'mean_accuracy': 0.40898183944708627, 'r2': -0.3624571042356257, 'mean_absolute_error': 0.8305480861390149}
        #(GaussianProcessClassifier,{'kernel':1.0 * RBF(1.0), 'random_state':42}), # - 3 -  Killed
        (DecisionTreeClassifier,{'max_depth':5, 'random_state':42}), # - 4 - {'mean_accuracy': 0.42998091919359144, 'r2': -0.5768736325864317, 'mean_absolute_error': 0.8047435980771631}
        (RandomForestClassifier,{ # - 5 - {'mean_accuracy': 0.4307004598504015, 'r2': -0.38660972523256726, 'mean_absolute_error': 0.7805300659044455}
            'max_depth':5, 'n_estimators':10, 'max_features':1, 'random_state':42
        }),
        (MLPClassifier,{'alpha':1, 'max_iter':1000, 'random_state':42}), # - 6 - {'mean_accuracy': 0.3810996793795832, 'r2': -1.0385668709157607, 'mean_absolute_error': 0.9201079004789642}
        (AdaBoostClassifier,{'algorithm':"SAMME", 'random_state':42}), # - 7 - {'mean_accuracy': 0.3489018226572143, 'r2': -1.0664042232905473, 'mean_absolute_error': 0.9482324650504768}
        (GaussianNB,{}), # - 8 - {'mean_accuracy': 0.2852852826629438, 'r2': -1.3622269339287902, 'mean_absolute_error': 1.179277551586621}
        (QuadraticDiscriminantAnalysis,{}) # - 9 - {'mean_accuracy': 0.21275619770564919, 'r2': -2.348372986325595, 'mean_absolute_error': 1.485690439369573}
    ]

    def __init__(self,dataset:pandas.DataFrame,target:str,features:list=[],categorical_features:list=[],algorithm:int=0):
        self.clf,self.params = ClassifierModel.algorithms[algorithm]
        self.model = self.clf(**self.params)
        self.dataset = dataset
        self.target = target
        self.features = features
        self.categorical_features = categorical_features
        self.encoders = {key:LabelEncoder() for key in categorical_features}

    def train(self):
        for key in self.categorical_features:
            self.encoders[key].fit(self.dataset[key])
            self.dataset[key] = self.encoders[key].transform(self.dataset[key])
        X = self.dataset[self.features+self.categorical_features]
        y = self.dataset[self.target]
        self.model.fit(X, y)
    
    def predict(self,state:pandas.DataFrame):
        for key in self.categorical_features:
            state[key] = self.encoders[key].transform(state[key])
        X = state[self.features+self.categorical_features]
        return self.model.predict(X)
    
    def score(self,state:pandas.DataFrame,y_true:list):
        for key in self.categorical_features:
            state[key] = self.encoders[key].transform(state[key])
        X = state[self.features+self.categorical_features]
        return self.model.score(X,y_true)