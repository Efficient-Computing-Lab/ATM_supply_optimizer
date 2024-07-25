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

from atm_load_prediction.data_handler import DataHandler, Preprocessor, Analyser
from atm_load_prediction.models import LinearModel, ClassifierModel
from atm_load_prediction.utils import timer, test_stationary, printProgressBar
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, \
    mean_squared_log_error, mean_absolute_percentage_error, median_absolute_error, max_error, \
    explained_variance_score
from statistics import mean, stdev
import pandas, numpy, holidays, datetime
import matplotlib.pyplot as plt
from pathlib import Path
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt

# Evaluation options
target_feature = 'value'
train_features = ['day_of_week','workday','holiday','value']
train_features_full = ['day_of_week','day_of_month','month','workday','holiday','value']
limiter = None
#limiter = 3
train_granularity = 'month'
target_transform = lambda x: round(10000*abs(x))
#target_transform = lambda x: round(10000*((x - x.min()) / (x.max() - x.min())))
#target_transform = lambda x: abs(x)
#train_features = ['monday','tuesday','wednesday','thursday','friday','saturday','sunday','workday','holiday']
#train_features_full = ['monday','tuesday','wednesday','thursday','friday','saturday','sunday','day_of_month','month','workday','holiday']
train_features_lstm = ['day_of_week','day_of_month','month','workday','holiday','value']
class_feature_lstm = 'value_t'

def atm_supply_analysis():
    """ Performs simple statistical analysis on supply and timeseries datasets """
    atmomore_handler = DataHandler()
    supply_info = atmomore_handler.get_supply_info()
    atmomore_preprocessor = Preprocessor(supply_info)
    atmomore_preprocessor.clean_supply_types()
    atmomore_analyser = Analyser(atmomore_preprocessor.clean_data)
    analysis = atmomore_analyser.supply_type_statistics()
    Path(f"results").mkdir(parents=True, exist_ok=True)  
    analysis.to_csv('results/load_stats.csv',sep=';',decimal=',',quotechar='"',index=False)
    analysis = atmomore_analyser.timeseries_statistics()
    analysis.to_csv('results/load_timeseries_stats.csv',sep=';',decimal=',',quotechar='"',index=False)

    timeseries = atmomore_preprocessor.create_load_timeseries()
    atmomore_analyser = Analyser(timeseries)
    analysis = atmomore_analyser.timeseries_statistics(date_interval='hour',date_field='timestamp')
    analysis.to_csv('results/load_timeseries_stats_filled.csv',sep=';',decimal=',',quotechar='"',index=False)

def train_model(data:pandas.DataFrame,algorithm:int=0,train_features=[],target_feature=None):
    """ Trains a linear regression model and returns it """
    if algorithm < 6:
        lm = LinearModel(
            dataset=data,
            target=target_feature,
            features=train_features,
            algorithm=algorithm
        )
        lm.train()
        return lm
    else:
        algorithm = algorithm-6
        lm = ClassifierModel(
            dataset=data,
            target=target_feature,
            features=train_features,
            algorithm=algorithm
        )
        lm.train()
        return lm

def train_models(grouped_timeseries=None,time_granularity=None,algorithm=0):
    """ Trains a list of models and returns them as a dict having the ATM code as key and the model as value """
    if grouped_timeseries is None:
        atmomore_handler = DataHandler()
        supply_info = atmomore_handler.get_supply_info()

        atmomore_preprocessor = Preprocessor(supply_info)
        atmomore_preprocessor.clean_supply_types()

        timeseries = atmomore_preprocessor.create_load_timeseries()
        grouped_timeseries = Preprocessor.group_by('ATM',timeseries)

    models_dict = {}
    stationary_tests = []
    for atm in grouped_timeseries:
        try:
            models_dict[atm] = {}
            dataframe = pandas.DataFrame.from_records(grouped_timeseries[atm]).sort_values(by=['timestamp']).reset_index()
            dataframe[[target_feature]] = dataframe[[target_feature]].apply(target_transform).astype(int)
            stat_test = test_stationary(dataframe[target_feature])
            stationary_tests.append({'atm':atm,'dataset':'full','score':stat_test[0]})
            models_dict[atm]['full'] = train_model(dataframe,algorithm,train_features_full)
            if time_granularity is not None:
                dataframes = [x for _, x in dataframe.groupby(time_granularity)]
                for data in dataframes:
                    key = int(data[time_granularity].iloc[0])
                    if key < 3:
                        stat_test = test_stationary(data[target_feature])
                        stationary_tests.append({'atm':atm,'dataset':key,'score':stat_test[0]})
                        models_dict[atm][key]=train_model(data,algorithm,train_features)
        except Exception as ex:
            print(f"Training error: {ex}")
            continue
    pandas.DataFrame.from_records(stationary_tests).to_csv('results/stationary.csv',sep=';',decimal=',',quotechar='"')
    
    return models_dict

def apply_model(model,data:pandas.DataFrame,avg_change:float):
    """ Performs a regression trying to estimate the amount of days before the ATM needs cash supply """
    greece_holidays = holidays.country_holidays('GR')
    time = data.index[0]
    value = float(data['value_t-1'].iloc[0])
    cur_data = data.copy()
    train_features_actual = model.features
    days = 0
    while value > 0.40:
        prev_value = value
        value = float(model.predict(cur_data[train_features_actual]))
        time:datetime.datetime = time + datetime.timedelta(days=1)

        if value > prev_value:
            return (days,cur_data)
        elif value == prev_value:
            value = prev_value+avg_change

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

def initiate_evaluation_datasets():
    """ Initiates the datasets needed for the evaluation """
    # ======= Load Full Training data =======
    start_timer,_ = timer()
    atmomore_handler = DataHandler()
    supply_info = atmomore_handler.get_supply_info()
    supply_timer,_ = timer(start=start_timer,label="Got Supply Info:")
    atmomore_preprocessor = Preprocessor(supply_info)
    atmomore_preprocessor.clean_supply_types()
    clean_timer,_ = timer(start=supply_timer,label="Cleansed info:")
    supply_info = atmomore_preprocessor.clean_data
    timeseries = atmomore_preprocessor.create_load_timeseries()
    grouped_timeseries = Preprocessor.group_by('ATM',timeseries)
    timeline_timer,_ = timer(start=clean_timer,label="Train Timeline created:")
    atm_codes = list(grouped_timeseries.keys())
    #print(atm_codes)

    # ======= Load Test data ======= 
    supply_info_test = atmomore_handler.import_test_supply_info()
    atmomore_preprocessor_test = Preprocessor(supply_info_test)
    atmomore_preprocessor_test.clean_supply_types()
    supply_info_test = atmomore_preprocessor_test.clean_data
    timeseries_test = atmomore_preprocessor_test.create_load_timeseries(coverage_threshold=0.0)
    test_timeseries = Preprocessor.group_by('ATM',timeseries_test)
    test_timeline_timer,_ = timer(start=timeline_timer,label="Test Timeline created:")

    # ======= Create evaluation training data, subset of training data ======= 
    common_codes = [atm for atm in atm_codes if atm in test_timeseries]
    if not limiter is None:
        common_codes = common_codes[:limiter]
    train_timeseries = {atm:grouped_timeseries[atm] for atm in common_codes}

    return {
        'train_timeseries':train_timeseries,
        'test_timeseries':test_timeseries
    }

def initiate_training_datasets():
    """ Initiates the datasets needed for the training of models """
    # ======= Load Full Training data =======
    start_timer,_ = timer()
    atmomore_handler = DataHandler()
    supply_info = atmomore_handler.get_supply_info()
    supply_timer,_ = timer(start=start_timer,label="Got Supply Info:")
    atmomore_preprocessor = Preprocessor(supply_info)
    atmomore_preprocessor.clean_supply_types()
    clean_timer,_ = timer(start=supply_timer,label="Cleansed info:")
    supply_info = atmomore_preprocessor.clean_data
    timeseries = atmomore_preprocessor.create_load_timeseries()
    grouped_timeseries = Preprocessor.group_by('ATM',timeseries)
    timer(start=clean_timer,label="Train Timeline created:")

    return grouped_timeseries

def initiate_metrics():
    # ======= Set up metrics and stats =======  
    metrics = {
        'r2':{'func':r2_score,'scores':[],'keys':[]},
        'mean_absolute_error':{'func':mean_absolute_error,'scores':[],'keys':[]},
        'mean_squared_error':{'func':mean_squared_error,'scores':[],'keys':[]},
        'mean_squared_log_error':{'func':mean_squared_log_error,'scores':[],'keys':[],'converter':abs},
        'mean_absolute_percentage_error':{'func':mean_absolute_percentage_error,'scores':[],'keys':[]},
        'median_absolute_error':{'func':median_absolute_error,'scores':[],'keys':[]},
        'max_error':{'func':max_error,'scores':[],'keys':[]},
        'explained_variance_score':{'func':explained_variance_score,'scores':[],'keys':[]},
        'mean_accuracy':{'func':lambda y_true,y_pred,X,model:model.score(X,y_true),'scores':[],'keys':[]}
    }
    stats = {
        'sample_size':{'func':lambda y_sample,y_test: len(y_sample),'scores':[],'keys':[]},
        'test_size':{'func':lambda y_sample,y_test: len(y_test),'scores':[],'keys':[]},
        'average_test':{'func':lambda y_sample,y_test: sum(y_test)/len(y_test),'scores':[],'keys':[]},
        'average_sample':{'func':lambda y_sample,y_test: sum(y_sample)/len(y_sample),'scores':[],'keys':[]},
        'mean_test':{'func':lambda y_sample,y_test: mean(y_test),'scores':[],'keys':[]},
        'mean_sample':{'func':lambda y_sample,y_test: mean(y_sample),'scores':[],'keys':[]},
        'std_test':{'func':lambda y_sample,y_test: stdev(y_test),'scores':[],'keys':[]},
        'std_sample':{'func':lambda y_sample,y_test: stdev(y_sample),'scores':[],'keys':[]},
        'max_test':{'func':lambda y_sample,y_test: max(y_test),'scores':[],'keys':[]},
        'max_sample':{'func':lambda y_sample,y_test: max(y_sample),'scores':[],'keys':[]},
        'min_test':{'func':lambda y_sample,y_test: min(y_test),'scores':[],'keys':[]},
        'min_sample':{'func':lambda y_sample,y_test: min(y_sample),'scores':[],'keys':[]},
    }

    return {
        'metrics':metrics,
        'stats':stats
    }

def evaluate_models_timelag(algorithm,train_timeseries,test_timeseries,lag_days=1) -> pandas.DataFrame:
    """ Train and evaluate models using the timelaged supervised data format as input features """
    # ======= Initiate evaluation parameters ======= 
    start_timer,_ = timer()
    met_stats = initiate_metrics()
    metrics = met_stats['metrics']
    stats = met_stats['stats']
    metric_keys_directory = set()

    train_dataset_dict = Preprocessor.timeseries_to_supervised(train_timeseries,lag_days)
    test_dataset_dict = Preprocessor.timeseries_to_supervised(test_timeseries,lag_days)
    common_atms = [atm for atm in train_dataset_dict if atm in test_dataset_dict]
    if not limiter is None:
        common_atms = common_atms[:limiter]

    index = 0
    for atm in common_atms:
        train_dataset = train_dataset_dict[atm]
        test_dataset = test_dataset_dict[atm]
        train_features_actual = []
        for lag in range(1,lag_days+1):
            train_features_actual += [f"{feature}_t-{lag}" for feature in train_features_full]

        model = train_model(train_dataset,algorithm,train_features=train_features_actual,target_feature=class_feature_lstm)
        Path(f"results/plots/{atm}/full").mkdir(parents=True, exist_ok=True)          
        predictions = model.predict(test_dataset[train_features_actual])
        y_pred = [float(score) for score in predictions]
        y_true = [float(datum) for datum in test_dataset[class_feature_lstm]]
        metrics['mean_accuracy']['params'] = {'model':model,'X':test_dataset}
        y_true_sample = [float(datum) for datum in train_dataset[class_feature_lstm]]
        plt.figure(figsize=(15, 10))  # Adjust the figure size as needed
        plt.plot(test_dataset.index,y_pred, label = "pred", linestyle="-")
        plt.plot(test_dataset.index,y_true, label = "true", linestyle="--")
        plt.xlabel('time')
        plt.ylabel(class_feature_lstm)
        plt.legend(loc="lower right") 
        plt.grid(True)
        plt.savefig(f'results/plots/{atm}/full/plot_{algorithm}_{atm}_full.png')
        plt.close()
        for metric in metrics:
            try:
                y_pred_copy = [metrics[metric].get('converter',float)(score) for score in y_pred]
                y_true_copy = [metrics[metric].get('converter',float)(score) for score in y_true]
                metrics[metric]['scores'].append(
                    metrics[metric]['func'](
                        y_true=y_true_copy,
                        y_pred=y_pred_copy,
                        **metrics[metric].get('params',{})
                    )
                )
                metrics[metric]['keys'].append(f'{atm}_full')
                metric_keys_directory.add(f'{atm}_full')
            except Exception as ex:
                metrics[metric]['scores'].append(None)
                metrics[metric]['keys'].append(f'{atm}_full')
                metric_keys_directory.add(f'{atm}_full')
                print(ex)
        for stat in stats:
            try:
                y_test = [stats[stat].get('converter',float)(score) for score in y_true]
                y_sample = [stats[stat].get('converter',float)(score) for score in y_true_sample]
                stats[stat]['scores'].append(
                    stats[stat]['func'](
                        y_test=y_test,
                        y_sample=y_sample,
                        **stats[stat].get('params',{})
                    )
                )
                stats[stat]['keys'].append(f'{atm}_full')
                metric_keys_directory.add(f'{atm}_full')
            except Exception as ex:
                stats[stat]['scores'].append(None)
                stats[stat]['keys'].append(f'{atm}_full')
                metric_keys_directory.add(f'{atm}_full')
                print(ex)
        index += 1
        printProgressBar(index,len(common_atms))
    
    # ======= Format Results ======= 
    results_formated = []
    for metric_key in metric_keys_directory:
        res = {'key':metric_key}
        for metric in metrics:
            key_index = metrics[metric]['keys'].index(metric_key)
            if key_index >= 0:
                res[metric] = metrics[metric]['scores'][key_index]
        for stat in stats:
            key_index = stats[stat]['keys'].index(metric_key)
            if key_index >= 0:
                res[stat] = stats[stat]['scores'][key_index]
        results_formated.append(res)
    result = pandas.DataFrame.from_records(results_formated)
    timer(start=start_timer,label="Total time.")
    return result


def evaluate_models(algorithm,train_timeseries,test_timeseries):
    """ Trains and evaluates the models using current data as input features """
    # ======= Initiate evaluation parameters ======= 
    start_timer,_ = timer()
    met_stats = initiate_metrics()
    metrics = met_stats['metrics']
    stats = met_stats['stats']
    metric_keys_directory = set()

    # ======= Train models ======= 
    models = train_models(train_timeseries,train_granularity,algorithm=algorithm)
    model_timer,_ = timer(start=start_timer,label="Models trained:")

    # ======= Run evaluation ======= 
    index = 0
    for atm in models:
        try:
            atm_models = models[atm]

            # ---- Get train data ----
            full_data_train = pandas.DataFrame.from_records(train_timeseries[atm]).sort_values('timestamp').reset_index()
            full_data_train[[target_feature]] = full_data_train[[target_feature]].apply(target_transform).astype(int)
            train_data_splits_dict = {'full':full_data_train}
            for split in [x for _, x in train_data_splits_dict['full'].groupby(train_granularity)]:
                key = int(split[train_granularity].iloc[0])
                train_data_splits_dict[key] = split
            
            # ---- Get test data ----
            full_data_test = pandas.DataFrame.from_records(test_timeseries[atm]).sort_values(by='timestamp').reset_index()
            full_data_test['timestamp'] = pandas.to_datetime(full_data_test['timestamp'], unit='s')
            full_data_test[[target_feature]] = full_data_test[[target_feature]].apply(target_transform).astype(int)
            data_tests = {'full':full_data_test}
            for split in [x for _, x in data_tests['full'].groupby(train_granularity)]:
                key = int(split[train_granularity].iloc[0])
                if key in atm_models:
                    data_tests[key] = split
            
            # ---- Get predictions ----
            predictions = {}
            for test in data_tests:  
                Path(f"results/plots/{atm}/{test}").mkdir(parents=True, exist_ok=True)          
                predictions[test] = atm_models[test].predict(data_tests[test])
                y_pred = [float(score) for score in predictions[test]]
                y_true = [float(datum) for datum in data_tests[test][target_feature]]
                metrics['mean_accuracy']['params'] = {'model':atm_models[test],'X':data_tests[test]}
                y_true_sample = [float(datum) for datum in train_data_splits_dict[test][target_feature]]
                plt.figure(figsize=(15, 10))  # Adjust the figure size as needed
                plt.plot(data_tests[test]['timestamp'],y_pred, label = "pred", linestyle="-")
                plt.plot(data_tests[test]['timestamp'],y_true, label = "true", linestyle="--")
                plt.xlabel('time')
                plt.ylabel(target_feature)
                plt.legend(loc="lower right") 
                plt.grid(True)
                plt.savefig(f'results/plots/{atm}/{test}/plot_{algorithm}_{atm}_{test}.png')
                plt.close()
                for metric in metrics:
                    try:
                        y_pred_copy = [metrics[metric].get('converter',float)(score) for score in y_pred]
                        y_true_copy = [metrics[metric].get('converter',float)(score) for score in y_true]
                        metrics[metric]['scores'].append(
                            metrics[metric]['func'](
                                y_true=y_true_copy,
                                y_pred=y_pred_copy,
                                **metrics[metric].get('params',{})
                            )
                        )
                        metrics[metric]['keys'].append(f'{atm}_{test}')
                        metric_keys_directory.add(f'{atm}_{test}')
                    except Exception as ex:
                        metrics[metric]['scores'].append(None)
                        metrics[metric]['keys'].append(f'{atm}_{test}')
                        metric_keys_directory.add(f'{atm}_{test}')
                        print(ex)
                for stat in stats:
                    try:
                        y_test = [stats[stat].get('converter',float)(score) for score in y_true]
                        y_sample = [stats[stat].get('converter',float)(score) for score in y_true_sample]
                        stats[stat]['scores'].append(
                            stats[stat]['func'](
                                y_test=y_test,
                                y_sample=y_sample,
                                **stats[stat].get('params',{})
                            )
                        )
                        stats[stat]['keys'].append(f'{atm}_{test}')
                        metric_keys_directory.add(f'{atm}_{test}')
                    except Exception as ex:
                        stats[stat]['scores'].append(None)
                        stats[stat]['keys'].append(f'{atm}_{test}')
                        metric_keys_directory.add(f'{atm}_{test}')
                        print(ex)
        except Exception as ex:
            print(f"Evaluation error: {ex}")
            continue
        index += 1
        printProgressBar(index,len(models))
    
    # ======= Format Results ======= 
    results_formated = []
    for metric_key in metric_keys_directory:
        res = {'key':metric_key}
        for metric in metrics:
            key_index = metrics[metric]['keys'].index(metric_key)
            if key_index >= 0:
                res[metric] = metrics[metric]['scores'][key_index]
        for stat in stats:
            key_index = stats[stat]['keys'].index(metric_key)
            if key_index >= 0:
                res[stat] = stats[stat]['scores'][key_index]
        results_formated.append(res)
    result = pandas.DataFrame.from_records(results_formated)
    timer(start=model_timer,label="Evaluation completed.")
    timer(start=start_timer,label="Total time.")
    return result

def evaluate_lstm(timeseries,lag_days=1):
    train_dataset_dict = Preprocessor.timeseries_to_supervised(timeseries['train_timeseries'],lag_days)
    test_dataset_dict = Preprocessor.timeseries_to_supervised(timeseries['test_timeseries'],lag_days)
    common_atms = [atm for atm in train_dataset_dict if atm in test_dataset_dict]
    if not limiter is None:
        common_atms = common_atms[:limiter]

    results = []
    Path(f"results/figures/lstm").mkdir(parents=True, exist_ok=True)  
    for atm in common_atms:
        train_dataset = train_dataset_dict[atm]
        test_dataset = test_dataset_dict[atm]
        train_features_lstm_actual = []
        for lag in range(1,lag_days+1):
            train_features_lstm_actual += [f"{feature}_t-{lag}" for feature in train_features_lstm]
        train_X = train_dataset[train_features_lstm_actual].values.astype('float32')
        train_Y = train_dataset[class_feature_lstm].values.astype('float32')
        test_X = test_dataset[train_features_lstm_actual].values.astype('float32')
        test_Y = test_dataset[class_feature_lstm].values.astype('float32')
        # reshape input to be 3D [samples, timesteps, features]
        train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
        print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)
        # design network
        model = Sequential()
        model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')
        # fit network
        history = model.fit(train_X, train_Y, epochs=100, batch_size=72, validation_data=(test_X, test_Y), verbose=2, shuffle=False)
        # plot history
        plt.figure(figsize=(20, 10))  # Adjust the figure size as needed
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend(loc="lower right")
        plt.xlabel('steps')
        plt.ylabel('loss')
        plt.title(f'{atm} Loss/Steps chart')
        plt.grid(True)
        plt.savefig(f'results/figures/lstm/lstm_training_{atm}_{lag_days}.png')
        plt.close()
        yhat = model.predict(test_X)
        test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
        # invert scaling for forecast
        inv_yhat = numpy.concatenate((yhat, test_X[:, 1:]), axis=1)
        inv_yhat = inv_yhat[:,0]
        # invert scaling for actual
        test_Y = test_Y.reshape((len(test_Y), 1))
        inv_y = numpy.concatenate((test_Y, test_X[:, 1:]), axis=1)
        inv_y = inv_y[:,0]
        # calculate RMSE
        rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
        print('ATM %s Test RMSE: %.3f' % (atm,rmse))
        results.append({'key':atm,'rmse':rmse,'mae':mean_absolute_error(inv_y, inv_yhat)})
    return results

def evaluate_days(algorithm,train_timeseries,test_timeseries,lag_days=1):
    """ Evaluate models using the days to resupply as an error metric """

    train_dataset_dict = Preprocessor.timeseries_to_supervised(train_timeseries,lag_days)
    test_dataset_dict = Preprocessor.timeseries_to_supervised(test_timeseries,lag_days)
    common_atms = [atm for atm in train_dataset_dict if atm in test_dataset_dict]
    if not limiter is None:
        common_atms = common_atms[:limiter]

    train_features_actual = []
    for lag in range(1,lag_days+1):
        train_features_actual += [f"{feature}_t-{lag}" for feature in train_features_full]

    index = 0
    results = []
    for atm in common_atms:
        train_dataset:pandas.DataFrame = train_dataset_dict[atm]
        test_dataset:pandas.DataFrame = test_dataset_dict[atm]
        model = train_model(train_dataset,algorithm,train_features=train_features_actual,target_feature=class_feature_lstm)

        resupply_times = []
        for row_index,datum in test_dataset.iterrows():
            if float(datum['value_t-1']) == 1.0:
                resupply_times.append(row_index)
        
        for resupply_index,resupply_time in enumerate(resupply_times[:-1]):
            resupply_row = test_dataset.index.get_loc(resupply_time)
            res_days = model.apply(test_dataset[resupply_row:resupply_row+1])[0]
            actual_days = (resupply_times[resupply_index+1] - resupply_time).days
            results.append({'atm':atm,'actual':actual_days,'pred':res_days,'error':(res_days-actual_days),'error_perc':(res_days-actual_days)/actual_days})

        index += 1
        printProgressBar(index,len(common_atms))
    
    # ======= Format Results ======= 
    result = pandas.DataFrame.from_dict(results)
    return result