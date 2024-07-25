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

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from arch.unitroot import KPSS
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def print_confusion_matrix(true_y:pd.Series|list,pred_y:pd.Series|list,filepath:str,verbose:bool=False,labels:list=None) -> np.ndarray:
    """ Creates a confusion matrix, prints the values if verbose is true and saves the figure in 
    the provided path."""

    cm = confusion_matrix(true_y,pred_y)
    if verbose:
        print("=== Confusion matrix ===")
        print(cm)
        print("========================")
    cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=labels)
    plt.figure(figsize=(800, 800))
    cm_display.plot()
    plt.savefig(filepath)
    plt.close()
     

def plot(data:pd.DataFrame,filepath:str,charttitle:str,column:str,timecol:str='timestamp') -> None:
    """ Plots a column of the dataframe against the timestamp column and saves the figure in the 
    provided filepath """
    data['timestamp'] = pd.to_datetime(data[timecol], unit='s')
    data = data.sort_values(by=timecol)
    plt.figure(figsize=(20, 10))  # Adjust the figure size as needed
    plt.plot(data[timecol], data[column], linestyle='-')
    plt.xlabel(timecol)
    plt.ylabel(column)
    plt.title(charttitle)
    plt.grid(True)
    plt.savefig(filepath)
    plt.close()

def resampler(data:pd.DataFrame,interval='1h') -> pd.DataFrame:
    """ Resamples the data in the dataframe using a linear function in order to have uniform 
    timeslots"""
    if interval=='1h':
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
        resampled_data = data.set_index('timestamp')
        resampled_data = resampled_data.resample(interval).asfreq().interpolate(method='linear')
        resampled_data.reset_index(inplace=True)
        return resampled_data
    elif interval=='1d':
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
        min_date = data['timestamp'].min()
        max_date = data['timestamp'].max()
        full_range = pd.date_range(start=min_date, end=max_date, freq='D')
        resampled_data = pd.DataFrame(full_range, columns=['timestamp'])
        resampled_data = resampled_data.merge(data, on='timestamp', how='left')
        columns_to_interpolate = resampled_data.columns.difference(['timestamp'])
        resampled_data[columns_to_interpolate] = resampled_data[columns_to_interpolate].interpolate(method='linear')
        resampled_data = resampled_data.sort_values(by='timestamp').reset_index(drop=True)
        return resampled_data


def clear_duplicate_timestamps(data:pd.DataFrame, timecol:str='timestamp') -> pd.DataFrame:
    """ Clears duplicate records based on the time column """
    return data.drop_duplicates(subset=[timecol], keep='first')

def timer(start:float=0.0,label:str=None) -> tuple[float,float]:
     """ Returns the current timestamp and the difference between a timestamp and now """
     now = datetime.now().timestamp()
     diff = abs(start-now)
     if label is not None:
          print(f"{label} {diff}")
     return (now,diff)

def test_stationary(dataset:pd.Series):
    """ Test if the dataset provided are stationary. 
    The dataset should regard only 1 ATM. """
    # Test for stationary
    kpss_test = KPSS(dataset)

    # Test summary 
    return (kpss_test.pvalue,kpss_test.summary().as_text())

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 2, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def csv_to_supervised(filepath:str,converter=None,**converter_params) -> dict:
    """ Parse a csv file and convert it to a list of supervised timeseries evaluation datasets, 
    one dataset per split key. As converter a dataframe to supervised converter should be used. """
    dataset = pd.read_csv(filepath,sep=';',decimal='.')
    if converter is not None:
        dataset = converter(dataset,**converter_params)
    return dataset