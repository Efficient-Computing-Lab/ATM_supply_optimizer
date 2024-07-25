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
from atm_load_prediction.models import LinearModel
from atm_load_prediction.data_handler import DataHandler, Preprocessor

class ATMOPredictor():
    """ Base class that handles ATM load predictions. 
    Optional creation parameters include a models and training data dictionaries having the atm 
    codes as keys and the model or list of dict training records as values. 
    Set autotrain to True to train the models on creation. """
    def __init__(self,models:dict={},train_data:dict=None,autotrain:bool=False):
        self.models = {}
        self.train_data = train_data
        self.train_features = ['day_of_week','day_of_month','month','workday','holiday','value']
        self.class_feature = 'value_t'
        if autotrain:
            self.train_models()

    @property
    def atm_codes(self) -> list:
        return list(self.models.keys())
    
    def load_train_data(self,atm_codes:list=None) -> None:
        """ A class that loads the training datasets from the MongoDB and stores them in 
        self.train_data. The atm_codes can be used to limit the loaded atm codes."""

        atmomore_handler = DataHandler()
        supply_info = atmomore_handler.get_supply_info()
        atmomore_preprocessor = Preprocessor(supply_info)
        atmomore_preprocessor.clean_supply_types()
        supply_info = atmomore_preprocessor.clean_data
        timeseries = atmomore_preprocessor.create_load_timeseries(atm_codes=atm_codes)
        self.train_data = Preprocessor.group_by('ATM',timeseries)
    
    def train_models(self,atm_codes:list=None) -> None:
        """ Trains the models based on the train_data provided during object creation. 
        If the train_data value is None then the full dataset is downloaded from MongoDB.
        If a list of atm codes is provided only models for the codes in the list are trained. """

        self.models = {}

        if self.train_data is None:
            self.load_train_data(atm_codes=atm_codes)
        
        train_dataset_dict = Preprocessor.timeseries_to_supervised(self.train_data,1)
        train_features_actual = [f"{feature}_t-1" for feature in self.train_features]
        for atm in self.train_data:
            train_dataset:pd.DataFrame = train_dataset_dict[atm]
            if atm not in train_dataset_dict:
                print(f"No training data for ATM {atm}. Skipping...")
                continue
            else:
                lm = LinearModel(
                    dataset=train_dataset,
                    target=self.class_feature,
                    features=train_features_actual,
                    algorithm=5
                )
                lm.train()
                self.models[atm] = lm
                
    def days_to_resupply(self,current_state:pd.DataFrame,atm_code:str=None,model:LinearModel=None) -> int:
        """ Calculate how many days are needed before the next resuply. 
        An atm code will be used to lookup for a pre-trained model in the ATMOPredictor directory. 
        If no ATM code is provided you need to provide a pre-trained model to use."""

        if model is None and atm_code is None:
            raise Exception('You need to specify either a custom model or an atm code to look '+
                            'for the model.')
        
        if model is None:
            model = self.models.get(atm_code,None)
            if model is None:
                raise Exception(f'No model found for ATM {atm_code}. '+
                                'Please train a model for this ATM.')
        
        return model.apply(current_state)[0]
    
    def is_atm_due(self,current_state:pd.DataFrame,atm_code:str=None,model:LinearModel=None) -> bool:
        """ Returns True if the ATM specified is predicted to need resupply tommorow.
        The atm code will be used to lookup for a pre-trained model in the ATMOPredictor directory. 
        If a pre-trained model is provided it will be used instead."""

        if model is None:
            model = self.models.get(atm_code,None)
            if model is None:
                raise Exception(f'No model found for ATM {atm_code}. '+
                                'Please train a model for this ATM.')

        days = model.apply(current_state)[0]
        return days <= 2