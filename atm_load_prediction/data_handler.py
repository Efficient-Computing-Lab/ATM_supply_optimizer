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

import pymongo, json, pandas, xlrd, os
from openpyxl import load_workbook
from datetime import datetime
import statistics
import holidays
from atm_load_prediction.utils import printProgressBar, resampler, clear_duplicate_timestamps, timer


class DataHandler():
    """ Class that handles the import, export, read and write of data for ATMO-MoRe """
    
    def __init__(self):
        self.db_conf = {
            'host':os.environ.get('MONGO_HOST', 'ai-mongodb-module'),
            'port':int(os.environ.get('MONGO_PORT', '27017')),
            'username':os.environ.get('MONGO_USERNAME', 'atmomore'),
            'password':os.environ.get('MONGO_PASSWORD', 'Atmo-more'),
            'authSource':os.environ.get('MONGO_AUTH_DB', 'admin'),
            'authMechanism':'SCRAM-SHA-256'
        }
        self.config = {
            'db':os.environ.get('MONGO_DATABASE', 'atmo_more'),
            'datastore_path':os.environ.get('DATASTORE_PATH', 'datastore'),
            'datastore_supply_path':os.environ.get('DATASTORE_SUPPLY_INFO', 'datastore/ΕΦΟΔΙΑΣΜΟΙ.xlsx'),
            'datastore_supply_test_path':os.environ.get('DATASTORE_SUPPLY_TEST_INFO', 'datastore/ΕΦΟΔΙΑΣΜΟΙ_test.xlsx'),
            'datastore_outage_path':os.environ.get('DATASTORE_OUTAGE_INFO', 'datastore/outage'),
        }

    @property
    def mongo(self) ->pymongo.MongoClient:
        client = pymongo.MongoClient(**self.db_conf)
        return client[self.config['db']]
    
    @property
    def supply_types(self) -> list:
        data = list(self.mongo['supply_info'].aggregate(
            [
                {
                    "$match": {
                        'lat':{'$ne':None},
                        'lon':{'$ne':None},
                    }
                },
                {
                    "$group":{
                        "_id":"$Service Type"
                    }
                },
                {
                    "$project":{
                        "_id": 0,
                        "Service Type": "$_id"
                    }
                }
            ]
        ))
        return [datum['Service Type'] for datum in data]
    
    @property
    def atm_codes(self) -> list:
        data = list(self.mongo['supply_info'].aggregate(
            [
                {
                    "$match": {
                        'lat':{'$ne':None},
                        'lon':{'$ne':None},
                    }
                },
                {
                    "$group":{
                        "_id":"$ATM"
                    }
                },
                {
                    "$project":{
                        "_id": 0,
                        "ATM": "$_id"
                    }
                }
            ]
        ))
        return [datum['ATM'] for datum in data]
    
    def import_atm_info(self,filepath=None) -> list:
        """#DEPRECATED# Reads atm info from file and inserts it into mongo """
        atm_info = []
        if filepath is None:
            filepath = self.config['datastore_path']+'/atmomore/atm_info.json'
        with open(filepath,'r',encoding='utf-8') as f_in:
            atm_info = json.load(f_in)
        self.mongo.drop_collection('atm_info')
        self.mongo['atm_info'].insert_many(atm_info)
        return atm_info
        
    def import_supply_info(self,filepath=None,atm_info=None) -> list:
        """ Reads atm supply info from file and inserts it into mongo """
        if filepath is None:
            filepath = self.config['datastore_supply_path']
        supply_info = pandas.ExcelFile(filepath).parse(sheet_name='Sheet1').to_dict(orient='records')
        for record in supply_info:
            if isinstance(record['Date'],str):
                record['Date'] = datetime.strptime(record['Date'], "%d/%m/%Y")
            record['lat'] = float(record.pop('Latitude',None))
            record['lon'] = float(record.pop('Longitude',None))
            if atm_info and ( record['lat'] is None or record['lon'] is None):
                for row in atm_info:
                    if record['ATM'] == row['Code']:
                        record['lat'] = row['Latitude']
                        record['lon'] = row['Longitude']
                        break
        self.mongo.drop_collection('supply_info')
        self.mongo['supply_info'].insert_many(supply_info)
        return supply_info
    
    def import_test_supply_info(self,filepath=None,atm_info=None) -> list:
        """ Reads atm supply info from file and inserts it into mongo """
        if filepath is None:
            filepath = self.config['datastore_supply_test_path']
        supply_info = pandas.ExcelFile(filepath).parse(sheet_name='Sheet1').to_dict(orient='records')
        for record in supply_info:
            if isinstance(record['Date'],str):
                record['Date'] = datetime.strptime(record['Date'], "%d/%m/%Y")
            record['lat'] = float(record.pop('Latitude',None))
            record['lon'] = float(record.pop('Longitude',None))
            if atm_info and ( record['lat'] is None or record['lon'] is None):
                for row in atm_info:
                    if record['ATM'] == row['Code']:
                        record['lat'] = row['Latitude']
                        record['lon'] = row['Longitude']
                        break
        self.mongo.drop_collection('supply_info_test')
        self.mongo['supply_info_test'].insert_many(supply_info)
        return supply_info
    
    def import_outage_info(self,filepath=None,atm_info=None) -> list:
        """ Reads atm outage info from file and inserts it into mongo """
        if filepath is None:
            filepath = self.config['datastore_outage_path']
        outage_info = []
        filepaths = []
        [filepaths.extend([x[0]+'/'+filepath for filepath in x[2]]) for x in os.walk(filepath) if 'off-site' in x[0]]
        for file in filepaths[:1]:
            start_row = 0
            end_row = 0
            if '.xlsx' in file:
                wb = load_workbook(file)
                ws = wb.active
                for index,row in enumerate(ws.rows):
                    if row[2].value == "Primary ID":
                        start_row=index
                    if row[2] == "Mean":
                        end_row=index-1
                        break
            elif '.xls' in file:
                wb = xlrd.open_workbook(file)
                ws = wb.sheet_by_index(0)
                for index in range(ws.nrows):
                    row_value = ws.row_values(index)
                    if row_value[2] == "Primary ID":
                        start_row=index
                    if row_value[2] == "Mean":
                        end_row=index-1
                        break
            input_df = pandas.ExcelFile(file).parse(skiprows=start_row,nrows=end_row-start_row)
            input_df = input_df.loc[:, ~input_df.columns.str.contains('^Unnamed')]
            input_df = input_df.rename(columns={
                'Primary ID':'ATM', 
                'Location':'address', 
                'In \n Service(%)':'in_service', 
                'Out of \n Service(%)':'out_of_service',
                'Hard \n Faults(%)':'hard_faults', 
                'Vandal- \n ism(%)':'vandalism', 
                'Supply \n Out(%)':'supply_out',
                'Cash \n Out(%)':'cash_out', 
                'Comms(%)':'comms', 
                'Host \n Down(%)':'host_down', 
                'Daily \n Balance(%)':'daily_balance',
                'Main- \n tenance(%)':'maintenance'
            })
            outage_info = input_df.to_dict(orient='records')
            for record in outage_info:
                record['lat'] = None
                record['lon'] = None
                if atm_info:
                    for row in atm_info:
                        if record['ATM'] == row['Code']:
                            record['lat'] = row['Latitude']
                            record['lon'] = row['Longitude']
                            break
        self.mongo.drop_collection('outage_info')
        self.mongo['outage_info'].insert_many(outage_info)
        return outage_info
    
    def get_supply_info(self,filters={},aggregation=None):
        if aggregation:
            return list(self.mongo['supply_info'].aggregate(aggregation))
        else:
            return list(self.mongo['supply_info'].find(filters))
    
    def get_atm_info(self) -> dict:
        return list(self.mongo['supply_info'].aggregate([
            {
                "$group":{
                    "_id":"$ATM",
                    'lat':{"$first": "$lat"},
                    'lon':{"$first": "$lon"}
                }
            },
            {
                "$project":{
                    "_id": 0,
                    "ATM": "$_id",
                    "lat": 1,
                    "lon":1
                }
            }
        ]))
    
    def atm_renew_data(self):
        """ Loads or refreshes the atmo more data in the database """
        #self.import_atm_info()
        try:
            self.import_supply_info()
        except Exception as ex:
            print(f"Data import failed: {ex}")
        try:
            self.import_test_supply_info()
        except Exception as ex:
            print(f"Data import failed: {ex}")
        try:
            self.import_outage_info()
        except Exception as ex:
            print(f"Data import failed: {ex}")

class Preprocessor():
    """ Basic preprocessor class for atmo more data """
    replaces = {
        'ΑΝΕΦ/ΣΜΟΣ ΑΤΜ':'supply',
        'ΕΦΟΔΙΑΣΜΟΣ':'supply',  
        'Εφοδιασμός ΑΤΜ':'supply',
        'Ανεφοδιασμός /Αποσυμφόρηση':'supply', 
        'ΕΚΤΑΚΤΟΣ ΕΦΟΔΙΑΣΜΟΣ':'cashout',
        'Έκτακτος Ανεφοδιασμός / Αποσυμφόρηση ':'cashout', 
        'Έκτακτος Ανεφοδιασμός /Αποσυμφόρηση':'cashout', 
        'Ανεφοδιασμός / Αποσυμφόρηση ':'supply',
        'ΣΥΝΔΥΑΣΤΙΚΟΣ ΕΦΟΔ.':'supply'
    }

    ignored = [
        'Αποσυμφόρηση ΑΤΜ', 
        'ΚΟΣΤΟΣ ΔΙΑΝΥΚΤΕΡΕΥΣΗΣ', 
        'ΑΠΟΣΥΜΦΟΡΗΣΗ'
    ]

    def __init__(self,dataset=None):
        self.dataset = dataset

    def group_by(field:str,data:list) -> dict:
        """ Groups the dataset by the specified filed and returns a dict having the field values as keys """
        grouped = {}
        for datum in data:
            if not datum[field] in grouped:
                grouped[datum[field]] = []
            new_datum = {key:datum[key] for key in datum if key != field}
            grouped[datum[field]].append(new_datum)
        return grouped
    
    def degroup(field:str,data:dict) -> list:
        """ Reverses the group_by function """
        degrouped = []
        for key in data:
            key_data = data[key]
            for datum in key_data:
                datum[field] = key
                degrouped.append(datum)
        return degrouped

    def clean_supply_types(self):
        """ Cleans the supply types based on the replaces and ignored set """
        self.clean_data = []
        for datum in self.dataset:
            if datum['Service Type'] not in self.ignored:
                datum['Service Type'] = self.replaces.get(datum['Service Type'],datum['Service Type'])
                self.clean_data.append(datum)
        return self.clean_data
    
    def apply_coverage_threshold(self,threshold=0.7,data=None) -> list:
        """ Removes the atm data that do not achieve the specified threshold in coverage of days / year """
        validated_data = {}
        atm_grouped = Preprocessor.group_by('ATM',data=data)
        print(f"coverage tested: {len(atm_grouped)}")
        for atm in atm_grouped:
            atm_data = sorted(atm_grouped[atm],key= lambda x:x['Date'])
            coverage = round((atm_data[-1]['Date'] - atm_data[0]['Date']).days / pandas.Timestamp(atm_data[-1]['Date'].year,12,31).day_of_year,4)
            if coverage >= threshold:
                validated_data[atm] = atm_data
        print(f"coverage made it: {len(validated_data)}")
        return Preprocessor.degroup('ATM',validated_data)
    
    def apply_samples_threshold(self,threshold=10,data=None) -> list:
        """ Removes the atm data that do not achieve the specified threshold in number of samples """
        validated_data = {}
        atm_grouped = Preprocessor.group_by('ATM',data=data)
        print(f"samples tested: {len(atm_grouped)}")
        for atm in atm_grouped:
            atm_data = atm_grouped[atm]
            samples = len(atm_data)
            if samples >= threshold:
                validated_data[atm] = atm_data
        print(f"samples made it: {len(validated_data)}")
        return Preprocessor.degroup('ATM',validated_data)
    
    def create_load_timeseries(self,skip:int=0,size:int=0,coverage_threshold:float=0.7,atm_codes:list=None) -> list:
        """ Creates the load timeseries per ATM on daily basis after applying the thresholds and pre-processing """
        dataset = self.apply_samples_threshold(threshold=20,data=self.clean_data)
        dataset = self.apply_coverage_threshold(data=dataset,threshold=coverage_threshold)
        greece_holidays = holidays.country_holidays('GR')
        atm_grouped = Preprocessor.group_by('ATM',dataset)
        timeseries_set = {key:[] for key in atm_grouped}
        samples = list(atm_grouped.keys())
        if atm_codes is not None and len(atm_codes) > 0:
            samples = [atm for atm in samples if atm in atm_codes]
        if skip > 0:
            samples = samples[skip:]
        if size > 0:
            samples = samples[:size]
        print('Creating timeseries...')
        for atm_index,atm in enumerate(samples):
            if atm not in timeseries_set:
                timeseries_set[atm] = []
            atm_data = sorted(atm_grouped[atm],key= lambda x:x['Date'])
            timeseries = []
            for atm_datum in atm_data:
                timestamp_current = int(datetime.timestamp(atm_datum['Date']))
                timeslot_current = datetime.fromtimestamp(timestamp_current)
                timeslot_current = timeslot_current.replace(hour=12, minute=0, second=0)
                timestamp_current = int(timeslot_current.timestamp())
                timestamp_before = timestamp_current - (3600*24)
                if atm_datum['Service Type'] == 'cashout':
                    timeseries.append({
                        'timestamp': timestamp_before,
                        'value':0.0
                    })
                else:
                    timeseries.append({
                        'timestamp':timestamp_before,
                        'value':0.20
                    })
                timeseries.append({
                    'timestamp':timestamp_current,
                    'value':1.0
                })

            resampled = resampler(clear_duplicate_timestamps(pandas.DataFrame.from_records(timeseries)),interval='1d')
            timeseries_set[atm] = resampled.to_dict(orient='records')
            timeseries_set[atm][0]['change'] = 0.0
            days_elapsed = []
            for index in range(len(timeseries_set[atm])):
                datum_time = timeseries_set[atm][index]['timestamp']
                timeseries_set[atm][index]['day_of_week'] = datum_time.weekday()
                timeseries_set[atm][index]['monday'] = datum_time.weekday() == 0
                timeseries_set[atm][index]['tuesday'] = datum_time.weekday() == 1
                timeseries_set[atm][index]['wednesday'] = datum_time.weekday() == 2
                timeseries_set[atm][index]['thursday'] = datum_time.weekday() == 3
                timeseries_set[atm][index]['friday'] = datum_time.weekday() == 4
                timeseries_set[atm][index]['saturday'] = datum_time.weekday() == 5
                timeseries_set[atm][index]['sunday'] = datum_time.weekday() == 6
                timeseries_set[atm][index]['workday'] = datum_time.weekday() < 5
                timeseries_set[atm][index]['holiday'] = int(datetime.timestamp(datum_time)) in greece_holidays
                timeseries_set[atm][index]['month'] = datum_time.month
                timeseries_set[atm][index]['day_of_month'] = datum_time.day
                if timeseries_set[atm][index]['value'] == 1.0:
                    days_elapsed.append(index)
                    for day in days_elapsed:
                        timeseries_set[atm][day]['lifetime'] = (
                            timeseries_set[atm][days_elapsed[-1]]['timestamp'] - \
                            timeseries_set[atm][days_elapsed[0]]['timestamp']
                        ).days
                    days_elapsed = []
                else:
                    days_elapsed.append(index)
                if index == 0:
                    timeseries_set[atm][index]['change'] = 0.0
                else:
                    timeseries_set[atm][index]['change'] = timeseries_set[atm][index]['value'] - timeseries_set[atm][index - 1]['value']
                    if timeseries_set[atm][index]['change'] > 0:
                        timeseries_set[atm][index]['change'] = 0.0
            for day in days_elapsed:
                timeseries_set[atm][day]['lifetime'] = timeseries_set[atm][day]['lifetime'] = (
                    timeseries_set[atm][days_elapsed[-1]]['timestamp'] - \
                    timeseries_set[atm][days_elapsed[0]]['timestamp']
                ).days + 1
                days_elapsed = []
            last_supply = timeseries_set[atm][-1]['timestamp']
            for index in reversed(list(range(len((timeseries_set[atm]))))):
                timeseries_set[atm][index]['lifespan'] = (last_supply - timeseries_set[atm][index]['timestamp']).days
                if timeseries_set[atm][index]['value'] == 1.0:
                    last_supply = timeseries_set[atm][index]['timestamp']
                if timeseries_set[atm][index]['lifetime'] > 4:
                    timeseries_set[atm][index]['lifeclass'] = 4
                else:
                    timeseries_set[atm][index]['lifeclass'] = timeseries_set[atm][index]['lifetime']
            printProgressBar(atm_index,len(samples),suffix=atm)
        return Preprocessor.degroup(data=timeseries_set,field='ATM')
    
    def create_load_timeseries_hour(self,skip=0,size=0,coverage_threshold=0.7,previous_slot=3600,resampling="1h") -> list:
        """ Creates the load timeseries per ATM on hourly basis after applying the thresholds and pre-processing """
        dataset = self.apply_coverage_threshold(data=self.clean_data,threshold=coverage_threshold)
        dataset = self.apply_samples_threshold(threshold=20,data=dataset)
        greece_holidays = holidays.country_holidays('GR')
        atm_grouped = Preprocessor.group_by('ATM',dataset)
        timeseries_set = {key:[] for key in atm_grouped}
        samples = list(atm_grouped.keys())
        if skip > 0:
            samples = samples[skip:]
        if size > 0:
            samples = samples[:size]
        for atm_index,atm in enumerate(samples):
            if atm not in timeseries_set:
                timeseries_set[atm] = []
            atm_data = sorted(atm_grouped[atm],key= lambda x:x['Date'])
            timeseries = []
            for atm_datum in atm_data:
                timestamp_current = int(datetime.timestamp(atm_datum['Date']))
                timeslot_before = datetime.fromtimestamp( timestamp_current - previous_slot)
                if atm_datum['Service Type'] == 'cashout':
                    timeseries.append({
                        'timestamp': datetime.timestamp(timeslot_before),
                        'value':0.0
                    })
                else:
                    timeseries.append({
                        'timestamp':datetime.timestamp(timeslot_before),
                        'value':0.15
                    })
                timeseries.append({
                    'timestamp':timestamp_current,
                    'value':1.0
                })
            resampled = resampler(clear_duplicate_timestamps(pandas.DataFrame.from_records(timeseries)),interval=resampling)
            timeseries_set[atm] = resampled.to_dict(orient='records')
            timeseries_set[atm][0]['change'] = 0.0
            days_elapsed = []
            for index,datum in enumerate(timeseries_set[atm]):
                datum_time = timeseries_set[atm][index]['timestamp']
                timeseries_set[atm][index]['day_of_week'] = datum_time.weekday()
                timeseries_set[atm][index]['monday'] = datum_time.weekday() == 0
                timeseries_set[atm][index]['tuesday'] = datum_time.weekday() == 1
                timeseries_set[atm][index]['wednesday'] = datum_time.weekday() == 2
                timeseries_set[atm][index]['thursday'] = datum_time.weekday() == 3
                timeseries_set[atm][index]['friday'] = datum_time.weekday() == 4
                timeseries_set[atm][index]['saturday'] = datum_time.weekday() == 5
                timeseries_set[atm][index]['sunday'] = datum_time.weekday() == 6
                timeseries_set[atm][index]['workday'] = datum_time.weekday() < 5
                timeseries_set[atm][index]['holiday'] = int(datetime.timestamp(datum_time)) in greece_holidays
                timeseries_set[atm][index]['hour'] = datum_time.hour
                timeseries_set[atm][index]['month'] = datum_time.month
                timeseries_set[atm][index]['day_of_month'] = datum_time.day
                if timeseries_set[atm][index]['value'] == 1.0:
                    days_elapsed.append(index)
                    for day in days_elapsed:
                        timeseries_set[atm][day]['lifetime'] = (
                            timeseries_set[atm][days_elapsed[-1]]['timestamp'] - \
                            timeseries_set[atm][days_elapsed[0]]['timestamp']
                        ).days
                    days_elapsed = []
                else:
                    days_elapsed.append(index)
                if index == 0:
                    timeseries_set[atm][index]['change'] = 0.0
                else:
                    timeseries_set[atm][index]['change'] = timeseries_set[atm][index]['value'] - timeseries_set[atm][index - 1]['value']
                    if timeseries_set[atm][index]['change'] > 0:
                        timeseries_set[atm][index]['change'] = 0.0
            for day in days_elapsed:
                timeseries_set[atm][day]['lifetime'] = timeseries_set[atm][day]['lifetime'] = (
                    timeseries_set[atm][days_elapsed[-1]]['timestamp'] - \
                    timeseries_set[atm][days_elapsed[0]]['timestamp']
                ).days + 1
                days_elapsed = []
            last_supply = timeseries_set[atm][-1]['timestamp']
            for index,datum in reversed(list(enumerate(timeseries_set[atm]))):
                timeseries_set[atm][index]['lifespan'] = (last_supply - timeseries_set[atm][index]['timestamp']).days
                if timeseries_set[atm][index]['value'] == 1.0:
                    last_supply = timeseries_set[atm][index]['timestamp']
                if timeseries_set[atm][index]['lifetime'] > 4:
                    timeseries_set[atm][index]['lifeclass'] = 4
                else:
                    timeseries_set[atm][index]['lifeclass'] = timeseries_set[atm][index]['lifetime']
            printProgressBar(atm_index,len(samples),suffix=atm)
        return Preprocessor.degroup(data=timeseries_set,field='ATM')
    
    def timeseries_to_supervised(timeseries:list[dict]|dict,lag_days:int=1) -> dict:
        """ Convert the timeseries from create_load_timeseries to a supervised learning dataset """
        if type(timeseries) == list:
            timeseries = Preprocessor.group_by('ATM',timeseries)

        supervised_data_dict = {}
        for atm in timeseries:
            timeseries_df = pandas.DataFrame.from_records(timeseries[atm]).set_index('timestamp').sort_index()
            col_names = [name for name in list(timeseries_df) if name != 'timestamp']
            cols, names = list(), list()
            for i in range(lag_days, 0, -1):
                cols.append(timeseries_df.shift(i))
                names += [f'{name}_t-{i}' for name in col_names]
            cols.append(timeseries_df)
            names += [f'{name}_t' for name in col_names]
            agg = pandas.concat(cols, axis=1).astype('float32')
            agg.columns = names
            agg.dropna(inplace=True)
            supervised_data_dict[atm] = agg
        return supervised_data_dict
    
    def test_to_supervised(simple_data:pandas.DataFrame,train_features:list) -> dict:
        """ Convert simple test data [ATM, date, value] to full supervised format and split 
        by ATM """

        try:
            simple_data['date'] = simple_data['date'].apply(lambda x: datetime.strptime(x, "%d/%m/%Y"))
        except:
            pass
        simple_data_dict = simple_data.to_dict(orient='records')
        grouped_data = Preprocessor.group_by('ATM',data=simple_data_dict)
        greece_holidays = holidays.country_holidays('GR')
        supervised_data = {}
        for atm in grouped_data:
            supervised_data_list = []
            for rec in grouped_data[atm]:
                datum_time = rec['date']
                rec['day_of_week_t-1'] = datum_time.weekday()
                rec['monday_t-1'] = datum_time.weekday() == 0
                rec['tuesday_t-1'] = datum_time.weekday() == 1
                rec['wednesday_t-1'] = datum_time.weekday() == 2
                rec['thursday_t-1'] = datum_time.weekday() == 3
                rec['friday_t-1'] = datum_time.weekday() == 4
                rec['saturday_t-1'] = datum_time.weekday() == 5
                rec['sunday_t-1'] = datum_time.weekday() == 6
                rec['workday_t-1'] = datum_time.weekday() < 5
                rec['holiday_t-1'] = int(datetime.timestamp(datum_time)) in greece_holidays
                rec['hour_t-1'] = datum_time.hour
                rec['month_t-1'] = datum_time.month
                rec['day_of_month_t-1'] = datum_time.day
                rec['value_t-1'] = rec['value']
                rec['timestamp_t-1'] = rec['date']
                supervised_data_list.append(rec)
            supervised_data[atm] = pandas.DataFrame.from_records(supervised_data_list).set_index('timestamp_t-1').sort_index()
            train_features_actual = [f"{feat}_t-1" for feat in train_features]
            supervised_data[atm][train_features_actual] = supervised_data[atm][train_features_actual].astype('float32')
        return supervised_data
    
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
    
class Analyser():
    """ Class that performs basic analysis on data """
    def __init__(self,dataset):
        self.dataset = dataset
    
    def unique(self,field:str,data=None) -> list:
        """ Get unique values of a field """
        grouped = {}
        if data is None:
            data = self.dataset
        for datum in data:
            if not datum[field] in grouped:
                grouped[datum[field]] = None
        return grouped
    
    def select(self,field:str,values:list,data=None) -> list:
        """ Select values from the dataset """
        if data is None:
            data = self.dataset
        results = []
        for datum in data:
            if datum[field] in values:
                results.append(datum)
        return results
    
    def supply_type_statistics(self) -> pandas.DataFrame:
        """ Get basic statistics about the Service Type values """
        types = list(self.unique('Service Type').keys())
        atm_grouped = Preprocessor.group_by('ATM',self.dataset)
        results = []
        for atm in atm_grouped:
            val = Preprocessor.group_by('Service Type',atm_grouped[atm])
            res = {'atm':atm,'total':len(atm_grouped[atm])}
            for type in types:
                res[type] = 0
            for key in val:
                res[key] = len(val[key])
            results.append(res)
        return pandas.DataFrame.from_records(results)
    
    def timeseries_statistics(self,key_field='ATM',date_field='Date',date_interval='day') -> pandas.DataFrame:
        """ Get basic statistics about the timeseries per ATM """
        atm_grouped = Preprocessor.group_by(key_field,self.dataset)
        results = []
        for atm in atm_grouped:
            atm_stats = []
            atm_data = sorted(atm_grouped[atm],key= lambda x:x[date_field])
            prev_date = None
            for atm_datum in atm_data:
                if prev_date:
                    if date_interval == 'day':
                        atm_stats.append((atm_datum[date_field] - prev_date).days)
                    elif date_interval == 'hour':
                        atm_stats.append((atm_datum[date_field] - prev_date).seconds // 3600)
                prev_date = atm_datum[date_field] 
            if len(atm_stats) > 1:
                average = sum(atm_stats)/len(atm_stats)
                median = statistics.median(atm_stats)
                stdev = statistics.stdev(atm_stats)
                total_duration = None
                if date_interval == 'day':
                    total_duration = (atm_data[-1][date_field] - atm_data[0][date_field]).days
                elif date_interval == 'hours':
                    total_duration = (atm_data[-1][date_field] - atm_data[0][date_field]).seconds // 3600
                coverage = None
                if date_interval == 'day':
                    coverage = round((atm_data[-1][date_field] - atm_data[0][date_field]).days / pandas.Timestamp(atm_data[-1][date_field].year,12,31).day_of_year,4)
                elif date_interval == 'hour':
                    coverage = round(((atm_data[-1][date_field] - atm_data[0][date_field]).seconds // 3600) / pandas.Timestamp(atm_data[-1][date_field].year,12,31).day_of_year * 24,4)
                results.append({
                    'atm':atm,
                    'avg':average,
                    'median':median,
                    'stdev':stdev,
                    'max':max(atm_stats),
                    'min':min(atm_stats),
                    'length':len(atm_stats),
                    'total_duration':total_duration,
                    'first':atm_data[0][date_field],
                    'last':atm_data[-1][date_field],
                    'coverage':coverage
                })
        return pandas.DataFrame.from_records(results)