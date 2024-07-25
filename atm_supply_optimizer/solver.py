#Part of ATMO-MoRe ATM Supply Optimizer, a system that optimizes ATM resupply planning.
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

import networkx as nx
import datetime as dt
from atm_load_prediction.data_handler import DataHandler, Preprocessor
from atm_load_prediction.utils import timer, printProgressBar
from atm_supply_optimizer.models import VertexSolver, ApproxSolver, GeneticSolver, GreedySolver, ILPSolver
from atm_supply_optimizer.visualizer import Grapher
import sys, random, pandas, statistics
from atm_load_prediction.evaluator import train_model, train_features_full, class_feature_lstm, \
    apply_model

class ATMOSolver():
    """ Class representation of an ATMO-MoRe supply network optimization problem """

    def __init__(self,solver_clf:VertexSolver,daily_graphs:dict=None,db_handler:DataHandler=None,
                 dataset:list=None,**solver_args):
        """ Initialization of the ATMO-MoRe solver
        
        Params:
        solver:VertexSolver A solver from the models.graph_models to be used for the solving process
        daily_graphs:dict A dict of day to day graphs with the day as key to be solved if none the 
        ATMOSolver will try to load the daily graphs from a dataset.
        db_handler:DataHandler An ATMO-MoRe database handler to be used for MongoDB communications
        dataset:list A dataset to be used for the creation of the daily graphs
        """

        # Set parameters provided
        self.solver_clf = solver_clf
        self.solver_params = solver_args
        self.daily_graphs = daily_graphs
        if self.daily_graphs is None:
            if dataset is None:
                self.db_handler = db_handler
                if self.db_handler is None:
                    self.db_handler = DataHandler()
                self.create_graphs_from_test_dataset()
            else:
                self.create_graphs_from_dataset(dataset)
        self.solved_graphs = []

        # Set configuration options
        sys.setrecursionlimit(100000)

    def create_graphs_from_test_dataset(self):
        """ Loads the test dataset from the database and creates the daily graphs """
        supply_info_test = self.db_handler.import_test_supply_info()
        atmomore_preprocessor_test = Preprocessor(supply_info_test)
        atmomore_preprocessor_test.clean_supply_types()
        supply_info_test = atmomore_preprocessor_test.clean_data
        timeseries_test = atmomore_preprocessor_test.create_load_timeseries(samples_threshold=1,
                                                                            coverage_threshold=0.0)
        test_timeseries = Preprocessor.group_by('ATM',timeseries_test)
        test_dataset_dict = Preprocessor.timeseries_to_supervised(test_timeseries,1)
        for atm in test_dataset_dict:
            test_dataset_dict[atm] = test_dataset_dict[atm].reset_index(names="timestamp_t-1")\
                                    .to_dict(orient='records')
        test_dataset_list = Preprocessor.degroup('ATM',test_dataset_dict)
        test_dataset_list_clean = [{
            'ATM':rec['ATM'],
            'value':rec['value_t-1'],
            'date':(rec['timestamp_t-1'] - dt.timedelta(days=1)).date()
        } for rec in test_dataset_list]
        supply_lists = get_daily_supply_lists_raw(test_dataset_list_clean)
        self.daily_graphs = {}
        for day in supply_lists:
            self.daily_graphs[day] = atms_to_graph(supply_lists[day],day)
    
    def create_graphs_from_dataset(self,dataset:list):
        """ Creates the daily graphs from the provided dataset """
        supply_lists = get_daily_supply_lists_raw(dataset)
        self.daily_graphs = {}
        for day in supply_lists:
            self.daily_graphs[day] = atms_to_graph(supply_lists[day],day)

    def get_cost(self,raw:bool=False):
        """ Returns the aggregated cost metrics by applying the cost function of each solver """
        day_metrics = {}
        group_metrics = {}
        total_metrics = {'raw':[],'avg':0.0,'median':0.0,'max':0.0,'min':0.0,'std':0.0,'len':0,'time':0,'vehicles':[]}
        for day in self.costs:
            day_costs = self.costs[day]
            day_vehicles = self.vehicles[day]
            day_times = self.times[day]
            if day not in day_metrics:
                day_metrics[day] = {'raw':[],'avg':0.0,'median':0.0,'max':0.0,'min':0.0,'std':0.0,'len':0,'time':day_times['total'],'vehicles':[]}
            if isinstance(day_costs,dict):
                for group in day_costs:
                    if group not in group_metrics:
                        group_metrics[group] = {'raw':[],'avg':0.0,'median':0.0,'max':0.0,'min':0.0,'std':0.0,'len':0,'time':[],'vehicles':[]}
                    group_cost = day_costs[group]
                    group_vehicles = day_vehicles[group]
                    group_metrics[group]['time'].append(day_times[group])
                    group_metrics[group]['raw'].append(group_cost)
                    day_metrics[day]['raw'].append(group_cost)
                    total_metrics['raw'].append(group_cost)
                    group_metrics[group]['vehicles'].append(group_vehicles)
                    day_metrics[day]['vehicles'].append(group_vehicles)
                    total_metrics['vehicles'].append(group_vehicles)
            else:
                day_metrics[day]['raw'].append(day_costs)
                total_metrics['raw'].append(day_costs)
                day_metrics[day]['vehicles'].append(group_vehicles)
                total_metrics['vehicles'].append(group_vehicles)
        total_metrics['time']= self.times['total']

        for day in day_metrics:
            day_metrics[day]['len'] = len(day_metrics[day]['raw'])
            day_metrics[day]['avg'] = sum(day_metrics[day]['raw']) / day_metrics[day]['len']
            day_metrics[day]['vehicles'] = sum(day_metrics[day]['vehicles']) / len(day_metrics[day]['vehicles'])
            day_metrics[day]['max'] = max(day_metrics[day]['raw'])
            day_metrics[day]['min'] = min(day_metrics[day]['raw'])
            day_metrics[day]['median'] = statistics.median(day_metrics[day]['raw'])
            try:
                day_metrics[day]['std'] = statistics.stdev(day_metrics[day]['raw'])
            except statistics.StatisticsError:
                day_metrics[day]['std'] = None
            if not raw:
                day_metrics[day].pop('raw')
        for group in group_metrics:
            group_metrics[group]['time'] = sum(group_metrics[group]['time'])
            group_metrics[group]['len'] = len(group_metrics[group]['raw'])
            group_metrics[group]['avg'] = sum(group_metrics[group]['raw']) / group_metrics[group]['len']
            group_metrics[group]['vehicles'] = sum(group_metrics[group]['vehicles']) / len(group_metrics[group]['vehicles'])
            group_metrics[group]['max'] = max(group_metrics[group]['raw'])
            group_metrics[group]['min'] = min(group_metrics[group]['raw'])
            group_metrics[group]['median'] = statistics.median(group_metrics[group]['raw'])
            try:
                group_metrics[group]['std'] = statistics.stdev(group_metrics[group]['raw'])
            except statistics.StatisticsError:
                day_metrics[day]['std'] = None
            if not raw:
                group_metrics[group].pop('raw')
        total_metrics['len'] = len(total_metrics['raw'])
        total_metrics['vehicles'] = sum(total_metrics['vehicles']) / len(total_metrics['vehicles'])
        total_metrics['avg'] = sum(total_metrics['raw']) / total_metrics['len']
        total_metrics['max'] = max(total_metrics['raw'])
        total_metrics['min'] = min(total_metrics['raw'])
        total_metrics['median'] = statistics.median(total_metrics['raw'])
        try:
            total_metrics['std'] = statistics.stdev(total_metrics['raw'])
        except statistics.StatisticsError:
            day_metrics[day]['std'] = None
        if not raw:
            total_metrics.pop('raw')


        return {"day":day_metrics,"group":group_metrics,"total":total_metrics}

    def solve(self):
        """ A method that tries to solve the daily graphs using the pre-defined solver class """
        self.solved_graphs = {}
        self.costs = {}
        self.times = {}
        self.vehicles = {}
        start_time,_ = timer()
        for day in self.daily_graphs:
            start_time_day,_ = timer(label=f"Solving {day}...")
            day_graph = self.daily_graphs[day]
            if isinstance(day_graph,nx.Graph):
                nx.set_edge_attributes(day_graph, values=0, name='usage')
                solver:VertexSolver = self.solver_clf(graph = day_graph,**self.solver_params)
                solver.solve()
                solved_day_graph = solver.solved_graph
                self.solved_graphs[day] = solved_day_graph
                self.costs[day] = solver.get_cost()
                self.vehicles[day] = solver.solved_graph.degree(solver.cit_node)
                _,duration = timer(start=start_time_day)
                self.times[day]={'total':duration}
            elif isinstance(day_graph,dict):
                solved_day_graphs = {}
                self.costs[day] = {}
                self.times[day] = {}
                self.vehicles[day] = {}
                for group in self.daily_graphs[day]:
                    start_time_group,_ = timer()
                    group_graph = day_graph[group]
                    nx.set_edge_attributes(group_graph, values=0, name='usage')
                    solver:VertexSolver = self.solver_clf(graph = group_graph,**self.solver_params)
                    solver.solve()
                    solved_day_graphs[group] = solver.solved_graph
                    self.costs[day][group] = solver.get_cost()
                    self.vehicles[day][group] = solver.solved_graph.degree(solver.cit_node)
                    _,duration = timer(start=start_time_group)
                    self.times[day][group] = duration
                _,duration = timer(start=start_time_day)
                self.times[day]["total"] = duration
                self.solved_graphs[day] = solved_day_graphs
            else:
                raise Exception(
                    f"Cannot solve, unknown type of graph for day {day}: {type(day_graph)}"
                )
        _,duration = timer(start=start_time)
        self.times["total"]= duration

    def get_solution_text(self):
        """ Get a text representation of the paths in the solved graphs """
        day_paths = {}
        for day in self.solved_graphs:
            day_paths[day] = []
            day_graph:nx.Graph = self.solved_graphs[day]
            cit_node = None
            for node in day_graph.nodes:
                if day_graph.nodes[node]['cit_node']:
                    cit_node = node
                    break
            if cit_node:
                leaves = [node for node in day_graph.nodes if day_graph.degree[node] == 1]
                for leaf in leaves:
                    day_paths[day].append(nx.shortest_path(day_graph,cit_node,leaf))
        return day_paths

    def visualize(self,prefix:str=None):
        """ Visualize the solved graphs """
        for day in self.solved_graphs:
            print(f"Visualizing {day}...")
            day_graph = self.solved_graphs[day]
            if isinstance(day_graph,dict):
                for group in day_graph:      
                    grapher = Grapher(dataset=day_graph[group],lat_lon_cols=("lat","lon"),
                                      key_col='ATM')
                    grapher.visualize_traversal(
                        title=f"Supply for {day.isoformat()}",
                        filepath=f"results/graphs/{prefix+'_' if prefix else ''}"+\
                                 f"{group.lower()}_supply_graph_{day.isoformat()}.png",
                        geoposition=False
                    )
            elif isinstance(day_graph,nx.Graph):
                grapher = Grapher(dataset=day_graph,lat_lon_cols=("lat","lon"),
                                    key_col='ATM')
                grapher.visualize_traversal(
                    title=f"Supply for {day.isoformat()}",
                    filepath=f"results/graphs/{prefix+'_' if prefix else ''}"+\
                             f"supply_graph_{day.isoformat()}.png",
                    geoposition=False
                )
            else:
                raise Exception(
                    f"Cannot visualize, unknown type of graph for day {day}: {type(day_graph)}"
                )
            
def anonymize(text:str):
    letters =["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t",
              "u","v","w","x","y","z","A","B","C","D","E","F","G","H","I","J","K","L","M","N",
              "O","P","Q","R","S","T","U","V","W","X","Y","Z"]
    random.shuffle(letters)
    hashed = str(hash(text))
    hashed_parts = []
    for char_index in range(len(hashed)-1):
        hashed_parts.append(hashed[char_index:char_index+2])
    res = ""
    for hashed_part in hashed_parts:
        mod = int(hashed_part)%len(letters)
        res += letters[mod]
    return res

def atms_to_graph(atm_list:list,date:dt.date,paint:bool=False) -> dict:
    """ Gets the location and cit info of an ATM list and creates a graph based on their distance 
    from their cit """
    dh = DataHandler()
    atm_info = dh.get_atm_info(atm_list)
    grouped_data = Preprocessor.group_by('cit_local',atm_info)
    cit_info = dh.get_cit_info()
    grouped_atms = {}
    for group in grouped_data:
        cit_details = find(cit_info,'_id',group)
        if cit_details is not None:
            grouped_data[group].append(
                {'ATM':group,'lat':cit_details['lat'],'lon':cit_details['lon'],
                 'cit_local':group,'cit_node':True}
            )
        for datum in grouped_data[group]:
            datum['ATM'] = anonymize(datum['ATM'])
        gv = Grapher(grouped_data[group],lat_lon_cols=('lat','lon'),key_col='ATM')
        if paint:
            gv.visualize(
                title=f"Supply for {date.isoformat()}",
                filepath=f"results/graphs/{group.lower()}_supply_graph_{date.isoformat()}.png",
                geoposition=False
            )
        grouped_atms[group] = gv.graph
    return grouped_atms

def get_supply_graph(models_dict:dict,avg_changes_dict:dict,test_dataset_dict:dict):
    """ Apply the provided model and create a supply graph for the next day """

    common_atms = [atm for atm in test_dataset_dict if atm in models_dict]
    supply_targets = {}
    for atm in common_atms:
        model = models_dict[atm]
        test_dataset = test_dataset_dict[atm]
        res_days = apply_model(model,test_dataset,avg_change=avg_changes_dict[atm])[0]
        if res_days < 2:
            supply_targets[atm] = res_days
    return list(supply_targets.keys())

def get_daily_supply_lists(train_dataset_dict:dict,simple_test_dataset:list[dict]) -> dict:
    """ Get a dict of daily supply lists based on the simple_test_dataset [{'ATM','Value','Date'}] 
    provided """

    clean_test_data = []
    for row in simple_test_dataset:
        atm = row['ATM']
        value = float(row['value'])
        date = row['date']
        if type(date) == str:
            date = dt.datetime.strptime(date, "%d/%m/%Y")
        clean_test_data.append({
            'ATM':atm,
            'value':value,
            'date':date
        })
    clean_test_data.sort(key=lambda x:x['date'])
    test_dataset_days_dict = Preprocessor.group_by('date',clean_test_data)
    train_features_actual = [f"{feature}_t-1" for feature in train_features_full]
    models_dict = {}
    avg_changes_dict = {}
    for atm in train_dataset_dict:
        train_dataset = train_dataset_dict[atm]
        models_dict[atm] = train_model(train_dataset,5,train_features=train_features_actual,target_feature=class_feature_lstm)
        avg_changes_dict[atm] = train_dataset_dict[atm]['change_t-1'].mean()

    daily_lists = {}
    for day in test_dataset_days_dict:
        test_data = pandas.DataFrame.from_records([{
            'ATM':rec['ATM'],
            'value':rec['value'],
            'date':day
        } for rec in test_dataset_days_dict[day]])
        test_dataset = Preprocessor.test_to_supervised(test_data,train_features_full)
        day_list = get_supply_graph(models_dict,avg_changes_dict,test_dataset)
        daily_lists[(day + dt.timedelta(days=1)).date()] = day_list
    return daily_lists

def get_daily_supply_lists_raw(simple_test_dataset:list[dict]) -> dict:
    """ Get a dict of daily supply lists based on the simple_test_dataset [{'ATM','Value','Date'}] 
    provided """

    clean_test_data = []
    for row in simple_test_dataset:
        atm = row['ATM']
        value = float(row['value'])
        date = row['date']
        if type(date) == str:
            date = dt.datetime.strptime(date, "%d/%m/%Y").date()
        clean_test_data.append({
            'ATM':atm,
            'value':value,
            'date':date
        })
    clean_test_data.sort(key=lambda x:x['date'])
    test_dataset_days_dict = Preprocessor.group_by('date',clean_test_data)

    daily_lists = {}
    for day in test_dataset_days_dict:
        day_list = list({rec['ATM']:1 for rec in test_dataset_days_dict[day]}.keys())
        daily_lists[day + dt.timedelta(days=1)] = day_list
    return daily_lists

def get_daily_graphs_from_test_csv(test_file:str) -> dict:
    """ Gets the daily supply graphs from a csv file """
    test_dataset = pandas.read_csv(test_file,sep=';',decimal='.')
    train_dataset = Preprocessor.initiate_training_datasets()
    train_dataset_dict = Preprocessor.timeseries_to_supervised(train_dataset,1)
    supply_lists = get_daily_supply_lists(train_dataset_dict,test_dataset.to_dict(orient='records'))

    print("Converting daily supply lists to graphs...")
    graphs_dict = {}
    day_index = 1
    for day in supply_lists:
        graphs_dict[day] = atms_to_graph(supply_lists[day],day)
        printProgressBar(day_index,len(supply_lists),
                         suffix=f" Day {day.isoformat()}: {len(supply_lists[day])}")
        day_index += 1
    return graphs_dict

def get_daily_graphs_from_test_data():
    """ Gets the daily supply graphs from the test data by applying the supply models """
    evaluation_datasets = Preprocessor.initiate_evaluation_datasets()
    train_dataset_dict = Preprocessor.timeseries_to_supervised(evaluation_datasets['train_timeseries'],1)
    test_dataset_dict = Preprocessor.timeseries_to_supervised(evaluation_datasets['test_timeseries'],1)
    for atm in test_dataset_dict:
        test_dataset_dict[atm] = test_dataset_dict[atm].reset_index(names="timestamp_t-1").to_dict(orient='records')
    test_dataset_list = Preprocessor.degroup('ATM',test_dataset_dict)
    test_dataset_list_clean = [{
        'ATM':rec['ATM'],
        'value':rec['value_t-1'],
        'date':rec['timestamp_t-1'] - dt.timedelta(days=1)
    } for rec in test_dataset_list]
    supply_lists = get_daily_supply_lists(train_dataset_dict,test_dataset_list_clean)
    
    print("Converting daily supply lists to graphs...")
    graphs_dict = {}
    day_index = 1
    for day in supply_lists:
        graphs_dict[day] = atms_to_graph(supply_lists[day],day)
        printProgressBar(day_index,len(supply_lists),
                         suffix=f" Day {day.isoformat()}: {len(supply_lists[day])}")
        day_index += 1
    return graphs_dict

def get_daily_graphs_from_test_data_raw() -> dict:
    """ Gets the daily supply graphs based on the test data without applying the models """
    evaluation_datasets = Preprocessor.initiate_evaluation_datasets()
    test_dataset_dict = Preprocessor.timeseries_to_supervised(evaluation_datasets['test_timeseries'],1)
    for atm in test_dataset_dict:
        test_dataset_dict[atm] = test_dataset_dict[atm].reset_index(names="timestamp_t-1").to_dict(orient='records')
    test_dataset_list = Preprocessor.degroup('ATM',test_dataset_dict)
    test_dataset_list_clean = [{
        'ATM':rec['ATM'],
        'value':rec['value_t-1'],
        'date':(rec['timestamp_t-1'] - dt.timedelta(days=1)).date()
    } for rec in test_dataset_list]
    supply_lists = get_daily_supply_lists_raw(test_dataset_list_clean)

    print("Converting daily supply lists to graphs...")
    graphs_dict = {}
    day_index = 1
    for day in list(supply_lists.keys()):
        graphs_dict[day] = atms_to_graph(supply_lists[day],day,True)
        printProgressBar(day_index,len(supply_lists),
                         suffix=f" Day {day.isoformat()}: {len(supply_lists[day])}")
        day_index += 1
    return graphs_dict

def collect_supply_optimization_results():
    solver_clfs = [
        "approx",
        "greedy",
        "ilp",
        "genetic"
    ]
    totals = []
    groups = []
    days = []
    for solver_selection in solver_clfs:
        total_df = pandas.read_csv(
            f"results/route_optimization/costs_{solver_selection}_total.csv",
            sep=";",decimal=",",encoding="utf8")
        total_df['algorithm'] = solver_selection
        totals.append(total_df)
    
        day_df = pandas.read_csv(
            f"results/route_optimization/costs_{solver_selection}_day.csv",
            sep=";",decimal=",",encoding="utf8")
        day_df['algorithm'] = solver_selection
        days.append(day_df)
    
        group_df = pandas.read_csv(
            f"results/route_optimization/costs_{solver_selection}_group.csv",
            sep=";",decimal=",",encoding="utf8")
        group_df['algorithm'] = solver_selection
        groups.append(group_df)
    pandas.concat(totals).to_csv(f"results/route_optimization/costs_collected_total.csv",sep=";",decimal=",",encoding="utf8")
    pandas.concat(days).to_csv(f"results/route_optimization/costs_collected_day.csv",sep=";",decimal=",",encoding="utf8")
    pandas.concat(groups).to_csv(f"results/route_optimization/costs_collected_group.csv",sep=";",decimal=",",encoding="utf8")

def daily_supply_optimization_evaluation(daily_supply_graphs:dict,solver_selection:str) -> list | dict:
    """ Applies the graph models to optimize the supply graphs per day """
    solved_graphs = {}
    solver_clfs = {
        "approx":ApproxSolver,
        "greedy":GreedySolver,
        "ilp":ILPSolver,
        "genetic":GeneticSolver
    }
    if solver_selection:
        solved_graphs[solver_selection] = daily_supply_optimization(daily_supply_graphs,solver_selection).solved_graphs
    else:
        for solver_selection in solver_clfs:
            print(f"\nInitiating solving with method: {solver_selection}...")
            solver = ATMOSolver(solver_clf=solver_clfs[solver_selection],daily_graphs=daily_supply_graphs)
            solver.solve()
            solver.visualize(prefix="solved")
            costs = solver.get_cost()
            day_cost_records = []
            group_cost_records = []
            total_cost_records = []
            print('\n===============================\n')
            print('Day costs:')
            for day in costs['day']:
                print(f"{day} : {costs['day'][day]}")
                rec = costs['day'][day]
                rec['day'] = day
                day_cost_records.append(rec)
            print('\nGroup costs:')
            for group in costs['group']:
                print(f"{group} : {costs['group'][group]}")
                rec = costs['group'][group]
                rec['group'] = group
                group_cost_records.append(rec)
            print('\nTotal costs:')
            print(f"Total : {costs['total']}")
            total_cost_records.append(costs['total'])
            print('\n===============================\n')
            pandas.DataFrame.from_records(day_cost_records).reset_index().to_csv(
                f"results/route_optimization/costs_{solver_selection}_day.csv",
                sep=";",decimal=",",encoding="utf8",index=False)
            pandas.DataFrame.from_records(group_cost_records).reset_index().to_csv(
                f"results/route_optimization/costs_{solver_selection}_group.csv",
                sep=";",decimal=",",encoding="utf8",index=False)
            pandas.DataFrame.from_records(total_cost_records).reset_index().to_csv(
                f"results/route_optimization/costs_{solver_selection}_total.csv",
                sep=";",decimal=",",encoding="utf8",index=False)
            solved_graphs[solver_selection] = solver.solved_graphs
    return solved_graphs

def daily_supply_optimization(daily_supply_graphs:dict,solver_selection:str) -> list | dict:
    """ Applies the graph models to optimize the supply graphs per day """
    solver_clfs = {
        "approximation":ApproxSolver,
        "greedy":GreedySolver,
        "ilp":ILPSolver,
        "genetic":GeneticSolver
    }
    print(f"\nInitiating solving with method: {solver_selection}...")
    solver = ATMOSolver(solver_clf=solver_clfs[solver_selection],daily_graphs=daily_supply_graphs)
    solver.solve()
    solver.visualize(prefix="solved")
    costs = solver.get_cost()
    day_cost_records = []
    group_cost_records = []
    total_cost_records = []
    print('\n===============================\n')
    print('Day costs:')
    for day in costs['day']:
        print(f"{day} : {costs['day'][day]}")
        rec = costs['day'][day]
        rec['day'] = day
        day_cost_records.append(rec)
    print('\nGroup costs:')
    for group in costs['group']:
        print(f"{group} : {costs['group'][group]}")
        rec = costs['group'][group]
        rec['group'] = group
        group_cost_records.append(rec)
    print('\nTotal costs:')
    print(f"Total : {costs['total']}")
    total_cost_records.append(costs['total'])
    print('\n===============================\n')
    pandas.DataFrame.from_records(day_cost_records).reset_index().to_csv(
        f"results/route_optimization/costs_{solver_selection}_day.csv",
        sep=";",decimal=",",encoding="utf8",index=False)
    pandas.DataFrame.from_records(group_cost_records).reset_index().to_csv(
        f"results/route_optimization/costs_{solver_selection}_group.csv",
        sep=";",decimal=",",encoding="utf8",index=False)
    pandas.DataFrame.from_records(total_cost_records).reset_index().to_csv(
        f"results/route_optimization/costs_{solver_selection}_total.csv",
        sep=";",decimal=",",encoding="utf8",index=False)
    return solver