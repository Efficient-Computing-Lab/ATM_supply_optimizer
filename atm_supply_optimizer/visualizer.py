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
import matplotlib.pyplot as plt
from gmplot import gmplot
from geopy.distance import geodesic
import utm

class Grapher():
    """ Class that creates a graph from a dataset of records and visualizes it """

    def __init__(
            self,
            dataset:dict|list[dict]|nx.Graph,
            lat_lon_cols:tuple[str,str]=["lat","lon"],
            key_col:str="ID"
        ):
        """ Creates and initializes the Grapher depending on the dataset class.
        If it is a dict it creates grouped distances.
        If it is a list it creates full distances.
        If it is a graph it tries to reverse engineer the dataset."""
        if isinstance(dataset, list):
            self.dataset:list[dict] = dataset
            self.calculate_distances_full(lat_lon_cols,key_col)
            self.create_graph(key_col)
        elif isinstance(dataset, dict):
            self.dataset:list[dict] = []
            for key in dataset:
                datum = dataset[key].copy()
                datum[key_col] = key
                self.dataset.append(datum)
            self.calculate_distances_groups(
                key_col=key_col,
                lat_lon_cols=lat_lon_cols,
                grouped_data=dataset  
            )
            self.create_graph(key_col)
        elif isinstance(dataset, nx.Graph):
            self.dataset:list[dict] = []
            lats:dict = nx.get_node_attributes(dataset, lat_lon_cols[0])
            lons:dict = nx.get_node_attributes(dataset, lat_lon_cols[1])
            for node in dataset.nodes:
                self.dataset.append({
                    key_col:str(node),
                    lat_lon_cols[0]:lats.get(node,None),
                    lat_lon_cols[1]:lons.get(node,None),
                })
            self.distances:dict = {}
            distances = nx.get_edge_attributes(dataset,'distance')
            weights = nx.get_edge_attributes(dataset,'weight')
            for edge in dataset.edges:
                distance = distances.get(edge,None)
                if distance is None:
                    distance = weights.get(edge,None)
                    if distance is not None:
                        distance = round(1/distance,2)
                if distance is not None:
                    self.distances[edge[0],edge[1]] = distance
            self.graph = dataset
        else:
            raise Exception(f"Cannot process dataset, unknown type {type(dataset)}")

    def calculate_distances_full(self,lat_lon_cols:tuple[str,str],key_col:str) -> dict:
        """ Calculates the distances in pairs of datapoints and returns it in a dict having 
        (key_col1,key_col2) as key"""
        self.distances = {}
        for i in range(len(self.dataset)):
            for j in range(i+1, len(self.dataset)):
                datum1 = (self.dataset[i][lat_lon_cols[0]], self.dataset[i][lat_lon_cols[1]])
                datum2 = (self.dataset[j][lat_lon_cols[0]], self.dataset[j][lat_lon_cols[1]])
                self.distances[(self.dataset[i][key_col], self.dataset[j][key_col])] = \
                    round(geodesic(datum1, datum2).kilometers,2)
        return self.distances
    
    def calculate_distances_groups(self,lat_lon_cols:tuple[str,str],key_col:str,grouped_data:dict) -> dict:
        """ Calculates the distances in pairs of datapoints keeping each group separate and 
        returns it in a dict having (key_col1,key_col2) as key"""
        self.distances = {}
        for group in grouped_data:
            data = grouped_data[group]
            for i in range(len(data)):
                for j in range(i+1, len(data)):
                    datum1 = (data[i][lat_lon_cols[0]], data[i][lat_lon_cols[1]])
                    datum2 = (data[j][lat_lon_cols[0]], data[j][lat_lon_cols[1]])
                    self.distances[(data[i][key_col], data[j][key_col])] = \
                        round(geodesic(datum1, datum2).kilometers,2)
            found = False
            for datum in self.dataset:
                if datum[key_col] == data[i][key_col]:
                    found = True
                    break
            if not found:
                self.dataset.append({
                    key_col:data[i][key_col],
                    lat_lon_cols[0]:data[i][lat_lon_cols[0]],
                    lat_lon_cols[1]:data[i][lat_lon_cols[1]],
                })
        return self.distances
    
    def create_graph(self,key_col:str) -> nx.Graph:
        """ Creates the graph from dataset """
        self.graph = nx.Graph()
        for datum in self.dataset:
            self.graph.add_node(
                datum[key_col],
                lat=datum.get('lat',None),
                lon=datum.get('lon',None),
                cit_node=datum.get('cit_node',False)
            )
        for edge, weight in self.distances.items():
            adjusted_w = weight
            if weight > 0:
                adjusted_w = 1/weight
            self.graph.add_edge(edge[0], edge[1], weight=adjusted_w, distance=weight)
        return self.graph
    
    def visualize(self, title:str, filepath:str, graph:nx.Graph=None, 
                  active_edges:list[tuple]=None, geoposition:bool=True):
        """ Visualizes the graph and stores it as an image file """
        if graph is None:
            graph = self.graph
        pos = None

        if geoposition:
            positions = {}
            lats = nx.get_node_attributes(graph, "lat")
            lons = nx.get_node_attributes(graph, "lon")
            min_lat = 99999999999
            min_lon = 99999999999
            max_lat = 0
            max_lon = 0
            for node in graph.nodes:
                utm_pos = utm.from_latlon(lats[node], lons[node])
                positions[node] = (utm_pos[0],utm_pos[1])
                min_lat = min([min_lat,utm_pos[0]])
                min_lon = min([min_lon,utm_pos[1]])
                max_lat = max([max_lat,utm_pos[0]])
                max_lon = max([max_lon,utm_pos[1]])
            for node in positions:
                positions[node] = (
                    100 * ((positions[node][0]- min_lat) / (max_lat - min_lat)),
                    100 * ((positions[node][1]- min_lon) / (max_lon - min_lon))
                )
            pos = nx.spring_layout(graph, pos=positions, fixed=list(positions.keys()))  # Position nodes using the spring layout algorithm
        else:
            pos = nx.spring_layout(graph,weight="weight")  # Position nodes using the spring layout algorithm

        plt.figure(figsize=(20,20))

        # Draw the graph
        distances = nx.get_edge_attributes(graph,'distance')
        edge_labels = {edge:distances[edge] for edge in graph.edges}
        nx.draw(graph, pos, with_labels=True, node_size=2000, node_color='skyblue',font_size=8)

        # Highlight nodes and edges up to the current step
        if active_edges is not None:
            nx.draw_networkx_edges(graph, pos, edgelist=active_edges, edge_color='red', width=2)

        # Draw edge labels
        for edge, label in edge_labels.items():
            try:
                nx.draw_networkx_edge_labels(graph, pos, edge_labels={edge: label}, font_size=6)
            except ValueError as e:
                pass

        # Save the image
        plt.title(title)
        plt.axis('off')  # Turn off axis
        plt.savefig(filepath)
        plt.close()
    
    def visualize_traversal(self, title:str, filepath:str, geoposition:bool=True):
        """ Visualizes the graph using only the activated edges and stores it an 
        image file"""
        reducted_graph = self.graph.copy()
        for edge in self.graph.edges:
            if self.graph.edges[edge]['usage'] <= 0:
                reducted_graph.remove_edge(u=edge[0],v=edge[1])
        self.visualize(title=title,filepath=filepath,graph=reducted_graph,geoposition=geoposition)