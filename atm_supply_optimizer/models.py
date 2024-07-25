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
import numpy as np
import random
from random import randint, uniform, choice, shuffle
from heapq import heapify, heappop
from pulp import LpMinimize, LpProblem, LpStatus, LpVariable, lpSum, PULP_CBC_CMD

#### ================ SOLVERS ================ ####
class VertexSolver():
    """ An abstract class that implements a vertex cover solver """
    def __init__(self,graph:nx.Graph):
        self.graph = graph
        self.size = []
        self.coverset = []
        self.solved_graph:nx.Graph = None
        self.shortest_paths = nx.shortest_path(graph)
        self.cit_node = None
        for node in self.graph.nodes:
            if self.graph.nodes[node]['cit_node']:
                self.cit_node = node
                break
        
    def solve(self):
        """ Abstract definition of the solve action method """
        raise Exception('Not implemented')
    
    def get_cost(self):
        """ Calculates the picked solution cost """
        return sum([
            self.solved_graph.edges[edge[0],edge[1]]['distance'] * \
            self.solved_graph.edges[edge[0],edge[1]]['usage'] \
            for edge in self.solved_graph.edges
        ])
    
    def remove_extra_edges(self):
        for node in self.solved_graph.nodes:
            if node != self.cit_node:
                while self.solved_graph.degree(node) > 2:
                    edges = list(self.solved_graph.edges(node))
                    costs = [self.solved_graph[edge[0]][edge[1]]['distance'] for edge in edges]
                    max_cost = max(costs)
                    max_cost_edge = edges[costs.index(max_cost)]
                    self.solved_graph.remove_edge(max_cost_edge[0],max_cost_edge[1])
    
    def fix_broken_paths(self):
        """ Fixes broken paths between host node and active nodes """
        #self.remove_extra_edges()
        missing_edge = self.check_path()
        tried_solutions = []
        tried_neighbours = []
        while missing_edge is not None:
            if len(tried_solutions) > 1 and missing_edge[1] != tried_solutions[-1][0]:
                for sol in tried_solutions[:-1]:
                    self.solved_graph[sol[0]][sol[1]]['usage'] = 0
                tried_solutions = []
                tried_neighbours = []
            
            new_neighbours = [
                nei for nei in self.graph.neighbors(missing_edge[1]) \
                if nei != missing_edge[0] and \
                nei not in tried_neighbours and \
                self.solved_graph[missing_edge[1]][nei]['usage'] == 0 and \
                self.solved_graph.degree(nei) < 2
            ]
            if len(new_neighbours) > 0:
                new_neighbour = random.choice(new_neighbours)
                self.solved_graph[missing_edge[1]][new_neighbour]['usage'] = 1
                tried_solutions.append((missing_edge[1],new_neighbour))
                tried_neighbours.append(new_neighbour)
                missing_edge = self.check_path()
            else:
                self.solved_graph[missing_edge[1]][missing_edge[0]]['usage'] = 1
                tried_solutions.append((missing_edge[1],missing_edge[0]))
                tried_neighbours.append(missing_edge[0])
                missing_edge = self.check_path()
    
    def check_path(self):
        """ Checks if a path exists in the solved graph between two nodes """
        reducted_graph = self.solved_graph.copy()
        for edge in self.solved_graph.edges:
            if self.solved_graph.edges[edge]['usage'] <= 0:
                reducted_graph.remove_edge(u=edge[0],v=edge[1])
        cit_node = None
        for node in self.solved_graph.nodes:
            if self.solved_graph.nodes[node]['cit_node']:
                cit_node = node
                break
        
        for node in self.solved_graph.nodes:
            if cit_node is not None and node != cit_node:
                try:
                    shortest_paths = nx.shortest_path(
                        G=reducted_graph,
                        source=cit_node,
                        target=node,
                        weight="distance"
                    )
                except nx.exception.NetworkXNoPath:
                    return (cit_node,node)
        return None
    
class ApproxSolver(VertexSolver):
    """ Representation of an approximation solver """

    def solve(self):
        """ Run the solver on the provided graph """
        self.solved_graph = self.graph.copy()
        nx.set_edge_attributes(self.solved_graph,0,'usage')
        edges = self.solved_graph.edges
        self.size = []
        self.coverset = []
        s = 0
        cover_ = []
        for edge in edges:
            if edge[0] not in cover_ and edge[1] not in cover_:
                cover_.append(edge[0])
                cover_.append(edge[1])
                s += 2
                self.solved_graph[edge[0]][edge[1]]['usage'] = 1
        self.size.append(s)
        self.coverset.append(cover_)
        self.fix_broken_paths()

class GreedySolver(VertexSolver):
    """ Representation of a greedy solver """
    
    def __init__(self,graph:nx.Graph):
        self.graph = graph
        self.coverset = []
        self.edges = {}
        self.heap:Heap = None
        self.scores = {}
        for node in self.graph.nodes:
            if self.graph.nodes[node]['cit_node']:
                self.cit_node = node
                break

    def get_node_score(self,node):
        """ Gets the score of the node in order to be used in the heap """
        score = self.graph.degree[node]
        if score > 0:
            score = 1/score
        return score

    def build_heap(self):
        """ Creates the heap of nodes sorted by their node score """
        self.heap = Heap()
        self.scores = {}

        data = []  # data format: [node_degree, node_index]
        for node in self.graph.nodes:
            node_index = node
            self.scores[node_index] = self.get_node_score(node_index)#*host_preservation_weight
            # multiply to -1 for desc order
            data.append([-1 * self.scores[node_index], node_index])
        self.heap.init(data)

    def place_image(self,node_index):
        """ Places an image to the specified node """
        adj = set(self.solved_graph.edges([node_index]))
        for u, v in adj.intersection(self.edges):
            # remove edge from list
            self.edges.discard((u, v))
            self.edges.discard((v, u))
            used_edges_u = len([edge for edge in self.solved_graph.edges(u) \
                              if self.solved_graph[edge[0]][edge[1]]['usage'] == 1 or \
                              self.solved_graph[edge[1]][edge[0]]['usage'] == 1])
            used_edges_v = len([edge for edge in self.solved_graph.edges(v) \
                              if self.solved_graph[edge[0]][edge[1]]['usage'] == 1 or \
                              self.solved_graph[edge[1]][edge[0]]['usage'] == 1])

            if used_edges_u < 1 and used_edges_v < 2:
                self.solved_graph[u][v]['usage'] = 1

            # update neighbors
            if self.heap.contains(v):
                new_degree = self.scores[v] - 1
                # update index
                self.scores[v] = new_degree
                # update heap
                self.heap.update(v, -1 * new_degree)
        # add node in mvc
        self.mvc.add(node_index)
        self.coverset.append(self.mvc)

    def solve(self):
        """ Runs the algorithm on the provided graph """
        self.solved_graph = self.graph.copy()
        nx.set_edge_attributes(self.solved_graph,0,'usage')
        self.mvc = set()
        self.build_heap()
        self.edges = set(self.solved_graph.edges)
        while len(self.mvc) < len(list(self.graph.nodes)):
            # remove node with max degree
            _, node_index = self.heap.pop()
            self.place_image(node_index)
        self.fix_broken_paths()
        return

class GeneticSolver(VertexSolver):
    """ The solver class using genetic algorithm """

    def __init__(self,graph:nx.Graph):
        # Constants
        # note: keep population_size and elite_population_size of same parity
        self.population_size = 20
        self.elite_population_size = 4
        self.mutation_probability = 0.04
        self.num_iterations = 5

        # Parameters
        self.graph=graph

        # Initialise
        self.population = Population(
            self.graph, 
            self.population_size, 
            self.elite_population_size, 
            self.mutation_probability
        )
        self.population.evaluate_fitness_ranks()
        self.population.evaluate_diversity_ranks()
        self.coverset = []
        self.size = []
        for node in self.graph.nodes:
            if self.graph.nodes[node]['cit_node']:
                self.cit_node = node
                break

    def solve(self):
        """ Runs the solver """
        self.coverset = []

        # breed and mutate this population num_iterations times
        for _ in range(1, self.num_iterations + 1):
            self.population.breed()
            self.population.mutate()
            # find the new ranks
            self.population.evaluate_fitness_ranks()
            self.population.evaluate_diversity_ranks()

        # vertex cover with best fitness is our output
        best_vertex_cover = None
        best_fitness = 0
        for vertex_cover in self.population.vertexcovers:
            if vertex_cover.fitness > best_fitness:
                best_vertex_cover = vertex_cover
                best_fitness = vertex_cover.fitness
        if best_vertex_cover is not None:
            list_mvc = list(best_vertex_cover.vertexarray.keys())
            self.coverset.append(list_mvc)
            self.coverset.append(best_fitness)
            self.coverset.append(best_vertex_cover.transfered)
            self.solved_graph = best_vertex_cover.solved_graph
            self.fix_broken_paths()
        else:
            self.solved_graph = None
            self.coverset.append([])
            self.coverset.append(-1)
            self.coverset.append([])

class ILPSolver(VertexSolver):
    """ The solver class using integer linear programming algorithm """
        
    def solve(self):
        time_limit = 500
        self.model = LpProblem(name="vrp", sense=LpMinimize)
        self.solved_graph = self.graph.copy()
        
        activation = LpVariable.dicts("activation",(n for n in self.graph.nodes),cat='Integer',lowBound=0,upBound=1)
        transfered = LpVariable.dicts("transfered",(edge for edge in self.graph.edges),cat='Integer',lowBound=0,upBound=1)
        
        for n in [node for node in self.graph.nodes]: 
            self.model += lpSum([transfered[edge] for edge in self.graph.edges if edge[0] == n or edge[1] == n]) >= 1

        for n in [node for node in self.graph.nodes if node != self.cit_node]: 
            self.model += lpSum([transfered[edge] for edge in self.graph.edges if edge[0] == n or edge[1] == n]) <= 2
        
        for edge in self.graph.edges:
            self.model += transfered[edge] <= activation[edge[0]]+activation[edge[1]]

        #An = activation, self.volume = V , capacity W
        self.model += lpSum(
            [transfered[edge]*(self.graph.edges[edge[0],edge[1]]['distance']) for edge in self.graph.edges]
        )
        
        status = self.model.solve(PULP_CBC_CMD(msg=0, timeLimit=time_limit, threads=1))

        for edge in self.solved_graph.edges:
            self.solved_graph[edge[0]][edge[1]]['usage'] = min(int(round(transfered[edge].value())),1)
        self.fix_broken_paths()
        
        return {'statusCode':self.model.status,'status':LpStatus[self.model.status],'variables':self.model.variables()}

#### ================ UTILITIES ================ ####
class Heap():
    """ Representation of a Heap data structure """
    # data format: [node_degree, node_index]
    heap = []
    hash = dict()

    def init(self, initial):
        self.heap = initial
        for value, index in initial:
            self.hash[index] = value
        self.rebuild()

    def rebuild(self):
        heapify(self.heap)

    def pop(self):
        return heappop(self.heap)

    def contains(self, index):
        return index in self.hash

    def update(self, index, value):
        self.hash[index] = value
        for i, e in enumerate(self.heap):
            if e[1] == index:
                self.heap[i] = [value, index]
                break
        self.rebuild()

    def get(self, index):
        return self.hash.get(index)

    def size(self):
        return len(self.heap)

def weighted_choice(choices, weights):
    normalized_weights = np.array([weight for weight in weights]) / np.sum(weights)
    threshold = uniform(0, 1)
    total = 1
    for index, normalized_weight in enumerate(normalized_weights):
        total -= normalized_weight
        if total < threshold:
            return choices[index]
        
class Population():
    """ Representation of a population of vertex covers """
    def __init__(self, G:nx.Graph, population_size:int, elite_population_size:int,mutation_probability:float):
        self.vertexcovers:list[VertexCover] = []
        self.population_size = population_size
        self.graph = G.copy()
        self.elite_population_size = elite_population_size
        self.mutation_probability = mutation_probability

        for vertex_cover_number in range(self.population_size):
            vertex_cover = VertexCover(self)
            vertex_cover.evaluate_fitness()

            self.vertexcovers.append(vertex_cover)
            self.vertexcovers[vertex_cover_number].index = vertex_cover_number

        self.evaluated_fitness_ranks = False
        self.evaluated_diversity_ranks = False
        self.mean_fitness = 0
        self.mean_diversity = 0
        self.mean_vertex_cover_size = 0
        self.average_vertices = {}

    # evaluate fitness ranks for each vertex cover
    def evaluate_fitness_ranks(self):
        if not self.evaluated_fitness_ranks:
            for vertex_cover in self.vertexcovers:
                vertex_cover.fitness = vertex_cover.evaluate_fitness()
                self.mean_fitness += vertex_cover.fitness
                self.mean_vertex_cover_size += len(vertex_cover)

            self.mean_fitness /= self.population_size
            self.mean_vertex_cover_size /= self.population_size
            self.vertexcovers.sort(key=lambda vertex_cover: vertex_cover.fitness, reverse=True)

            for rank_number in range(self.population_size):
                self.vertexcovers[rank_number].fitness_rank = rank_number

            self.evaluated_fitness_ranks = True

    # evaluate diversity rank of each point in population
    def evaluate_diversity_ranks(self):
        if not self.evaluated_diversity_ranks:
            # find the average occurrence of every vertex in the population
            vertex_sums = {}
            for vertex_cover in self.vertexcovers:
                for vertex in vertex_cover.vertexarray:
                    if vertex not in vertex_sums:
                        vertex_sums[vertex] = 0
                    vertex_sums[vertex] += 1
            for vertex in vertex_sums:
                self.average_vertices[vertex] = vertex_sums[vertex] / self.population_size

            for vertex_cover in self.vertexcovers:
                diversity_sum = 0
                for vertex in vertex_cover.vertexarray:
                    diversity_sum += abs(1 - self.average_vertices.get(vertex,0))
                vertex_cover.diversity = diversity_sum/self.population_size
                self.mean_diversity += vertex_cover.diversity

            self.mean_diversity /= self.population_size
            self.vertexcovers.sort(key=lambda vertex_cover: vertex_cover.diversity, reverse=True)

            for rank_number in range(self.population_size):
                self.vertexcovers[rank_number].diversity_rank = rank_number

            self.evaluated_diversity_ranks = True

    # generate the new population by breeding vertex covers
    def breed(self):
        # sort according to fitness_rank
        self.vertexcovers.sort(key=lambda vertex_cover: vertex_cover.fitness_rank)

        # push all the really good ('elite') vertex covers first
        newpopulation = []
        for index in range(self.elite_population_size):
            newpopulation.append(self.vertexcovers[index])

        # assign weights to being selected for breeding
        weights = [1 / (1 + vertex_cover.fitness_rank + vertex_cover.diversity_rank) for vertex_cover in self.vertexcovers]

        # randomly select for the rest and breed
        while len(newpopulation) < self.population_size:
            parent1 = weighted_choice(list(range(self.population_size)), weights)
            parent2 = weighted_choice(list(range(self.population_size)), weights)

            # don't breed with yourself, dude!
            while parent1 == parent2:
                parent1 = weighted_choice(list(range(self.population_size)), weights)
                parent2 = weighted_choice(list(range(self.population_size)), weights)

            # breed now
            child1, child2 = self.vertexcovers[parent1].crossover(self.vertexcovers[parent2])

            # add the children
            newpopulation.append(child1)
            newpopulation.append(child2)

        # assign the new population
        self.vertexcovers = newpopulation

        self.evaluated_fitness_ranks = False
        self.evaluated_diversity_ranks = False

    # mutate population randomly
    def mutate(self):
        for vertex_cover in self.vertexcovers:
            test_probability = uniform(0, 1)
            if test_probability < self.mutation_probability:
                vertex_cover.mutate()
                vertex_cover.evaluate_fitness()

                self.evaluated_fitness_ranks = False
                self.evaluated_diversity_ranks = False

class VertexCover():
    """ Representation of a single vertex cover solution """
    def __init__(self, associated_population:Population=None):
        # population to which this point belongs
        self.associated_population = associated_population
        self.solved_graph = self.associated_population.graph.copy()
        nx.set_edge_attributes(self.solved_graph,0,'usage')

        # randomly create chromosome
        self.chromosomes = [randint(0, 1) for _ in range(len(self.associated_population.graph.nodes()))]

        # initialize
        self.vertexarray = {}
        self.chromosomenumber = 0
        self.transfered = {edge:0 for edge in self.associated_population.graph.edges}

        # required when considering the entire population
        self.index = -1
        self.fitness = 0.0
        self.diversity = 0.0
        self.fitness_rank = -1
        self.diversity_rank = -1
        self.evaluated_fitness = False

    def get_total_transfers(self,node):
        total_transfers = 0
        for tran in self.transfered:
            if tran[0] == node or tran[1] == node:
                total_transfers += self.transfered[tran]
        return total_transfers

    def evaluate_fitness(self):
        if not self.evaluated_fitness:
            working_graph = self.associated_population.graph.copy()

            self.vertexarray = {}
            self.transfered = {edge:0 for edge in self.associated_population.graph.edges}
            self.chromosomenumber = 0

            while len(self.vertexarray) < len(self.associated_population.graph.nodes()):
                # shuffle the list of vertices
                node_list = [node for node in working_graph.nodes if node not in self.vertexarray]
                shuffle(node_list)

                # check all vertices one-by-one
                vertex = choice(node_list)
                vertex_transfers = self.get_total_transfers(vertex)
                # make a choice depending on the chromosome bit
                if self.chromosomes[self.chromosomenumber] == 0:
                    # add one neighbour to vertex cover
                    neighbors = list(working_graph.neighbors(vertex))
                    available_neighbors = []
                    for nei in neighbors:
                        if self.get_total_transfers(nei) < 2:
                            available_neighbors.append(nei)
                    othervertex = choice(available_neighbors)
                    self.vertexarray[vertex] = 1
                    self.vertexarray[othervertex] = 1
                    if (vertex,othervertex) in self.transfered:
                        self.transfered[(vertex,othervertex)] = 1
                    else:
                        self.transfered[(othervertex,vertex)] = 1
                elif self.chromosomes[self.chromosomenumber] == 1:
                    # add two neighbour to vertex cover
                    neighbors = list(working_graph.neighbors(vertex))
                    available_neighbors = []
                    for nei in neighbors:
                        if self.get_total_transfers(nei) < 2:
                            available_neighbors.append(nei)
                    othervertex = choice(available_neighbors)
                    self.vertexarray[vertex] = 1
                    self.vertexarray[othervertex] = 1
                    if (vertex,othervertex) in self.transfered:
                        self.transfered[(vertex,othervertex)] = 1
                    else:
                        self.transfered[(othervertex,vertex)] = 1
                    if len(available_neighbors) > 1 and vertex_transfers < 2:
                        available_neighbors.remove(othervertex)
                        other_vertex = choice(available_neighbors)
                        self.vertexarray[othervertex] = 1
                        if (vertex,othervertex) in self.transfered:
                            self.transfered[(vertex,othervertex)] = 1
                        else:
                            self.transfered[(othervertex,vertex)] = 1
                # go to the next chromosome to be checked
                self.chromosomenumber = self.chromosomenumber + 1
                continue

            request_failed_nodes = []
            
            for vertex in self.associated_population.graph.nodes:
                if not vertex in self.vertexarray:
                    gotImage = False
                    for edge in self.transfered:
                        if (edge[0] == vertex or edge[1] == vertex) and self.transfered[edge] > 0:
                            gotImage = True
                    for other_vertex in self.vertexarray:
                        if not gotImage and self.get_total_transfers(other_vertex) < 2:
                            if (vertex,other_vertex) in self.transfered:
                                self.transfered[(vertex,other_vertex)] = 1
                                gotImage = True
                            elif (other_vertex,vertex) in self.transfered:
                                gotImage = True
                                self.transfered[(other_vertex,vertex)] = 1
                    if not gotImage:
                        request_failed_nodes.append(vertex)
            if len(request_failed_nodes) > 0:
                self.fitness = 0.0
            else:
                score = sum([self.transfered[edge]*self.associated_population.graph.edges[edge[0],edge[1]]['distance'] for edge in self.associated_population.graph.edges])
                if score == 0:
                    self.fitness = 1.0
                else:
                    self.fitness = 1 / score
            self.evaluated_fitness = True
            for edge in self.transfered:
                self.solved_graph[edge[0]][edge[1]]['usage'] = self.transfered[edge]
        return self.fitness

    def mutate(self):
        """ Mutates the chromosome at a random index """
        if self.chromosomenumber > 0:
            index = randint(0, self.chromosomenumber)
        else:
            index = 0

        if self.chromosomes[index] == 0:
            self.chromosomes[index] = 1
        elif self.chromosomes[index] == 1:
            self.chromosomes[index] = 0

        self.evaluated_fitness = False
        self.evaluate_fitness()

    def __len__(self):
        return len(self.vertexarray)

    def __iter__(self):
        return iter(self.vertexarray)
    
    def crossover(self, parent2) -> tuple:
        """ Crossover between two vertex cover chromosomes """
        if self.associated_population != parent2.associated_population:
            raise ValueError("Vertex covers belong to different populations.")
        child1 = VertexCover(self.associated_population)
        child2 = VertexCover(parent2.associated_population)
        # find the point to split and rejoin the chromosomes
        # note that chromosome number + 1 is the actual length of the chromosome in each vertex cover encoding
        split_point = randint(0, min(self.chromosomenumber, parent2.chromosomenumber))
        # actual splitting and recombining
        child1.chromosomes = self.chromosomes[:split_point] + parent2.chromosomes[split_point:]
        child2.chromosomes = parent2.chromosomes[:split_point] + self.chromosomes[split_point:]
        # evaluate fitnesses
        child1.evaluate_fitness()
        child2.evaluate_fitness()
        return (child1, child2)
    
