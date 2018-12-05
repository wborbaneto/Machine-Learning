import numpy as np
import matplotlib as plt

class AntColony():
    def __init__(self, network, num_ants):
        tau = 0.1
        self.history = []
        # Initializing Ant Colony
        self.colony = []
        # Initializing distances between vertexes
        self.net = network
        # Initializing ferormones for each edge
        self.net_f = np.ones(network.shape) * tau
        self.net_f[network==0] = 0
        # Auxiliar array for counting indexes
        self.net_i = np.arange(0,network.shape[0])
        # Creating each ant
        for n in range(num_ants):
            self.colony.append(self.create_ant())
        self.best_ant = self.create_ant(dist = 100000)
        
        
    def create_ant(self, path_index=0,dist=0):
        # Random vertex for each ant
        vertex = np.random.randint(0, self.net.shape[0])
        # Each ant knows its path and its current vertex
        ant = {'index':len(self.colony) + 1,
               'path': [vertex],
               'path_index': path_index,
               'dist': dist}
        return ant

    def time_step(self):
        alfa =  10
        beta = 1
        # Initializing desirability
        self.d = np.zeros_like(self.net,dtype=float)
        # Calculating desirability 
        # 1/dist
        np.divide(1, self.net, out=self.d, where=self.net!=0)
        # desirability = 1/dist * ferormone
        self.d = alfa*self.d * beta*self.net_f
    
    def ant_cogito(self):
        Q = 10000
        sigma = 0.1
        # Ferormone decay
        self.net_f = (1-sigma)*self.net_f
        
        for ant in self.colony:
            path_index = ant['path_index']
            c_vertex = ant['path'][path_index]
            edges = np.where(self.net[c_vertex]!=0)
            edges = np.setdiff1d(edges, ant['path'])
            
            if len(edges):
                desire = self.d[c_vertex,[edges]]
                p = desire/np.sum(desire)
                p_vertex = c_vertex
                c_vertex = edges[self.roulette(p)]
                ant['path_index'] += 1
                ant['path'].append(c_vertex)
                self.net_f[p_vertex,c_vertex] += self.net[p_vertex,c_vertex]/Q
                self.net_f[c_vertex,p_vertex] += self.net[c_vertex,p_vertex]/Q
        
                ant['dist'] += self.net[p_vertex, c_vertex]
            else:
                self.history.append(ant['dist'])
                if ant['dist'] < self.best_ant['dist']:
                    self.best_ant['path'] = ant['path']
                    self.best_ant['dist'] = ant['dist']
                ant['path'] = [np.random.randint(0, self.net.shape[0])]
                ant['path_index'] = 0
                ant['dist'] = 0
        
        
    def roulette(self, p):
        win_index = np.random.rand(1)
        buffer = list()
        for n in range(p.size):
            buffer.append(np.sum(p[0:n+1]))
            if buffer[n] > win_index:
                 break
        return n
    
    def solution(self,vertex):
        ant = {'index': 0,
               'path': [vertex],
               'path_index': 0,
               'dist': 0}
        loop_break = 1
        while(loop_break):
            path_index = ant['path_index']
            c_vertex = ant['path'][path_index]
            edges = np.where(self.net[c_vertex]!=0)
            edges = np.setdiff1d(edges, ant['path'])
            if len(edges) <= 1:
                loop_break = 0
            c_vertex = edges[np.argmax(self.net_f[c_vertex][edges])]
            ant['path_index'] += 1
            ant['path'].append(c_vertex)
        ant['dist'] = self.traveled_distance(ant)
        return ant
             
        
network = np.array([[0, 600, 1900, 3600, 2900],
                    [600, 0, 2700, 3900, 2300],
                    [1900, 2700, 0, 4000, 2800],
                    [3600, 3900, 4000, 0, 2400],
                    [2900, 2300, 2800, 2400, 0]])
num_ants = 5
ant = AntColony(network,num_ants)

i = 0
# Number of loops
while i<=20:
    # The ants take 4 steps to conclude the walk
    for n in range(num_ants-1):
        ant.time_step()
        ant.ant_cogito()
    i += 1

net_f = ant.net_f
print(ant.best_ant)
print(ant.history)