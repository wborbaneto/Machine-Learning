import numpy as np
import matplotlib as plt

class AntColony():
    def __init__(self, network, num_ants):
        self.colony = []
        self.net = network
        self.net_f = np.ones(network.shape) * 0.1
        self.net_f[network==0] = 0
        for n in range(num_ants):
            self.colony.append(self.create_ant())

    def create_ant(self):
        vertex = np.random.randint(0, self.net.shape[0])
        ant = {'index':len(self.colony) + 1,
               'current_vertex': vertex,
               'previous_vertex': 0,
               'path': [vertex]}
        return ant

    def time_step(self):
        self.d = np.zeros_like(self.net,dtype=float)
        np.divide(1, self.net, out=self.d, where=self.net!=0)
        self.d = self.d * self.net_f
    
    def ant_cogito(self):
        self.net_f = (1-0.01)*self.net_f
        for ant in self.colony:
            c_vertex = ant['current_vertex']
            edges = self.d[c_vertex]
            p = edges/np.sum(edges)
            p_vertex = c_vertex
            c_vertex = self.roulette(p)
            
            ant['previous_vertex'] = p_vertex
            ant['current_vertex'] = c_vertex
            ant['path'].append(c_vertex)
            self.net_f[p_vertex,c_vertex] += self.net[p_vertex,c_vertex]/1000
            self.net_f[c_vertex,p_vertex] += self.net[c_vertex,p_vertex]/1000
        
            

    def roulette(self, p):
        win_index = np.random.rand(1)
        buffer = list()
        for n in range(p.size):
            buffer.append(np.sum(p[0:n+1]))
            if buffer[n] > win_index:
                 break
        return n

            
network = np.array([[0, 600, 1900, 3600, 2900],
                    [60, 0, 2700, 3900, 2300],
                    [1900, 2700, 0, 4000, 2800],
                    [3600, 3900, 4000, 0, 2400],
                    [2900, 2300, 2800, 2400, 0]])
num_ants = 5
ant = AntColony(network,num_ants)

for i in range(5):
    ant.time_step()
    ant.ant_cogito()

net_f = ant.net_f
n_v = 0
colony = ant.colony
for n in range(5):
    n_v = np.argmax(net_f[n_v])
    print(n_v)
    