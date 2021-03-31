import numpy as np
 
class Particle:
    def __init__(self, initial_position, initial_velocity, position_boundary, velocity_boundary):
        self.position = np.array(initial_position)
        self.velocity = np.array(initial_velocity)
        self.position_boundary = position_boundary
        self.velocity_boundary = velocity_boundary
        self.best_position = None
        self.initialized = False
        self.dim = len(initial_position)
    
    def fit(self, fitness_function):
        self.fitness = fitness_function(self.position)
        if (self.initialized == False) or (self.fitness < fitness_function(self.best_position)):
            self.best_position = self.position.copy()
            self.best_fitness = self.fitness.copy()
            self.initialized = True

    def update_attributes(self, w_inertia, C1, C2, global_best_position):
        w_inertia = 0.5
        C1 = 1
        C2 = 2
        rand1 = np.random.rand(len(self.position))
        rand2 = np.random.rand(len(self.position))
        cognitive_velocity = C1*rand1*(self.best_position - self.position)
        social_velocity = C2*rand2*(global_best_position - self.position)
        self.velocity = w_inertia*self.velocity + cognitive_velocity + social_velocity
        self.position = self.position + self.velocity
        for i in range(self.dim):
            if self.position[i] > self.position_boundary[i][1]:
                self.position[i] = self.position_boundary[i][1]
            if self.position[i] < self.position_boundary[i][0]:
                self.position[i] = self.position_boundary[i][0]
    
class PSO:
    def __init__(self, size, n_iter, fitness_function, position_boundary, velocity_boundary, w_inertia, C1, C2, random_state, save_history):
        self.size = size
        self.n_iter = n_iter
        self.fitness_function = fitness_function
        self.position_boundary = position_boundary
        self.velocity_boundary = velocity_boundary
        self.w_inertia = w_inertia
        self.C1 = C1
        self.C2 = C2
        self.save_history = save_history
        np.random.seed(random_state)

    def simulate(self):
        # Initialize
        swarm = []
        for i in range(self.size):
            position = []
            velocity = []
            for pos_bound in self.position_boundary:
                pos_ = np.random.uniform(pos_bound[0], pos_bound[1])
                position.append(pos_)
            for vel_bound in self.velocity_boundary: 
                vel_ = np.random.uniform(vel_bound[0], vel_bound[1])
                velocity.append(vel_)
            initial_particle = Particle(position, velocity, self.position_boundary, self.velocity_boundary)
            swarm.append(initial_particle)

        if self.save_history == True:
            swarm_history = []
            swarm_history.append([x.position for x in swarm])

        # Optimization
        iteration = 0
        global_best_fitness = -1
        while iteration < self.n_iter:
            temp = []

            for j in range(self.size):
                swarm[j].fit(self.fitness_function)
                if (swarm[j].fitness < global_best_fitness) or global_best_fitness == -1:
                    global_best_position = swarm[j].position
                    global_best_fitness = swarm[j].fitness

            for j in range(self.size):
                swarm[j].update_attributes(self.w_inertia, self.C1, self.C2, global_best_position)
                temp.append(swarm[j].position)

            iteration += 1
            print(f'iteration: {iteration} | best_position : {global_best_position} | best_fit : {global_best_fitness}')
            if self.save_history == True:
                swarm_history.append(temp)

        self.sol = global_best_position
        if self.save_history == True:
            self.swarm_history = swarm_history