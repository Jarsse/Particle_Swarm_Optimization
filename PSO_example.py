import matplotlib.pyplot as plt
import numpy as np
import math
import random


def banana(X, Y):
    """
    :param X: x locations
    :param Y: y locations
    :return: height at [X,Y]
    """
    return ((1 - X) ** 2) + 100 * (Y - X ** 2.0) ** 2.0


def schwefel(X, Y):
    """
    :param X: x location
    :param Y: y location
    :return: height at [X,Y]
    """
    return 418.9829 * 2 - (X * math.sin(math.sqrt(abs(X))) + Y * math.sin(math.sqrt(abs(Y))))


def np_schwefel(array):
    total = np.dot(array.T, np.sin(np.sqrt(np.abs(array))))
    return 418.9829 * len(array) - total


x, y, z = [], [], []  # locations and height
iterations = 1000  # max num of iterations
rangeXY = [[-500, 500, 1], [-2, 2, .1]]  # min, max, step
func = 0  # schwefel = 0, banana = 1


def main():
    global rangeXY
    global func
    rangeXY = rangeXY[func]
    generate_function(rangeXY[0], rangeXY[1], rangeXY[2])
    swarm = Swarm(100)
    plot_func(swarm, rangeXY[0], rangeXY[1])


def generate_function(min_range, max_range, step):
    """
    updates x, y, z values based on the chosen function func
    :param min_range: minimum x,y-range of the function
    :param max_range: maximum x,y-range of the function
    :param step: length of the step
    :return: nothing
    """
    global x
    global y
    global z
    global func
    x = np.arange(min_range, max_range + .1, step)
    y = np.arange(min_range, max_range + .1, step)
    x, y = np.meshgrid(x, y)
    if func == 0:
        z = schwefel_function(x, y)
    elif func == 1:
        z = banana(x, y)


def schwefel_function(X, Y):
    """
    Generates Z-values for the schwefel function.
    :param X: x values
    :param Y: y values
    :return: z values for schwefel function
    """
    output = np.zeros(X.shape)
    for i in range(len(X[0])):
        for j in range(len(Y[0])):
            output[i][j] = (schwefel(X[i][j], Y[i][j]))
    return output


def plot_func(swarm, min_range, max_range):
    """
    Plots the function, best found location, and drone movements.
    Stops when max iterations reached, 0 height has been found, or new best height hasn't been found in a while.
    Updates Swarm and drones each iteration.
    :param swarm: current Swarm
    :type swarm: Swarm
    :param min_range: minimum x, y range
    :param max_range: maximum x, y range
    """
    global x
    global y
    global z
    global iterations
    fig = plt.figure(figsize=(6, 5))
    i = 0  # iteration num
    ax = fig.add_subplot()  # subplot for drawing the drones and best_location
    map_func = plt.contourf(x, y, z, 25)  # draw function as background
    fig.colorbar(map_func)  # add color bar to the fig
    while swarm.best_height > 0 and i <= iterations and swarm.last_update < iterations*.2:
        if(len(ax.lines)) > 0:
            ax.lines.clear()  # clear the list of drone and best_location drawings
        plt.title("Best Height: {}\nIterations: {}/{}".format(swarm.best_height, i, iterations))
        for drone in swarm.drones:
            # plot each drone, blue if drone is searching alone, red otherwise
            if drone.search_alone:
                plt.plot([drone.previous_pos[0], drone.x], [drone.previous_pos[1], drone.y], "bo", linestyle="-",
                         markersize=.1)
            else:
                plt.plot([drone.previous_pos[0], drone.x], [drone.previous_pos[1], drone.y], "ro", linestyle="-",
                         markersize=.1)
        swarm.update_swarm(min_range, max_range)
        for best_loc in swarm.best_location:
            plt.plot(best_loc[0], best_loc[1], "w+")  # plot each best_location

        plt.draw()  # draw updates and pause
        plt.pause(0.00001)
        i += 1
    print(swarm.best_location)
    plt.show()


class Swarm:
    drones = []  # list of all the drones belonging to the swarm
    best_height = 1  # best height found by any drone in the swarm
    best_location = []  # location of the best_height
    last_update = 0  # number of updates since best_height was updated

    def __init__(self, num_drones=100):
        """
        Swarm of drones
        :param num_drones: number of drones in the swarm
        :type num_drones: int
        """
        if num_drones > 0:
            self.num_drones = num_drones
        else:
            self.num_drones = 100
        self.create_swarm(no_particles=num_drones)
        self.init_best_height()

    def create_swarm(self, no_particles=100, toggle_random=False):
        """
        Create and initialize drones for the amount of no_particles, either evenly across area or randomly.
        Add each Drone to swarm.drones.

        :param no_particles: number of particles added to the swarm
        :type no_particles: int
        :param toggle_random: True if random initial location, False if evenly distributed
        :type toggle_random: bool
        :return: Nothing
        """
        global rangeXY  # area where particles move. rangeXY[0] = min x,y value, [1] = max x,y value
        min_range = rangeXY[0]
        max_range = rangeXY[1]
        step_size = (max_range-min_range)/no_particles

        if not toggle_random:
            # locations for evenly distributed drones
            temp_x = np.array(np.linspace(min_range, max_range, math.ceil(np.sqrt(no_particles))))
            temp_y = np.array(np.linspace(min_range, max_range, math.ceil(np.sqrt(no_particles))))
            mesX, mesY = np.meshgrid(temp_x, temp_y)
            locations = np.vstack([mesX.ravel(), mesY.ravel()])

        for i in range(no_particles):
            if toggle_random:
                # initialize drones with random initialization
                drone = Drone(i, random.randrange(min_range, max_range, 1),
                              random.randrange(min_range, max_range, 1), step_size, swarm=self)
            else:
                # initialize drones with even distribution
                drone = Drone(i, locations[0, i],
                              locations[1, i], step_size, swarm=self)
            self.drones.append(drone)  # add each drone to swarm.drones
        self.init_best_height()  # initialize starting height for swarm

    def update_swarm(self, min_range, max_range):
        """
        Adds 1 to swarm.last_update and updates drone locations

        :param min_range: minimum x,y value of the area
        :param max_range: maximum x,y value of the area
        :return: nothing
        """
        self.last_update += 1
        for drone in self.drones:
            drone.update(min_range, max_range)

    def init_best_height(self):
        """
        initialize best_height and best_location for swarm.
        Checks every drone's height and picks best height and location and saves them into
        swarm's best_height and best_location.
        :return: nothing
        """
        if len(self.drones) > 0:
            self.best_height = self.drones[0].get_height()
            self.best_location = [[self.drones[0].x, self.drones[1].y]]
            for drone in self.drones:
                if drone.get_height() <= self.best_height:
                    self.best_height = drone.get_height()
                    self.best_location = [[drone.x, drone.y]]
                elif drone.get_height() == self.best_height:
                    if not self.best_location.__contains__([drone.x, drone.y]):
                        self.best_location.append([drone.x, drone.y])


class Drone:
    stepCount = 0  # how many updates drone has gone through
    velocity = [0, 0]  # velocity of the drone
    search_alone = True  # Drone searches alone in its neighbourhood in the beginning

    def __init__(self, name, X, Y, step_size, swarm):
        """
        Drone that belongs to a swarm. Searches for the best positions in function
        :param name: name of the drone
        :param X: initial x location of the drone
        :type X: float
        :param Y: initial y location of the drone
        :type Y: float
        :param step_size: step size of the drone. Used in updating drone's location
        :type step_size: float
        :param swarm: Swarm the drone belongs into
        :type swarm: Swarm
        """
        self.name = name
        self.x = X
        self.y = Y
        self.previous_pos = [self.x, self.y]
        self.step_size = step_size
        self.best_location = [[self.x, self.y]]
        self.lowest = self.get_height()
        self.swarm = swarm

    def get_height(self):
        """
        calculates the height at drone's current location.
        :return: drone's current height based on the current function
        """
        global func
        if func == 0:
            return schwefel(self.x, self.y)
        elif func == 1:
            return banana(self.x, self.y)

    def update(self, min_range, max_range):
        """
        Update drone's location [x,y]
        :param min_range: minimum x,y range
        :param max_range: maximum x,y range
        """
        self.update_heights()
        self.previous_pos = [self.x, self.y]

        # if search_alone is true, particle is searching alone around it's vicinity
        if self.search_alone:
            c0 = 0.7
            c1 = 1.2
            c2 = 0.4
            c3 = 1.6
            if random.random() < .08:
                # stop searching alone and start moving towards the swarm's best location
                self.search_alone = False
        else:
            # else particle searches around the swarm's best location
            c0 = 0.7  # inertia
            c1 = 1.6  # self confidence
            c2 = 2.4  # swarm confidence
            c3 = 0  # discovery confidence
        rand1 = random.uniform(0, 1 - (1/math.log(self.stepCount+3.1, 3)))
        rand2 = random.uniform(0, 1 - (1/math.log2(self.stepCount+2.1)))

        # current motion
        self.velocity = [c0 * self.velocity[0], c0 * self.velocity[1]]

        # particle memory influence
        x1 = self.best_location[0][0] - self.x
        y1 = self.best_location[0][1] - self.y
        pmi = [c1 * rand1 * x1 / (math.sqrt(x1 ** 2 + y1 ** 2) + 0.000001),
               c1 * rand1 * y1 / (math.sqrt(x1 ** 2 + y1 ** 2) + 0.000001)]

        # swarm influence
        x2 = self.swarm.best_location[0][0] - self.x
        y2 = self.swarm.best_location[0][1] - self.y
        si = [c2 * rand2 * x2 / (math.sqrt(x2 ** 2 + y2 ** 2) + 0.000001),
              c2 * rand2 * y2 / (math.sqrt(x2 ** 2 + y2 ** 2) + 0.000001)]

        # random influence
        discovery = [c3 * random.uniform(-1, 1), c3 * random.uniform(-1, 1)]

        self.x += (self.velocity[0] + pmi[0] + si[0] + discovery[0]) * self.step_size
        self.y += (self.velocity[1] + pmi[1] + si[1] + discovery[1]) * self.step_size
        # out of bounds handling
        if self.x < min_range:
            self.x = min_range
        elif self.x > max_range:
            self.x = max_range
        if self.y < min_range:
            self.y = min_range
        elif self.y > max_range:
            self.y = max_range
        self.stepCount += 1

    def update_heights(self):
        """
        updates drone's best height if height at current location is better than drone's previous best,
        updates swarm's best_height and best_location if drone.lowest is better.
        """
        if self.get_height() < self.lowest:
            self.lowest = self.get_height()
            self.best_location = [[self.x, self.y]]
            if self.lowest < self.swarm.best_height:
                self.swarm.best_height = self.lowest
                self.swarm.best_location = [[self.x, self.y]]
                self.swarm.last_update = 0
            elif self.lowest == self.swarm.best_height:
                if not self.swarm.best_location.__contains__([self.x, self.y]):
                    self.swarm.best_location.append([self.x, self.y])
        elif self.get_height() == self.lowest:
            self.best_location.append([self.x, self.y])


main()
