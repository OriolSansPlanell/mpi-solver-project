#!/usr/bin/env python

import numpy as np

class Grid:
    """
        Generate a computational grid and apply boundary conditions.
    """

    def __init__(self, nx=10, ny=10, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0):

        self.xmin, self.xmax, self.ymin, self.ymax = xmin, xmax, ymin, ymax
        self.nx, self.ny = nx, ny

        self.dx = (xmax - xmin) / (nx - 1)

        self.dy = (ymax - ymin) / (ny - 1)

        self.u = np.zeros((ny, nx), dtype=np.double)

    def set_boundary_condition(self, side='top', boundary_condition_function=lambda x, y: 0.0):

        xmin, ymin = self.xmin, self.ymin

        xmax, ymax = self.xmax, self.ymax

        x = np.arange(xmin, xmax + self.dx * 0.5, self.dx)

        y = np.arange(ymin, ymax + self.dy * 0.5, self.dy)

        if side == 'bottom':
            self.u[0, :] = boundary_condition_function(xmin, y)
        elif side == 'top':
            self.u[-1, :] = boundary_condition_function(xmin, y)
        elif side == 'left':
            self.u[:, 0] = boundary_condition_function(xmin, y)
        elif side == 'right':
            self.u[:, -1] = boundary_condition_function(xmin, y)


class LaplaceSolver(Grid):
    """
        Solves Laplace equation in 2D
    """

    def iterate(self):
        """
            A Python (slow) implementation of a finite difference iteration
        """

        u = self.u

        nx, ny = u.shape

        dx2, dy2 = self.dx ** 2, self.dy ** 2

        err = 0.0

        for i in range(1, nx - 1):

            for j in range(1, ny - 1):
                tmp = u[i, j]

                u[i, j] = ((u[i - 1, j] + u[i + 1, j]) * dy2 +
                           (u[i, j - 1] + u[i, j + 1]) * dx2) / (dx2 + dy2) / 2

                diff = u[i, j] - tmp

                err += diff * diff

        return np.sqrt(err)

    def solve(self, max_iterations=10000, tolerance=1.0e-16, printing=True):
        """
            Calls iterate() sequentially until the error is reduced below a tolerance.
        """

        for i in range(max_iterations):

            error = self.iterate()

            if error < tolerance:
                if printing:
                    print("Solution converged in " + str(i) + " iterations.")
                break

    def get_solution(self):
        return self.u

class LaplaceSolverMPI(LaplaceSolver):

    def __init__(self, comm, **kwargs):
        self.comm = comm
        self.rank = comm.rank
        self.size = comm.size
        super().__init__(**kwargs)

    def set_boundary_condition(self, **kwargs):
        if self.rank == 0:
            super().set_boundary_condition(**kwargs)

    def partition(self, data):

        if self.rank == 0:
            self.n_x, self.n_y = data.shape
            if self.size >= self.n_x:
                self.size = self.n_x

            arr = np.array_split(data, self.size)
            dummyrow = np.zeros((1, self.nx), dtype=np.double)

            for i in range(len(arr)):
                if i == 0:
                    arr[i] = np.concatenate((arr[i], dummyrow))
                elif i == len(arr) - 1:
                    arr[i] = np.concatenate((dummyrow, arr[i]))
                else:
                    arr[i] = np.concatenate((dummyrow, arr[i], dummyrow))

        elif self.rank > self.size - 1:
            pass

        else:
            data = []
            arr = []

        data = self.comm.scatter(arr, root=0)

        return data

    def solve(self, max_iterations=10000, tolerance=1.0e-16, printing=True):

        self.u = self.partition(self.u)

        for i in range(max_iterations):
            if self.rank == 0:
                self.comm.send(self.u[-2], dest=(self.rank + 1))
                self.u[-1] = self.comm.recv(source=(self.rank + 1))
            elif self.rank == self.size - 1:
                self.comm.send(self.u[1], dest=(self.rank - 1))
                self.u[0] = self.comm.recv(source=(self.rank - 1))
            elif self.rank > self.size - 1:
                pass
            else:
                self.comm.send(self.u[-2], dest=(self.rank + 1))
                self.comm.send(self.u[1], dest=(self.rank - 1))
                self.u[-1] = self.comm.recv(source=(self.rank + 1))
                self.u[0] = self.comm.recv(source=(self.rank - 1))

            error = self.iterate()

            if i % 50 == 0 and i > 0:
                reduced_error = self.comm.allreduce(error)
                if reduced_error < tolerance:
                    if printing:
                        print("Solution converged in " + str(i) + " iterations.")
                    break

    def get_solution(self):

        if self.rank == 0:
            self.u = np.delete(self.u, -1, 0)
        elif self.rank == self.size - 1:
            self.u = np.delete(self.u, 0, 0)
        elif self.rank > self.size - 1:
            pass

        else:
            self.u = np.delete(self.u, 0, 0)
            self.u = np.delete(self.u, -1, 0)
        newarray = self.comm.gather(self.u, root=0)
        if self.rank == 0:
            solution = np.vstack(newarray)
            return solution
        else:
            pass


if __name__ == "__main__":

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    print("<<---Number of parallel cores: {}".format(comm.size))

    l = LaplaceSolverMPI(comm, nx=10, ny=9)
    l.set_boundary_condition(side='top', boundary_condition_function=lambda x, y: 10)
    l.set_boundary_condition(side='bottom', boundary_condition_function=lambda x, y: 10)

    l.solve()
    u = l.get_solution()
    if comm.rank == 0:
        print(u)
