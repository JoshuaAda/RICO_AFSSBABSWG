import numpy as np
from casadi import *




class Four_Tank:
    def __init__(self):
        self._generate_casadi_nlp()
        self._generate_gurobi_model()

    def _generate_casadi_nlp(self):
        A = np.array([[0.923, 0, 0.181, 0],
                      [0, 0.946, 0, 0.149],
                      [0, 0, 0.811,0],
                      [0,0,0,0.846]])
        self.nx = A.shape[1]
        B = np.array([[0.283, 0.023],
                      [0.012, 0.306],
                      [0, 0.216],
                      [0.144, 0]])

        self.nu = B.shape[1]
        self.nz = B.shape[1]
        self.N = 20

        x = SX.sym("x", self.nx, 1)
        u = SX.sym("u", self.nu, 1)
        z = SX.sym("z", self.nz, 1)
        z_old = SX.sym("z_old", self.nz, 1)
        h_bar = np.array([12.4, 12.7, 1.8, 1.4])
        phi = vertcat(
            2.0 * np.sqrt(h_bar[0])
            * (sqrt(h_bar[0] + x[0]) - np.sqrt(h_bar[0])),

            2.0 * np.sqrt(h_bar[1])
            * (sqrt(h_bar[1] + x[1]) - np.sqrt(h_bar[1])),

            2.0 * np.sqrt(h_bar[2])
            * (sqrt(h_bar[2] + x[2]) - np.sqrt(h_bar[2])),

            2.0 * np.sqrt(h_bar[3])
            * (sqrt(h_bar[3] + x[3]) - np.sqrt(h_bar[3])),
        )
        # Defining the system
        x_next = vertcat(
            x[0]
            + (A[0, 0] - 1.0) * phi[0]
            + A[0, 2] * phi[2],

            x[1]
            + (A[1, 1] - 1.0) * phi[1]
            + A[1, 3] * phi[3],

            x[2]
            + (A[2, 2] - 1.0) * phi[2],

            x[3]
            + (A[3, 3] - 1.0) * phi[3],
        ) + B @ u
        #x_next = A @ x + B @ u  # +0.01* x.T@x #+ B @ z#0.1 x.T@x#+0.01*x.T@x

        # Create the CasADi function
        system = Function("sys", [x, u], [x_next])  # z

        # Define Initial point
        x0 = SX.sym("x0", self.nx, 1)  # np.ones((4, 1))
        Q = 1
        Q = Q * np.diag(np.array([1,1,0,0]))
        R = 0.1
        R = np.diag(R * np.ones(self.nu))

        c_k=np.array([0.1,0.2]).reshape((1,self.nz))#0.1,0.2
        # stage cost
        stage_cost = x.T @ Q @ x + u.T @ R @ u+c_k@z#-z_old).T@(z-z_old)
        stage_cost_fcn = Function('stage_cost', [x, u, z], [stage_cost])


        # terminal cost
        terminal_cost = x.T @ Q @ x
        terminal_cost_fcn = Function('terminal_cost', [x], [terminal_cost])
        # state constraints
        lb_x = -1 * np.ones((self.nx, 1))#-4
        ub_x = 1 * np.ones((self.nx, 1))#4
        # input constraints
        lb_u = -0.25 * np.ones((self.nu, 1))#-1.5
        ub_u = 0.25 * np.ones((self.nu, 1))#1.5
        lb_z = 0 * np.ones((self.nz, 1))  # -3
        ub_z = 1 * np.ones((self.nz, 1))  # 3
        X = SX.sym("X", (self.N) * self.nx, 1)
        U = SX.sym("U", self.N * self.nu, 1)
        Z = SX.sym("Z", self.N * self.nz, 1)

        J = 0
        lb_X = []  # lower bound for X.
        ub_X = []  # upper bound for X
        lb_U = []  # lower bound for U
        ub_U = []  # upper bound for U
        lb_Z = []  # lower bound for Z
        ub_Z = []  # upper bound for Z
        g = []  # constraint expression g
        h = []
        g_solver = []
        lb_g = []  # lower bound for constraint expression g
        ub_g = []  # upper bound for constraint expression g
        discrete = [False for k in range(self.nx * (self.N) + self.nu * self.N)] + [True for k in
                                                                                    range(self.nz * self.N)]
        self.discrete = discrete
        for k in range(self.N):
            if k == 0:
                x_k = x0
                z_k_1=np.zeros((self.nz,1))
            else:
                x_k = X[(k - 1) * self.nx:(k) * self.nx, :]
                z_k_1=Z[(k-1) * self.nz:(k) * self.nz, :]

            x_k_next = X[(k) * self.nx:(k + 1) * self.nx, :]
            u_k = U[k * self.nu:(k + 1) * self.nu, :]
            z_k = Z[k * self.nz:(k + 1) * self.nz, :]
            # objective
            J += stage_cost_fcn(x_k, u_k, z_k)#,z_k_1)

            # equality constraints (system equation)
            x_k_next_calc = system(x_k, u_k)  # z_k

            h.append(x_k_next - x_k_next_calc)
            g.append(sum(z_k) - 1)
            g.append(-sum(z_k))
            g_solver.append(x_k_next - x_k_next_calc)
            g_solver.append(sum(z_k) - 1)
            g.append(u_k - 0.25 * z_k)  # 1.5
            g.append(-u_k - 0.25 * z_k)  # 1.5
            g_solver.append(u_k + 0.25 * z_k)
            g_solver.append(u_k - 0.25 * z_k)

            g.append(x_k - 1)  # 4
            g.append(-x_k -1)  # -4
            g_solver.append(x_k)
            # g.append(u_k)
            g_solver.append(u_k)

            lb_g.append(np.zeros((self.nx, 1)))
            ub_g.append(np.zeros((self.nx, 1)))
            lb_g.append(-1 * np.ones((1, 1)))
            ub_g.append(0 * np.ones((1, 1)))
            lb_g.append(np.zeros((2, 1)))
            ub_g.append(inf)
            ub_g.append(inf)
            lb_g.append(-inf)
            lb_g.append(-inf)
            ub_g.append(np.zeros((2, 1)))
            lb_g.append(lb_x)
            ub_g.append(ub_x)
            lb_g.append(lb_u)
            ub_g.append(ub_u)

        x_terminal = X[(self.N - 1) * self.nx:(self.N) * self.nx, :]
        J += terminal_cost_fcn(x_terminal)
        g.append(x_terminal-1)#-4
        g.append(-1-x_terminal)#-4
        g_solver.append(x_terminal)
        lb_g.append(lb_x)
        ub_g.append(ub_x)
        z_lb = SX.sym("z_lb", self.N * self.nz, 1)
        z_ub = SX.sym("z_ub", self.N * self.nz, 1)
        for k in range(self.N):
            z_k = Z[k * self.nu:(k + 1) * self.nu, :]
            # lb= np.array([0,0])#p[2*k * self.nu:(2*k+1) * self.nu, :]
            # ub=np.array([1,1])#p[(2*k+1) * self.nu:2*(k+1) * self.nu, :]
            z_lb_k = z_lb[k * self.nz:(k + 1) * self.nz]
            z_ub_k = z_ub[k * self.nz:(k + 1) * self.nz]
            lb_g.append(-np.inf * np.ones((self.nz * 2, 1)))
            ub_g.append(np.zeros((self.nz * 2, 1)))
            g.append(-z_k + z_lb_k)
            g.append(z_k - z_ub_k)
            # g_solver.append(z_k)
            g_solver.append(-z_k + z_lb_k)
            g_solver.append(z_k - z_ub_k)

        self.lbz = vertcat(*lb_Z)
        self.ubz = vertcat(*ub_Z)
        self.lbx = vertcat(*lb_X, *lb_U)
        self.ubx = vertcat(*ub_X, *ub_U)
        self.x_lb = lb_X
        self.x_ub = ub_X
        self.discrete = discrete
        x = vertcat(X, U, Z)
        w0 = 0 * np.ones((x.shape[0], 1))
        g = vertcat(*g)
        h = vertcat(*h)
        g_solver = vertcat(*g_solver)
        self.lbg = vertcat(*lb_g)
        self.ubg = vertcat(*ub_g)
        n_h = h.size()[0]
        n_g = g.size()[0]
        n_w = x.size()[0]
        p = vertcat(x0, z_lb, z_ub)
        n_p = p.size()[0]

        self.prob = {'f': J, 'x': x, 'g': g_solver, 'p': p}
        self.nlp = {
            "w": x,
            "p": p,
            "f": J,
            "h": h,
            "g": g,
            "n_w": int(n_w),
            "n_p": int(n_p),
            "n_h": int(n_h),
            "n_g": int(n_g),
            "n_z": int(n_w + n_h + n_g),
            "w0": w0,
            "n_x": self.nx,
            "n_u": self.nu + self.nz,
            "N": self.N,
        }

    def _generate_gurobi_model(self):
        """Build Gurobi MIQP model matching CasADi formulation."""
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError:
            print("Warning: Gurobi not installed. Skipping _generate_gurobi_model()")
            self.model_gurobi = None
            self.model_gurobi_vars = None
            return

        A = np.array([
            [0.923, 0.0, 0.181, 0.0],
            [0.0, 0.946, 0.0, 0.149],
            [0.0, 0.0, 0.811, 0.0],
            [0.0, 0.0, 0.0, 0.846],
        ])

        B = np.array([
            [0.283, 0.023],
            [0.012, 0.306],
            [0.0, 0.216],
            [0.144, 0.0],
        ])

        Q = np.diag([1.0, 1.0, 0.0, 0.0])
        R = 0.1 * np.eye(self.nu)
        c_k = np.array([0.1, 0.2])

        self.model_gurobi = gp.Model("Linearized_Four_Tank_MIQP")
        self.model_gurobi.Params.OutputFlag = 1#0
        self.model_gurobi.Params.Cuts = 0
        self.model_gurobi.Params.Presolve = 0
        self.model_gurobi.Params.Heuristics = 0

        X = self.model_gurobi.addMVar(
            shape=(self.N + 1, self.nx),
            lb=-1.0,
            ub=1.0,
            name="X",
        )

        U = self.model_gurobi.addMVar(
            shape=(self.N, self.nu),
            lb=-0.25,
            ub=0.25,
            name="U",
        )

        Z = self.model_gurobi.addMVar(
            shape=(self.N, self.nz),
            lb=0.0,
            ub=1.0,
            vtype=GRB.BINARY,
            name="Z",
        )

        for k in range(self.N):
            h_bar = np.array([12.4, 12.7, 1.8, 1.4])


            phi_0 = (
                    2.0 * np.sqrt(h_bar[0])
                    * (
                            gp.nlfunc.sqrt(h_bar[0] + X[k, 0])
                            - np.sqrt(h_bar[0])
                    )
            )

            phi_1 = (
                    2.0 * np.sqrt(h_bar[1])
                    * (
                            gp.nlfunc.sqrt(h_bar[1] + X[k, 1])
                            - np.sqrt(h_bar[1])
                    )
            )

            phi_2 = (
                    2.0 * np.sqrt(h_bar[2])
                    * (
                            gp.nlfunc.sqrt(h_bar[2] + X[k, 2])
                            - np.sqrt(h_bar[2])
                    )
            )

            phi_3 = (
                    2.0 * np.sqrt(h_bar[3])
                    * (
                            gp.nlfunc.sqrt(h_bar[3] + X[k, 3])
                            - np.sqrt(h_bar[3])
                    )
            )

            self.model_gurobi.addConstr(
                X[k + 1, 0]
                == X[k, 0]
                + (A[0, 0] - 1.0) * phi_0
                + A[0, 2] * phi_2
                + B[0, 0] * U[k, 0]
                + B[0, 1] * U[k, 1]
            )

            self.model_gurobi.addConstr(
                X[k + 1, 1]
                == X[k, 1]
                + (A[1, 1] - 1.0) * phi_1
                + A[1, 3] * phi_3
                + B[1, 0] * U[k, 0]
                + B[1, 1] * U[k, 1]
            )

            self.model_gurobi.addConstr(
                X[k + 1, 2]
                == X[k, 2]
                + (A[2, 2] - 1.0) * phi_2
                + B[2, 0] * U[k, 0]
                + B[2, 1] * U[k, 1]
            )

            self.model_gurobi.addConstr(
                X[k + 1, 3]
                == X[k, 3]
                + (A[3, 3] - 1.0) * phi_3
                + B[3, 0] * U[k, 0]
                + B[3, 1] * U[k, 1]
            )

            self.model_gurobi.addConstr(
                Z[k].sum() <= 1
            )

            self.model_gurobi.addConstr(
                U[k] + 0.25 * Z[k] >= 0
            )

            self.model_gurobi.addConstr(
                U[k] - 0.25 * Z[k] <= 0
            )

        objective = 0

        for k in range(self.N):
            objective += (
                    X[k] @ Q @ X[k]
                    + U[k] @ R @ U[k]
                    + c_k @ Z[k]
            )

        objective += X[self.N] @ Q @ X[self.N]

        self.model_gurobi.setObjective(objective, GRB.MINIMIZE)

        self.model_gurobi_vars = {
            "X": X,
            "U": U,
            "Z": Z,
        }
class Linearized_Four_Tank:
    def __init__(self):
        self._generate_casadi_nlp()
        self._generate_gurobi_model()

    def _generate_casadi_nlp(self):
        A = np.array([[0.923, 0, 0.181, 0],
                      [0, 0.946, 0, 0.149],
                      [0, 0, 0.811,0],
                      [0,0,0,0.846]])
        self.nx = A.shape[1]
        B = np.array([[0.283, 0.023],
                      [0.012, 0.306],
                      [0, 0.216],
                      [0.144, 0]])

        self.nu = B.shape[1]
        self.nz = B.shape[1]
        self.N = 20

        x = SX.sym("x", self.nx, 1)
        u = SX.sym("u", self.nu, 1)
        z = SX.sym("z", self.nz, 1)
        z_old = SX.sym("z_old", self.nz, 1)
        # Defining the system
        x_next = A @ x + B @ u  # +0.01* x.T@x #+ B @ z#0.1 x.T@x#+0.01*x.T@x

        # Create the CasADi function
        system = Function("sys", [x, u], [x_next])  # z

        # Define Initial point
        x0 = SX.sym("x0", self.nx, 1)  # np.ones((4, 1))
        Q = 1
        Q = Q * np.diag(np.array([1,1,0,0]))
        R = 0.1
        R = np.diag(R * np.ones(self.nu))

        c_k=np.array([0.1,0.2]).reshape((1,self.nz))#0.1,0.2
        # stage cost
        stage_cost = x.T @ Q @ x + u.T @ R @ u+c_k@z#-z_old).T@(z-z_old)
        stage_cost_fcn = Function('stage_cost', [x, u, z], [stage_cost])


        # terminal cost
        terminal_cost = x.T @ Q @ x
        terminal_cost_fcn = Function('terminal_cost', [x], [terminal_cost])
        # state constraints
        lb_x = -1 * np.ones((self.nx, 1))#-4
        ub_x = 1 * np.ones((self.nx, 1))#4
        # input constraints
        lb_u = -0.25 * np.ones((self.nu, 1))#-1.5
        ub_u = 0.25 * np.ones((self.nu, 1))#1.5
        lb_z = 0 * np.ones((self.nz, 1))  # -3
        ub_z = 1 * np.ones((self.nz, 1))  # 3
        X = SX.sym("X", (self.N) * self.nx, 1)
        U = SX.sym("U", self.N * self.nu, 1)
        Z = SX.sym("Z", self.N * self.nz, 1)

        J = 0
        lb_X = []  # lower bound for X.
        ub_X = []  # upper bound for X
        lb_U = []  # lower bound for U
        ub_U = []  # upper bound for U
        lb_Z = []  # lower bound for Z
        ub_Z = []  # upper bound for Z
        g = []  # constraint expression g
        h = []
        g_solver = []
        lb_g = []  # lower bound for constraint expression g
        ub_g = []  # upper bound for constraint expression g
        discrete = [False for k in range(self.nx * (self.N) + self.nu * self.N)] + [True for k in
                                                                                    range(self.nz * self.N)]
        self.discrete = discrete
        for k in range(self.N):
            if k == 0:
                x_k = x0
                z_k_1=np.zeros((self.nz,1))
            else:
                x_k = X[(k - 1) * self.nx:(k) * self.nx, :]
                z_k_1=Z[(k-1) * self.nz:(k) * self.nz, :]

            x_k_next = X[(k) * self.nx:(k + 1) * self.nx, :]
            u_k = U[k * self.nu:(k + 1) * self.nu, :]
            z_k = Z[k * self.nz:(k + 1) * self.nz, :]
            # objective
            J += stage_cost_fcn(x_k, u_k, z_k)#,z_k_1)

            # equality constraints (system equation)
            x_k_next_calc = system(x_k, u_k)  # z_k

            h.append(x_k_next - x_k_next_calc)
            g.append(sum(z_k) - 1)
            g.append(-sum(z_k))
            g_solver.append(x_k_next - x_k_next_calc)
            g_solver.append(sum(z_k) - 1)
            g.append(u_k - 0.25 * z_k)  # 1.5
            g.append(-u_k - 0.25 * z_k)  # 1.5
            g_solver.append(u_k + 0.25 * z_k)
            g_solver.append(u_k - 0.25 * z_k)

            g.append(x_k - 1)  # 4
            g.append(-x_k -1)  # -4
            g_solver.append(x_k)
            # g.append(u_k)
            g_solver.append(u_k)

            lb_g.append(np.zeros((self.nx, 1)))
            ub_g.append(np.zeros((self.nx, 1)))
            lb_g.append(-1 * np.ones((1, 1)))
            ub_g.append(0 * np.ones((1, 1)))
            lb_g.append(np.zeros((2, 1)))
            ub_g.append(inf)
            ub_g.append(inf)
            lb_g.append(-inf)
            lb_g.append(-inf)
            ub_g.append(np.zeros((2, 1)))
            lb_g.append(lb_x)
            ub_g.append(ub_x)
            lb_g.append(lb_u)
            ub_g.append(ub_u)

        x_terminal = X[(self.N - 1) * self.nx:(self.N) * self.nx, :]
        J += terminal_cost_fcn(x_terminal)
        g.append(x_terminal-1)#-4
        g.append(-1-x_terminal)#-4
        g_solver.append(x_terminal)
        lb_g.append(lb_x)
        ub_g.append(ub_x)
        z_lb = SX.sym("z_lb", self.N * self.nz, 1)
        z_ub = SX.sym("z_ub", self.N * self.nz, 1)
        for k in range(self.N):
            z_k = Z[k * self.nu:(k + 1) * self.nu, :]
            # lb= np.array([0,0])#p[2*k * self.nu:(2*k+1) * self.nu, :]
            # ub=np.array([1,1])#p[(2*k+1) * self.nu:2*(k+1) * self.nu, :]
            z_lb_k = z_lb[k * self.nz:(k + 1) * self.nz]
            z_ub_k = z_ub[k * self.nz:(k + 1) * self.nz]
            lb_g.append(-np.inf * np.ones((self.nz * 2, 1)))
            ub_g.append(np.zeros((self.nz * 2, 1)))
            g.append(-z_k + z_lb_k)
            g.append(z_k - z_ub_k)
            # g_solver.append(z_k)
            g_solver.append(-z_k + z_lb_k)
            g_solver.append(z_k - z_ub_k)

        self.lbz = vertcat(*lb_Z)
        self.ubz = vertcat(*ub_Z)
        self.lbx = vertcat(*lb_X, *lb_U)
        self.ubx = vertcat(*ub_X, *ub_U)
        self.x_lb = lb_X
        self.x_ub = ub_X
        self.discrete = discrete
        x = vertcat(X, U, Z)
        w0 = 0 * np.ones((x.shape[0], 1))
        g = vertcat(*g)
        h = vertcat(*h)
        g_solver = vertcat(*g_solver)
        self.lbg = vertcat(*lb_g)
        self.ubg = vertcat(*ub_g)
        n_h = h.size()[0]
        n_g = g.size()[0]
        n_w = x.size()[0]
        p = vertcat(x0, z_lb, z_ub)
        n_p = p.size()[0]

        self.prob = {'f': J, 'x': x, 'g': g_solver, 'p': p}
        self.nlp = {
            "w": x,
            "p": p,
            "f": J,
            "h": h,
            "g": g,
            "n_w": int(n_w),
            "n_p": int(n_p),
            "n_h": int(n_h),
            "n_g": int(n_g),
            "n_z": int(n_w + n_h + n_g),
            "w0": w0,
            "n_x": self.nx,
            "n_u": self.nu + self.nz,
            "N": self.N,
        }

    def _generate_gurobi_model(self):
        """Build Gurobi MIQP model matching CasADi formulation."""
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError:
            print("Warning: Gurobi not installed. Skipping _generate_gurobi_model()")
            self.model_gurobi = None
            self.model_gurobi_vars = None
            return

        A = np.array([
            [0.923, 0.0, 0.181, 0.0],
            [0.0, 0.946, 0.0, 0.149],
            [0.0, 0.0, 0.811, 0.0],
            [0.0, 0.0, 0.0, 0.846],
        ])

        B = np.array([
            [0.283, 0.023],
            [0.012, 0.306],
            [0.0, 0.216],
            [0.144, 0.0],
        ])

        Q = np.diag([1.0, 1.0, 0.0, 0.0])
        R = 0.1 * np.eye(self.nu)
        c_k = np.array([0.1, 0.2])

        self.model_gurobi = gp.Model("Linearized_Four_Tank_MIQP")
        self.model_gurobi.Params.OutputFlag = 1
        self.model_gurobi.Params.Cuts = 0
        self.model_gurobi.Params.Presolve = 0
        self.model_gurobi.Params.Heuristics = 0

        X = self.model_gurobi.addMVar(
            shape=(self.N + 1, self.nx),
            lb=-1.0,
            ub=1.0,
            name="X",
        )

        U = self.model_gurobi.addMVar(
            shape=(self.N, self.nu),
            lb=-0.25,
            ub=0.25,
            name="U",
        )

        Z = self.model_gurobi.addMVar(
            shape=(self.N, self.nz),
            lb=0.0,
            ub=1.0,
            vtype=GRB.BINARY,
            name="Z",
        )

        for k in range(self.N):
            self.model_gurobi.addConstr(
                X[k + 1] == A @ X[k] + B @ U[k]
            )

            self.model_gurobi.addConstr(
                Z[k].sum() <= 1
            )

            self.model_gurobi.addConstr(
                U[k] + 0.25 * Z[k] >= 0
            )

            self.model_gurobi.addConstr(
                U[k] - 0.25 * Z[k] <= 0
            )

        objective = 0

        for k in range(self.N):
            objective += (
                    X[k] @ Q @ X[k]
                    + U[k] @ R @ U[k]
                    + c_k @ Z[k]
            )

        objective += X[self.N] @ Q @ X[self.N]

        self.model_gurobi.setObjective(objective, GRB.MINIMIZE)

        self.model_gurobi_vars = {
            "X": X,
            "U": U,
            "Z": Z,
        }

        }