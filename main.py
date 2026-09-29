from casadi import *
import numpy as np
import time
import torch
import sys
import dataclasses
from optimization import Optimization_Problem,Linearized_Four_Tank,Lokta_Volterra,QP_Problem,Nonlinear_double, Four_Tank
from cusadi.src.lisco.nn.predictor import Predictor
from cusadi.src.lisco.nn.solver import Solver
from cusadi.src.nlp_handler import NLPHandler
from cusadi.src.lisco.nn.nn_models import FeedforwardNN,FeedforwardNNConfig
import scipy
import gurobipy as gp
from gurobipy import GRB
import importlib
from pathlib import Path

#np.set_printoptions(treshold=sys.maxsize)
common_path = Path(__file__).parent / "common.py"
spec = importlib.util.spec_from_file_location("common", common_path)
common_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common_module)
project_root = common_module.project_root
class Node:
    def __init__(self,z_lb,z_ub,depth=0,parent_id=None,branch_var=None,branch_value=None,branched_list=[]):

        self.lbz=z_lb
        self.ubz=z_ub
        self.lb=inf
        self.var=inf
        self.is_integer=False
        self.pruned=False
        self.branched=False
        self.branched_list=branched_list
        self.depth=depth
        self.parent_id=parent_id
        self.branch_var=branch_var
        self.branch_value=branch_value
        self.zk=None
        self.Tk=0
class branch_and_bound_solver:

    def __init__(self,lbx,ubx,lbg,ubg,lbz,ubz,prob,discrete,model_gurobi,gurobi_vars,nx,nu,nz,N,r,num_candidates=40,device="cuda",max_iter=1000):
        #self._generate_casadi_nlp()
        #self._generate_gurobi_model()
        self.lbz = lbz
        self.ubz = ubz
        self.lbx = lbx
        self.ubx = ubx
        self.lbg = lbg
        self.ubg = ubg
        self.prob=prob
        self.model_gurobi=model_gurobi
        self.model_gurobi_vars=gurobi_vars
        self.nx=nx
        self.nu=nu
        self.nz=nz
        self.N=N
        self.discrete=discrete
        nmpc = Four_Tank()#Linearized_Four_Tank()#Four_Tank()#Linearized_Four_Tank()#Nonlinear_double()#QP_Problem()#Lokta_Volterra()#Linearized_Four_Tank()#Optimization_Problem()
        nlp = NLPHandler(
            nmpc,
            dtype=torch.float64,
            device=device,
            eps=1e-8,
            sigma=0.01,
            rho=1e-2,
            regularize=True,
        )
        self.device=device
        n_in, n_out = nlp.n_p, nlp.n_z
        nn_model = FeedforwardNN(
            n_in=n_in,
            n_out=n_out,
        )
        nn_model = nn_model.to(device)

        self.predictor=Predictor(model=nn_model,
            nlp=nlp)
        self.predictor_fixed=Predictor(model=nn_model,
            nlp=nlp)
        n_in, n_out = nlp.n_p + nlp.n_z + 1, nlp.n_z
        nn_model = FeedforwardNN(
            n_in=n_in,
            n_out=n_out,
        )
        nn_model = nn_model.to(device)
        #self.lisco=Solver()
        predictor_dir = Path("BandB_casadi/cusadi/results/predictor_training/20260907_151401__single")#20260903_110924__single")#20260907_151401__single")#20260903_110924__single")#20260827_101507__single")#")#20260902_233401__single")#20260827_101507__single")#20260831_085508__single")#20260830_195600__single")#20260826_192453__single")#20260825_210057__single")#20260825_163325__single")#20260821_081231__single")#20260821_135515__single")#20260821_081231__single")#20260820_224223__single")#20260820_215015__single")#20260820_111901__single")#20260820_092317__single")#20260819_224321__single")#20260819_145734__single")#20260818_174858__single")#20260818_111422__single")#20260817_151525__single")#20260626_220030__single")#20260805_084456__single")#20260626_220030__single")#20260606_095445__single")#20260604_215313
        if not predictor_dir.is_absolute():
            predictor_dir = (project_root() / predictor_dir).resolve()
        self.predictor=self.predictor.load(predictor_dir,nlp)

        predictor_fixed_dir = Path(
            "BandB_casadi/cusadi/results/predictor_training/20260918_090053__single")#20260912_125717__single")#20260903_234244__single")  # 20260827_101507__single")#")#20260902_233401__single")#20260827_101507__single")#20260831_085508__single")#20260830_195600__single")#20260826_192453__single")#20260825_210057__single")#20260825_163325__single")#20260821_081231__single")#20260821_135515__single")#20260821_081231__single")#20260820_224223__single")#20260820_215015__single")#20260820_111901__single")#20260820_092317__single")#20260819_224321__single")#20260819_145734__single")#20260818_174858__single")#20260818_111422__single")#20260817_151525__single")#20260626_220030__single")#20260805_084456__single")#20260626_220030__single")#20260606_095445__single")#20260604_215313
        if not predictor_fixed_dir.is_absolute():
            predictor_dir = (project_root() / predictor_fixed_dir).resolve()
        self.predictor_fixed = self.predictor_fixed.load(predictor_dir, nlp)

        self.lisco = Solver(
            model=nn_model,
            nlp=nlp,
            predictor=self.predictor)



        solver_dir = Path("BandB_casadi/cusadi/results/solver_training/20260907_174734_mono_pred")#")#20260903_164940_mono_pred")#20260907_174734_mono_pred")#20260903_164940_mono_pred")#20260903_132835_mono_pred")#20260903_101858_mono_pred")#20260902_231050_mono_pred")#20260903_020906_mono_pred")#20260902_231050_mono_pred")#20260902_234845_mono_pred")#20260902_231050_mono_pred")#20260829_223731_mono_pred")#20260902_214021_mono_pred")#20260829_223731_mono_pred")#20260831_101659_mono_pred")#20260831_091555_mono_pred")#20260827_000722_mono_pred")#20260826_194043_mono_pred")#20260825_211727_mono_pred")#20260825_164506_mono_pred")#20260822_110113_mono_pred")#20260821_160152_mono_pred")#20260821_101740_mono_pred")#20260821_003557_mono_pred")#20260820_222227_mono_pred")#20260820_153119_mono_pred")#20260820_114518_mono_pred")#20260820_093944_mono_pred")#20260819_230100_mono_pred")#20260819_151332_mono_pred")#20260818_180541_mono_pred")#20260818_112958_mono_pred")#20260817_204843_mono_pred")###20260626_222450_mono_pred")#20260805_090759_mono_pred")#20260608_152702_mono_pred")  #20260607_141703_mono_pred")#20260604_220204
        if not solver_dir.is_absolute():
            solver_dir = (project_root() / solver_dir).resolve()
        self.lisco=self.lisco.load(solver_dir,nlp)

        self.lisco_fixed = Solver(
            model=nn_model,
            nlp=nlp,
            predictor=self.predictor_fixed)

        solver_dir_fixed = Path(
            "BandB_casadi/cusadi/results/solver_training/20260918_092054_mono_pred")#20260912_131313_mono_pred")#20260903_164940_mono_pred")#20260910_202402_mono_pred")  # 20260903_164940_mono_pred")#20260903_132835_mono_pred")#20260903_101858_mono_pred")#20260902_231050_mono_pred")#20260903_020906_mono_pred")#20260902_231050_mono_pred")#20260902_234845_mono_pred")#20260902_231050_mono_pred")#20260829_223731_mono_pred")#20260902_214021_mono_pred")#20260829_223731_mono_pred")#20260831_101659_mono_pred")#20260831_091555_mono_pred")#20260827_000722_mono_pred")#20260826_194043_mono_pred")#20260825_211727_mono_pred")#20260825_164506_mono_pred")#20260822_110113_mono_pred")#20260821_160152_mono_pred")#20260821_101740_mono_pred")#20260821_003557_mono_pred")#20260820_222227_mono_pred")#20260820_153119_mono_pred")#20260820_114518_mono_pred")#20260820_093944_mono_pred")#20260819_230100_mono_pred")#20260819_151332_mono_pred")#20260818_180541_mono_pred")#20260818_112958_mono_pred")#20260817_204843_mono_pred")###20260626_222450_mono_pred")#20260805_090759_mono_pred")#20260608_152702_mono_pred")  #20260607_141703_mono_pred")#20260604_220204
        if not solver_dir_fixed.is_absolute():
            solver_dir_fixed = (project_root() / solver_dir_fixed).resolve()
        self.lisco_fixed = self.lisco_fixed.load(solver_dir_fixed, nlp)

        #self.lisco.setup_for_batch_eval("reduce-overhead")
        self.num_candidates=num_candidates
        batch_size = self.comb(self.num_candidates,r) * (2 ** r)
        print(f"Setting up LISCO with batch size: {batch_size} for r={r}")
        self.lisco.nlp.setup_for_batch(batch_size,force_rebuild=False)
        self.current_batch_size = batch_size
        #self.lisco.nlp.setup_for_batch(80)#,force_rebuild=True)
        #self.lisco.setup_for_single_eval(nn_device="cuda",kkt_device="cuda",kkt_backend="cusadi_batch",nn_compile_mode="compile",batch_size=batch_size)

        #p = torch.zeros(84, self.current_batch_size).to('cuda').T
        #self.lisco.solve_batch(p_batch=p,
        #                       max_iter=70,
        #                       alpha=1.0,
        #                       predictor=self.predictor,
        #                       update_mode="full",
        #                       return_trajectory=False)
        #torch.cuda.synchronize()
        self.solver = nlpsol('solver', 'ipopt', self.prob, {'ipopt.print_level': 5, 'print_time': 1, 'ipopt.sb': 'no'})
        self.solver_bonmin = nlpsol('solver', 'bonmin', self.prob, {'discrete': self.discrete})
        self.node_list=[]
        self.upper=inf
        self.upper_var=0
        self.nodes_visited=0
        self.max_iter=max_iter
        self.start_time=None
        self.branch_policy = None
        self.branch_policy_feature_names = []

    def solve_gurobi(self, x0):
        """Solve Gurobi model with given initial state x0"""
        if self.model_gurobi is None:
            raise RuntimeError("Gurobi model not initialized. Is Gurobi installed?")
        
        # Get variable references
        X = self.model_gurobi_vars['X']
        U = self.model_gurobi_vars['U']
        Z = self.model_gurobi_vars['Z']
        
        # Set initial condition: X[0] = x0
        for i in range(self.nx):
            X[0, i].LB = x0[i, 0]
            X[0, i].UB = x0[i, 0]
        
        # Optimize
        self.model_gurobi.optimize()
        
        # Extract objective and solution
        if self.model_gurobi.status == 2:  # Optimal
            obj_value = self.model_gurobi.ObjVal
            # Construct solution vector in same format as CasADi (X, U, Z concatenated)
            sol_X = X.X.flatten()
            sol_U = U.X.flatten()
            sol_Z = Z.X.flatten()
            solution = np.concatenate([sol_X, sol_U, sol_Z]).reshape(-1, 1)
            return obj_value, solution
        else:
            # Model not solved optimally
            return inf, None
    
    def solve_bonmin(self,x0):
        lbg=self.lbg
        ubg=self.ubg
        z_lb=np.zeros((self.N*self.nz,1))
        z_ub = np.ones((self.N *self.nz, 1))
        p=np.concatenate((x0,z_lb,z_ub))

        res=self.solver_bonmin(p=p,ubg=ubg,lbg=lbg)
        J=float(res['f'])
        x=res['x'].full()
        return J,x

    def get_node_tk_values(self):
        """
        Return the final Tk value stored for every node in self.node_list.

        The list index corresponds to the node ID/index in self.node_list.
        Missing or invalid values are represented by np.nan.
        """
        tk_values = []

        for node in self.node_list:
            value = getattr(node, "Tk", None)

            if value is None:
                tk_values.append(np.nan)
                continue

            try:
                # Handles torch tensors, including CUDA tensors.
                if hasattr(value, "detach"):
                    value = value.detach().cpu().numpy()

                value = np.asarray(value).reshape(-1)

                if value.size == 0:
                    tk_values.append(np.nan)
                else:
                    tk_values.append(float(value[0]))

            except (TypeError, ValueError, RuntimeError):
                tk_values.append(np.nan)

        return tk_values
    def solve_local_prob(self,x0,id,lisco=False):
        t1=time.time()
        node=self.node_list[id]
        z_lb=node.lbz
        z_ub=node.ubz
        p = np.concatenate((x0, z_lb, z_ub))
        if lisco:
            p = np.repeat(p, self.current_batch_size)
            p=p.reshape((-1, self.current_batch_size)).T
            p=torch.tensor(p, dtype=self.lisco.nlp.dtype, device=self.device)
            #z0=self.predictor.predict(p)
            #torch.cuda.synchronize()
            #z_solved=self.lisco.solve_fast_gpu(p=p,z0=z0,max_iter=70)
            z_solved, step_times = self.lisco.solve_batch(
                p_batch=p,
                max_iter=100,#self.max_iter,#self.max_iter,
                alpha=1.0,
                predictor=self.predictor,
                update_mode="full",#"safeguarded_armijo",#"full",
                return_trajectory=False,
                #z0_batch=torch.zeros(80,608,device=self.device),
            )
            Tk = self.lisco.nlp.Tk_batch_func(z_solved, p).reshape(-1)
            #print(sum(step_times))
            node.lb=float(self.get_cost_from_z(z_solved,p,self.current_batch_size)[0][0][0])
            node.var=self.get_primal_from_z(z_solved,0)
            node.zk=z_solved[0,:]
            node.Tk=Tk[0]

            t2=time.time()
            print("Overall time :"+str(t2-t1))
        else:
            res=self.solver( p=p,ubg=self.ubg,lbg=self.lbg)
            node.lb=float(res['f'])
            node.var=res['x'].full()
    
    def solve_relaxation_with_bounds(self, x0, z_lb, z_ub):
        """Solve NLP relaxation with given z bounds (for strong branching)"""

        p = np.concatenate((x0, z_lb, z_ub))
        res = self.solver(p=p,ubg=self.ubg, lbg=self.lbg)
        x=res['x'].full()
        return float(res['f']),x
    def decide_next_node(self):
        branchable_indices = [i for i, node in enumerate(self.node_list) 
                             if not node.is_integer and not node.pruned and not node.branched]
        if not branchable_indices:
            return None
        #print(len(branchable_indices))
        val=max(branchable_indices, key=lambda i: self.node_list[i].depth)
        indices = [x for i, x in enumerate(branchable_indices) if self.node_list[x].depth == self.node_list[val].depth]

        return min(branchable_indices, key=lambda i: self.node_list[i].lb)#max(branchable_indices, key=lambda i: self.node_list[i].depth)#min(branchable_indices, key=lambda i: self.node_list[i].lb)#max(branchable_indices, key=lambda i: self.node_list[i].depth)##max(branchable_indices, key=lambda i: len(self.node_list[i].branched_list))#min(branchable_indices, key=lambda i: self.node_list[i].lb)
    def decide_next_branch_variable(self,id):
        z_solution = self.node_list[id].var[self.nx * (self.N) + self.nu * self.N:]
        fractions = np.abs(z_solution - np.round(z_solution))
        return int(np.argmax(fractions))
    
    def solve_branch_strong(self, id, x0, num_candidates=5,lisco=False):
        """Strong branching: evaluate top candidate variables and pick the best"""
        node = self.node_list[id]
        z_solution = node.var[self.nx * (self.N) + self.nu * self.N:]
        
        # Get top fractional variables
        fractions = np.abs(z_solution - np.round(z_solution))
        candidate_indices = np.array([k for k in range(num_candidates) if k not in node.branched_list])#np.argsort(fractions.ravel())[-num_candidates:]
        #print(candidate_indices)
        best_score = -inf
        best_variable = candidate_indices[0]
        if lisco:
            p_list=[]
            for var_idx in candidate_indices:
                branch_value = np.clip(z_solution[var_idx][0],0.001,0.999)

                # Branch DOWN: z_idx <= floor(value)
                z_ub_down = DM(node.ubz)
                z_ub_down[var_idx] = max(floor(branch_value), node.lbz[var_idx])
                p=np.concatenate((x0,node.lbz,z_ub_down))
                p_list.append(p)
                z_lb_up = DM(node.lbz)

                z_lb_up[var_idx] = min(ceil(branch_value), node.ubz[var_idx])
                p = np.concatenate((x0, z_lb_up, node.ubz))
                p_list.append(p)
            for m in range(40-len(candidate_indices)):
                p_list.append(p)
                p_list.append(p)
            p = torch.tensor(p_list, dtype=self.lisco.nlp.dtype, device=self.device).squeeze()
            z_solved, step_times = self.lisco.solve_batch(
                p_batch=p,
                max_iter=self.max_iter,
                alpha=1.0,
                predictor=self.predictor,
                update_mode="full",#"safeguarded_armijo",#"full",
                return_trajectory=False,
            )
            #print(sum(step_times))
            feasibility_mask=np.array(self.lisco.nlp.Tk_batch_func(z_solved,p).detach().cpu()>1e-5)
            #print(np.array(p[0,4:44].cpu().detach()).tolist())
            #print(np.array(p[0, 44:].cpu().detach()).tolist())
            costs = np.array(self.get_cost_from_z(z_solved, p,self.current_batch_size))
            costs[feasibility_mask]=-np.inf
            cost_node=[0.8*min(costs[2*k]-node.lb,costs[2*k+1]-node.lb)+0.2*max(costs[2*k]-node.lb,costs[2*k+1]-node.lb) for k in range(num_candidates)]
            best_candidate=np.argmax(np.array(cost_node)[:len(candidate_indices)])
            best_score = cost_node[best_candidate]
            best_variable = candidate_indices[best_candidate]
            best_z_ub_down = p_list[2*best_candidate][self.nx+self.nu*self.N:]
            best_z_lb_up = p_list[2*best_candidate+1][self.nx:self.nx+self.nu*self.N]
            best_lb_down = costs[2*best_candidate]
            best_lb_up = costs[2*best_candidate+1]
            best_sol_up = self.get_primal_from_z(z_solved,2*best_candidate+1)
            best_sol_down = self.get_primal_from_z(z_solved,2*best_candidate)

        else:
            # Evaluate each candidate
            lb_list=[]
            sol_list=[]
            for var_idx in candidate_indices:
                branch_value = np.clip(z_solution[var_idx][0],0.00001,0.99999)#z_solution[var_idx][0]

                # Branch DOWN: z_idx <= floor(value)
                z_ub_down = DM(node.ubz)
                z_ub_down[var_idx] = max(floor(branch_value), node.lbz[var_idx])
                try:
                    lb_down,sol_down = self.solve_relaxation_with_bounds(x0, node.lbz, z_ub_down)
                except:
                    lb_down =- inf
                lb_list.append(lb_down)
                sol_list.append(sol_down)
                # Branch UP: z_idx >= ceil(value)
                z_lb_up = DM(node.lbz)
                z_lb_up[var_idx] = min(ceil(branch_value), node.ubz[var_idx])
                try:
                    lb_up,sol_up = self.solve_relaxation_with_bounds(x0, z_lb_up, node.ubz)
                except:
                    lb_up = -inf
                lb_list.append(lb_up)
                sol_list.append(sol_up)
                # Score: pick variable that maximizes the minimum bound
                lb=node.lb
                score = 0.3*min(lb_down-lb, lb_up-lb)+0.7*max(lb_down-lb, lb_up-lb)

                if score > best_score:
                    best_score = score
                    best_variable = var_idx
                    best_z_ub_down=z_ub_down
                    best_z_lb_up=z_lb_up
                    best_lb_down=lb_down
                    best_lb_up=lb_up
                    best_sol_up=sol_up
                    best_sol_down=sol_down
        #print(best_variable)
        return int(best_variable),best_z_ub_down,best_z_lb_up,best_lb_down,best_lb_up,best_sol_down,best_sol_up
    def is_integer_feasible(self, z_array, tolerance=1e-2):
        return np.all(np.abs(z_array - np.round(z_array)) < tolerance)
    def is_integer_feasible_torch(self,z_batch,tolerance=1e-3):
        return torch.all(torch.abs(z_batch-torch.round(z_batch))<tolerance)
    def check_current_sol(self,id):
        node = self.node_list[id]
        z_solution = node.var[self.nx * (self.N) + self.nu * self.N:]
        if self.is_integer_feasible(z_solution):
            node.is_integer = True
            if node.lb < self.upper:
                self.upper = node.lb
                self.upper_var=node.var
                self.upper_tk=node.Tk
                #self.upper_id=id

    def prune_nodes(self):
        for node in self.node_list:
            if node.lb>self.upper or node.lb==-np.inf:
                node.pruned=True
    
    def _print_tree_node(self, node_id, lines, children_map, is_last=True, prefix=""):
        node = self.node_list[node_id]
        if node.branched:
            status = "B"
        elif node.is_integer:
            status = "I"
        elif node.pruned:
            status = "P"
        else:
            status = "O"
        lb_str = f"{node.lb:.3f}" if node.lb != inf else "inf"
        connector = "└── " if is_last else "├── "
        val_str = f"{torch.log10(node.Tk):.3f}" #if node.branch_value is not None else "0.000"
        lines.append(f"{prefix}{connector}[{node_id}] var={node.branch_var}, val={val_str}, lb={lb_str}, status={status}")
        extension = "    " if is_last else "│   "
        children = children_map.get(node_id, [])
        for i, child_id in enumerate(children):
            self._print_tree_node(child_id, lines, children_map, i == len(children) - 1, prefix + extension)
    
    def write_tree_summary(self, filename='tree_summary.txt'):
        with open(filename, 'w') as f:
            # Header and statistics
            f.write("=" * 80 + "\n")
            f.write("BRANCH AND BOUND TREE SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            max_depth = max([node.depth for node in self.node_list]) if self.node_list else 0
            total_nodes = len(self.node_list)
            pruned_nodes = sum([1 for node in self.node_list if node.pruned]).full()[0][0]
            integer_nodes = sum([1 for node in self.node_list if node.is_integer]).full()[0][0]
            branched_nodes = sum([1 for node in self.node_list if node.branched]).full()[0][0]
            f.write(f"Total Nodes: {total_nodes}\n")
            f.write(f"Pruned Nodes: {pruned_nodes}\n")
            f.write(f"Integer Feasible: {integer_nodes}\n")
            f.write(f"Branched Nodes: {branched_nodes}\n")
            f.write(f"Max Depth: {max_depth}\n\n")
            
            # ASCII Tree visualization
            f.write("TREE STRUCTURE (O=Open, P=Pruned, I=Integer, B=Branched):\n")
            f.write("-" * 80 + "\n")
            children_map = {}
            for idx, node in enumerate(self.node_list):
                if node.parent_id is not None:
                    if node.parent_id not in children_map:
                        children_map[node.parent_id] = []
                    children_map[node.parent_id].append(idx)
            if self.node_list:
                root_node = self.node_list[0]
                status = "I" if root_node.is_integer else ("P" if root_node.pruned else "O")
                lb_str = f"{root_node.lb:.3f}" if root_node.lb != inf else "inf"
                f.write(f"[0] lb={lb_str}, status={status}\n")
                tree_lines = []
                children = children_map.get(0, [])
                for i, child_id in enumerate(children):
                    self._print_tree_node(child_id, tree_lines, children_map, i == len(children) - 1, "")
                for line in tree_lines:
                    f.write(line + "\n")
            f.write("-" * 80 + "\n\n")
            
            # Tabular format
            f.write("DETAILED NODE INFORMATION:\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'ID':>3} | {'Parent':>6} | {'Depth':>5} | {'Var':>3} | {'Value':>8} | {'LowerBound':>12} | {'Status':>12}\n")
            f.write("-" * 100 + "\n")
            for idx, node in enumerate(self.node_list):
                parent_str = str(node.parent_id) if node.parent_id is not None else "-"
                var_str = str(node.branch_var) if node.branch_var is not None else "-"
                val_str = f"{node.branch_value:.3f}" if node.branch_value is not None else "-"
                lb_str = f"{node.lb:.6f}" if node.lb != inf else "inf"
                if node.is_integer:
                    status = "integer"
                elif node.branched:
                    status = "branched"
                elif node.pruned:
                    status = "pruned"
                else:
                    status = "open"
                f.write(f"{idx:3d} | {parent_str:>6} | {node.depth:5d} | {var_str:>3} | {val_str:>8} | {lb_str:>12} | {status:>12}\n")
            f.write("-" * 100 + "\n")


    def get_cost_from_z(self,primal,p,batch_size):
        x0=np.array(p[:,:self.nx].detach().cpu()).reshape((batch_size,self.nx,-1))
        Q=np.diag([1,1,0,0])#[0.8,0.8])#0.2*np.eye(self.nx)
        R=0.1*np.eye(self.nu)
        x=np.array(primal[:,0:self.N*self.nx].cpu().detach()).reshape((batch_size,-1,self.nx))
        u=np.array(primal[:,self.N*self.nx:self.N*(self.nu+self.nx)].cpu().detach()).reshape((batch_size,-1,self.nu))
        z = np.array(primal[:, self.N * (self.nx+self.nu):self.N * (self.nu + self.nx+self.nz)].cpu().detach()).reshape(
            (batch_size, -1, self.nz))
        #z_old=np.concatenate((np.zeros((batch_size,1,self.nz)),z[:,:self.N-1,:]),axis=1)
        c_k=np.array([0.1,0.2]).reshape((-1,self.nz))
        #c_k@z[r,k,:]+
        #0.1*(z[r,k,:]-z_old[r,k,:]).T@(z[r,k,:]-z_old[r,k,:])
        #J=[np.sum([c_k@z[r,k,:]+x[r,k,:].T @ Q @ x[r,k,:] + u[r,k,:].T @ R @ u[r,k,:] for k in range(self.N)])+x0[r].T@Q@x0[r] for r in range(batch_size)]
        J = [np.sum(
            [ c_k@z[r,k,:]+(x[r, k, :]).T @ Q @ (x[r, k, :]) + u[r, k, :].T @ R @ u[r, k, :] for k in range(self.N)]) +
             (x0[r]).T @ Q @ (x0[r]) for r in range(batch_size)]
        return J
    def get_primal_from_z(self,z,ind):
        return np.array(z[ind,0:self.N*(self.nx+self.nu+self.nz)].detach().cpu()).reshape(-1,1)
    def generate_upper_bound(self,x0,lisco=False):
        node=self.node_list[0]
        z_rel=node.var[self.nx*(self.N)+self.nu*self.N:]
        z_fixed=np.array([round(z_rel[k][0]) for k in range(len(z_rel))])
        z_lb=z_fixed.reshape((-1,1))
        z_ub=z_fixed.reshape((-1,1))


        p = np.concatenate((x0, z_lb, z_ub))
        if lisco:
            def randomized_cardinality_rounding(z_relaxed, n_trials):
                schedules = []

                for _ in range(n_trials):
                    schedule = torch.zeros_like(z_relaxed)

                    for k in range(z_relaxed.shape[0]):
                        probabilities = z_relaxed[k].clamp(min=0)
                        probabilities /= probabilities.sum()
                        mode = torch.multinomial(probabilities, 1)
                        schedule[k, mode] = 1

                    schedules.append(schedule)

                return schedules

            #z_lb = randomized_cardinality_rounding(torch.tensor(z_rel.reshape((-1,self.nu)), dtype=self.lisco.nlp.dtype, device=self.device), self.current_batch_size)
            p = np.repeat(p, self.current_batch_size)
            p = p.reshape((-1, self.current_batch_size)).T
            p = torch.tensor(p, dtype=self.lisco.nlp.dtype, device=self.device)
            #for k in range(len(z_lb)):
            #    p[k,4:44]=z_lb[k].flatten()
            #    p[k,44:84]=z_lb[k].flatten()
            z_solved, step_times = self.lisco_fixed.solve_batch(
                p_batch=p,
                max_iter=100,#self.max_iter,#100,#self.max_iter,
                alpha=1.0,
                predictor=self.predictor_fixed,
                update_mode="full",#"safeguarded_armijo",#"full",
                return_trajectory=False,
            )
            Tk = self.lisco.nlp.Tk_batch_func(z_solved, p).reshape(-1)
            feasible = Tk <= 1e-5

            # torch.max(Tk))
            #print(sum(step_times))
            val=self.get_cost_from_z(z_solved,p,self.current_batch_size)
            for k in range(len(val)):
                if ~feasible[k]:
                    val[k]=np.array([[1000]])#np.inf]])
            self.upper=min(val)[0][0]#[0][0][0]
            ind=np.argmin(np.array(val))
            self.upper_var=self.get_primal_from_z(z_solved,ind)
            self.upper_tk=Tk[ind]
        else:
            res = self.solver(p=p,ubg=self.ubg, lbg=self.lbg)
            self.upper=float(res['f'])
            self.upper_var=res['x'].full()
    
    def feasibility_pump(self, x0, z_relaxed, max_iterations=20):
        """Feasibility pump: iteratively round and resolve to find feasible integer solution"""
        z = np.array([z_relaxed[k][0] for k in range(len(z_relaxed))])
        
        for iteration in range(max_iterations):
            # Round to nearest integer
            z_rounded = np.round(z)
            
            # Fix Z and solve NLP
            z_lb = z_rounded
            z_ub = z_rounded
            lbx = vertcat(self.lbx, z_lb)
            ubx = vertcat(self.ubx, z_ub)
            lbx[:self.nx] = x0
            ubx[:self.nx] = x0
            
            try:
                res = self.solver(ubg=self.ubg, lbg=self.lbg, ubx=ubx, lbx=lbx)
                
                # Extract solution
                obj = float(res['f'])
                start_idx = self.nx*(self.N+1) + self.nu*self.N
                z_new = res['x'].full()[start_idx:]
                z = np.array([z_new[k][0] for k in range(len(z_new))])
                
                # Check if solution is integer feasible
                if self.is_integer_feasible(z):
                    # Found integer feasible solution!
                    if obj < self.upper:
                        self.upper = obj
                        self.upper_var = res['x'].full()
                    return True
                
                # Not integer feasible yet - continue pumping
                
            except:
                # Solver failed - try next iteration
                pass
        
        return False  # Failed to find feasible solution

    def comb(self, n, r):
        """Compute binomial coefficient C(n,r)"""
        if r > n or r < 0:
            return 0
        if r == 0 or r == n:
            return 1
        from math import factorial
        return factorial(n) // (factorial(r) * factorial(n - r))
    
    def get_all_r_combinations(self, candidate_indices, r):
        """Generate all r-combinations from candidate indices"""
        from itertools import combinations
        return list(combinations(candidate_indices, r))
    
    def get_all_2r_patterns(self, r):
        """Generate all 2^r fixing patterns (0=down, 1=up for each variable)"""
        patterns = []
        for i in range(2**r):
            pattern = [(i >> j) & 1 for j in range(r)]
            patterns.append(pattern)
        return patterns
    
    def solve_branch_strong_multi(self, id, x0, num_candidates, r, lisco=False):
        """Multi-variable strong branching: evaluate r-combinations with 2^r patterns"""

        node = self.node_list[id]
        #print(id)
        z_solution = node.var[self.nx * (self.N) + self.nu * self.N:]
        
        # Get available candidates (not already branched)
        fractions = np.abs(z_solution - np.round(z_solution))
        candidates= sorted(range(len(fractions)),key=lambda i: fractions[i], reverse=True)
        candidate_indices = np.array([candidates[k] for k in range(len(z_solution)) if candidates[k] not in node.branched_list and fractions[candidates[k]][0] >= 1e-2][:num_candidates])
        #if len(node.branched_list)==0:
        #    candidate_indices=[4,0,6,34,33,36,26,29,31,39]
        #else:
        #    candidate_indices=[24,18,21,22,2,16,11,12,8,15]
        # Generate r-combinations and 2^r patterns
        r_combos = self.get_all_r_combinations(candidate_indices, r)
        patterns_2r = self.get_all_2r_patterns(r)
        
        num_problems = len(r_combos) * len(patterns_2r)
        best_score = -inf
        best_combo_idx = 0
        t1=time.time()
        if lisco:
            device = torch.device(self.device)
            dtype = self.lisco.nlp.dtype
            n_combos = len(r_combos)
            n_patterns = len(patterns_2r)
            n_binary = self.N * self.nz

            combo_variables = torch.as_tensor(
                np.asarray(r_combos, dtype=np.int64),
                device=device,
                dtype=torch.long,
            )
            patterns = torch.as_tensor(
                np.asarray(patterns_2r, dtype=np.bool_),
                device=device,
                dtype=torch.bool,
            )

            x0_tensor = torch.as_tensor(
                np.asarray(x0).reshape(-1),
                device=device,
                dtype=dtype,
            )
            node_lb = torch.as_tensor(
                np.asarray(DM(node.lbz)).reshape(-1),
                device=device,
                dtype=dtype,
            )
            node_ub = torch.as_tensor(
                np.asarray(DM(node.ubz)).reshape(-1),
                device=device,
                dtype=dtype,
            )

            batch_size = self.current_batch_size if hasattr(self, 'current_batch_size') else num_problems
            base_p = torch.cat((x0_tensor, node_lb, node_ub))
            p = base_p.unsqueeze(0).expand(batch_size, -1).clone()
            z0_base=node.zk.clone()
            z0=z0_base.unsqueeze(0).expand(batch_size, -1).clone()
            problem_variables = (
                combo_variables[:, None, :]
                .expand(n_combos, n_patterns, r)
                .reshape(num_problems, r)
            )
            problem_patterns = (
                patterns[None, :, :]
                .expand(n_combos, n_patterns, r)
                .reshape(num_problems, r)
            )

            lb_offset = self.nx
            ub_offset = self.nx + n_binary
            rows = torch.arange(
                num_problems,
                device=device,
            )[:, None].expand(num_problems, r)

            current_lb = p[rows, lb_offset + problem_variables]
            current_ub = p[rows, ub_offset + problem_variables]
            target_columns = torch.where(
                problem_patterns,
                lb_offset + problem_variables,
                ub_offset + problem_variables,
            )
            target_values = torch.where(
                problem_patterns,
                current_ub,
                current_lb,
            )
            p[rows, target_columns] = target_values

            if num_problems < batch_size:
                p[num_problems:] = p[num_problems - 1]

            t2=time.time()
            #print("p_list time: " + str(t2 - t1))
            z_solved, step_times = self.lisco.solve_batch(
                p_batch=p,
                max_iter=self.max_iter,
                alpha=1.0,
                z0_batch=z0,
                #predictor=self.predictor,
                update_mode="full",#"safeguarded_armijo",#"full",
                return_trajectory=False,
            )
            t3=time.time()
            #print("LISCO time: "+str(t3-t2))

            Tk = self.lisco.nlp.Tk_batch_func(z_solved, p).reshape(-1)
            #torch.max(Tk))
            feasible = Tk[:num_problems] <= 1e-5

            x_end = self.N * self.nx
            u_end = x_end + self.N * self.nu
            z_end=u_end+self.N*self.nz
            x_batch = z_solved[:num_problems, :x_end].reshape(num_problems,self.N, self.nx)[:,:, :2]#[:num_problems, :x_end].reshape(
                #num_problems,
                #self.N,
                #self.nx-2,
            #)
            u_batch = z_solved[:num_problems, x_end:u_end].reshape(
                num_problems,
                self.N,
                self.nu,
            )
            z_batch = z_solved[:num_problems, u_end:z_end].reshape(
                num_problems,
                self.N,
                self.nz,
            )
            z_batch_old = z_solved[:num_problems, u_end:z_end-self.nz].reshape(
                num_problems,
                self.N-1,
                self.nz,
            )
            z_batch_old=torch.concatenate((torch.zeros(num_problems,1,self.nz,device=self.device),z_batch_old),axis=1)
            #integer=self.is_integer_feasible_torch(z_batch)
            c_k = torch.tensor(
                [0.1, 0.2],
                dtype=z_batch.dtype,
                device=z_batch.device,
            )
            x0_batch = p[:num_problems, :self.nx-2]
            costs = (
                1 * x_batch.square().sum(dim=(1, 2))#0.2
                + 0.1 * u_batch.square().sum(dim=(1, 2))#0.1
                + 1 * x0_batch.square().sum(dim=1)#0.2
                + (z_batch * c_k).sum(dim=(1, 2))
                #+0.1*(z_batch-z_batch_old).square().sum(dim=(1, 2))
            )
            #costs = (
            #    0.8 * (x_batch).square().sum(dim=(1, 2))#0.2
            #    + 0.1 * u_batch.square().sum(dim=(1, 2))#0.1
            #    + 0.8 * (x0_batch).square().sum(dim=1)#0.2
            #    + (z_batch * 0.0).sum(dim=(1, 2))
            # )
            costs = costs.masked_fill(~feasible, torch.inf)

            cost_matrix = costs.reshape(n_combos, n_patterns)

            cost_change_matrix = torch.clamp(
                cost_matrix - node.lb,
                min=0.0,
            )

            alpha = 0.1 if np.isfinite(self.upper) else 0.7

            scores = (
                    alpha * cost_change_matrix.amin(dim=1)
                    + (1.0 - alpha) * cost_change_matrix.amax(dim=1)
            )
            best_combo_idx = int(torch.argmax(scores).item())

            first_problem = best_combo_idx * n_patterns
            best_problem_indices = torch.arange(
                first_problem,
                first_problem + n_patterns,
                device=device,
            )
            best_Tk=Tk[best_problem_indices]
            best_zk=z_solved[best_problem_indices]
            best_params = p[best_problem_indices]
            primal_size = self.N * (self.nx + self.nu + self.nz)

            best_lb_array = (
                best_params[:, lb_offset:ub_offset]
                .detach()
                .cpu()
                .numpy()
            )
            best_ub_array = (
                best_params[:, ub_offset:ub_offset + n_binary]
                .detach()
                .cpu()
                .numpy()
            )
            best_cost_array = (
                cost_matrix[best_combo_idx]
                .detach()
                .cpu()
                .numpy()
            )
            best_solution_array = (
                z_solved[best_problem_indices, :primal_size]
                .detach()
                .cpu()
                .numpy()
            )

            best_combo_bounds = [
                (
                    DM(best_lb_array[k].reshape(-1, 1)),
                    DM(best_ub_array[k].reshape(-1, 1)),
                )
                for k in range(n_patterns)
            ]
            best_combo_costs = [
                float(best_cost_array[k])
                for k in range(n_patterns)
            ]
            best_combo_solutions = [
                best_solution_array[k].reshape(-1, 1)
                for k in range(n_patterns)
            ]
            best_var_combo = tuple(r_combos[best_combo_idx])
            t4 = time.time()
            #print("best combo time: " + str(t4 - t3))
            return best_var_combo, best_combo_bounds, best_combo_costs, best_combo_solutions,best_Tk,best_zk
    
    def solve_lisco_parallel(self, x0, r):
        """Solve using parallel r-variable branching"""
        # Calculate batch size and setup
        batch_size = self.comb(self.num_candidates, r) * (2 ** r)
        print(f"Setting up LISCO with batch size: {batch_size} for r={r}")
        #self.lisco.nlp.setup_for_batch(batch_size)
        self.current_batch_size = batch_size  # Store for use in solve_branch_strong_multi
        id = 0
        self.node_list.append(Node(np.zeros((self.N * self.nz, 1)), np.ones((self.N * self.nz, 1))))
        self.solve_local_prob(x0, id, lisco=True)
        self.start_time = time.time()
        self.node_list = []  # Reset for fresh run
        self.upper = inf
        self.upper_var = 0
        self.nodes_visited = 0

        self.node_list.append(Node(np.zeros((self.N * self.nz, 1)), np.ones((self.N * self.nz, 1))))

        self.solve_local_prob(x0, id, lisco=True)
        t1 = time.time()
        self.check_current_sol(len(self.node_list) - 1)
        self.generate_upper_bound(x0, lisco=True)
        t2=time.time()
        #print("upper and lower bound time:"+str(t2-t1))
        while True:
            id = self.decide_next_node()
            
            if id is None:
                break
            
            self.nodes_visited += 1
            node = self.node_list[id]
            branched_list = node.branched_list.copy()
            
            # Strong branching for r variables
            best_var_combo, combo_bounds, combo_costs, combo_solutions,Tks,zks = self.solve_branch_strong_multi(
                id, x0, num_candidates=self.num_candidates, r=r, lisco=True)
            t3=time.time()
            # Add r variables to branched_list
            for var_idx in best_var_combo:
                branched_list.append(var_idx)
            
            # Create 2^r children
            for pattern_idx, (z_lb, z_ub) in enumerate(combo_bounds):
                self.node_list.append(
                    Node(z_lb, z_ub, depth=node.depth + 1, parent_id=id, 
                         branch_var=best_var_combo,
                         branch_value=None,
                         branched_list=branched_list))
                self.node_list[-1].var = combo_solutions[pattern_idx]
                self.node_list[-1].zk=zks[pattern_idx]
                self.node_list[-1].lb = combo_costs[pattern_idx][0][0] if isinstance(combo_costs[pattern_idx], np.ndarray) else combo_costs[pattern_idx]
                self.node_list[-1].Tk = Tks[pattern_idx]
                self.check_current_sol(id=len(self.node_list) - 1)
            
            self.node_list[id].branched = True
            self.prune_nodes()
            self.write_tree_summary('tree_summary_parallel.txt')
            t4 = time.time()
            #print("rest time:" + str(t4 - t3))
        

        solve_time = time.time() - self.start_time
        #tk_values = self.get_node_tk_values()
        return self.upper_var, self.upper, self.nodes_visited, solve_time,self.upper_tk#tk_values

    def solve_lisco(self, x0):

        self.start_time = time.time()
        self.node_list.append(Node(np.zeros((self.N * self.nz, 1)), np.ones((self.N * self.nz,
                                                                             1))))  # self.lbg[(self.nx*2+self.nu*3+1)*self.N+self.nx:],self.ubg[(self.nx*2+self.nu*3+1)*self.N+self.nx:]))
        id = 0
        self.solve_local_prob(x0, id,lisco=True)
        self.generate_upper_bound(x0,lisco=True)
        # root_z = self.node_list[0].var[self.nx*(self.N+1)+self.nu*self.N:]
        # self.feasibility_pump(x0, root_z, max_iterations=20)
        while True:
            id = self.decide_next_node()

            if id is None:
                break

            self.nodes_visited += 1
            node = self.node_list[id]
            branched_list=node.branched_list.copy()
            #print(branched_list)
            #print(len(branched_list))
            ind_z,ubz_lower_branch,lbz_upper_branch,lb_down,lb_up,sol_down,sol_up = self.solve_branch_strong(id, x0, num_candidates=40,lisco=True)
            branched_list.append(ind_z)
            branch_value = node.var[ind_z + self.nx * (self.N) + self.nu * self.N][0]
            lbz_lower_branch = DM(node.lbz)
            #ubz_lower_branch = DM(node.ubz)
            #ubz_lower_branch[ind_z] = max(floor(branch_value), lbz_lower_branch[ind_z])
            self.node_list.append(
                Node(lbz_lower_branch, ubz_lower_branch, depth=node.depth + 1, parent_id=id, branch_var=ind_z,
                     branch_value=branch_value,branched_list=branched_list))
            self.node_list[-1].var=sol_down
            self.node_list[-1].lb = lb_down[0][0]
            #self.solve_local_prob(x0, len(self.node_list) - 1)
            self.check_current_sol(id=len(self.node_list) - 1)
            #lbz_upper_branch = DM(node.lbz)
            ubz_upper_branch = DM(node.ubz)
            #lbz_upper_branch[ind_z] = min(ceil(branch_value), ubz_upper_branch[ind_z])
            self.node_list.append(
                Node(lbz_upper_branch, ubz_upper_branch, depth=node.depth + 1, parent_id=id, branch_var=ind_z,
                     branch_value=branch_value,branched_list=branched_list))
            #self.solve_local_prob(x0, len(self.node_list) - 1)
            self.node_list[-1].var = sol_up
            self.node_list[-1].lb = lb_up[0][0]
            self.check_current_sol(id=len(self.node_list) - 1)
            self.node_list[id].branched = True
            self.prune_nodes()

        solve_time = time.time() - self.start_time
        return self.upper_var, self.upper, self.nodes_visited, solve_time
    def get_branching_feature_names(self):
        return [
            "relaxed_value",
            "fractionality",
            "lower_bound",
            "upper_bound",
            "stage_fraction",
            "mode_fraction",
            "node_depth_fraction",
            "node_lower_bound",
            "relative_incumbent_gap",
            "unfixed_fraction",
            *[f"x0_{index}" for index in range(self.nx)],
        ]

    def get_branching_features(self, node_id, x0, tolerance=1e-2):
        """Build one feature row for every fractional binary candidate."""
        node = self.node_list[node_id]
        binary_offset = self.nx * self.N + self.nu * self.N
        z_solution = np.asarray(node.var[binary_offset:]).reshape(-1)
        lower_bounds = np.asarray(DM(node.lbz)).reshape(-1)
        upper_bounds = np.asarray(DM(node.ubz)).reshape(-1)
        x0_flat = np.asarray(x0, dtype=np.float64).reshape(-1)

        if x0_flat.shape[0] != self.nx:
            raise ValueError(
                f"Expected x0 with {self.nx} entries, got {x0_flat.shape[0]}"
            )

        branched = set(node.branched_list)
        n_binary = self.N * self.nz
        unfixed = np.count_nonzero(lower_bounds < upper_bounds)
        node_bound = float(node.lb) if np.isfinite(float(node.lb)) else 0.0
        if np.isfinite(float(self.upper)):
            gap_scale = max(1.0, abs(float(self.upper)))
            relative_gap = max(float(self.upper) - node_bound, 0.0) / gap_scale
        else:
            relative_gap = 0.0

        candidate_indices = []
        feature_rows = []
        for variable_index in range(n_binary):
            if variable_index in branched:
                continue
            if lower_bounds[variable_index] >= upper_bounds[variable_index]:
                continue

            relaxed_value = float(np.clip(z_solution[variable_index], 0.0, 1.0))
            fractionality = min(relaxed_value, 1.0 - relaxed_value)
            if fractionality <= tolerance:
                continue

            stage = variable_index // self.nz
            mode = variable_index % self.nz
            stage_fraction = stage / max(self.N - 1, 1)
            mode_fraction = mode / max(self.nz - 1, 1)
            depth_fraction = node.depth / max(n_binary, 1)

            feature_rows.append([
                relaxed_value,
                fractionality,
                float(lower_bounds[variable_index]),
                float(upper_bounds[variable_index]),
                stage_fraction,
                mode_fraction,
                depth_fraction,
                node_bound,
                relative_gap,
                unfixed / max(n_binary, 1),
                *x0_flat.tolist(),
            ])
            candidate_indices.append(variable_index)

        if not candidate_indices:
            raise RuntimeError(
                f"Node {node_id} is not marked integer-feasible but has no "
                "fractional branching candidates"
            )

        return (
            np.asarray(candidate_indices, dtype=np.int64),
            np.asarray(feature_rows, dtype=np.float32),
        )

    def set_branching_policy(self, policy, feature_names=None):
        """Attach an already constructed branching policy to this solver."""
        expected_names = self.get_branching_feature_names()
        if policy.config.input_dim != len(expected_names):
            raise ValueError(
                f"Branching policy expects {policy.config.input_dim} features, "
                f"but this solver provides {len(expected_names)}"
            )
        if feature_names and list(feature_names) != expected_names:
            raise ValueError(
                "Branching-policy feature names do not match the solver features"
            )
        self.branch_policy = policy
        self.branch_policy.eval()
        self.branch_policy_feature_names = expected_names

    def load_branching_policy(self, checkpoint_path, device="cpu"):
        from branching.branching_policy import BranchingPolicy

        policy, feature_names = BranchingPolicy.load(
            checkpoint_path,
            map_location=device,
        )
        self.set_branching_policy(policy, feature_names=feature_names)
        return policy

    def decide_next_branch_variable_nn(self, node_id, x0):
        if self.branch_policy is None:
            raise RuntimeError(
                "No branching policy is loaded. Call load_branching_policy() "
                "or set_branching_policy() first."
            )

        candidate_indices, features = self.get_branching_features(node_id, x0)
        policy_device = next(self.branch_policy.parameters()).device
        feature_tensor = torch.as_tensor(
            features,
            dtype=torch.float32,
            device=policy_device,
        )
        with torch.no_grad():
            scores = self.branch_policy(feature_tensor)
        selected_position = int(torch.argmax(scores).item())
        return int(candidate_indices[selected_position])

    def evaluate_branching_teacher_parallel(
        self,
        node_id,
        x0,
        num_candidates=None,
        max_iter=1000,
    ):
        """Data-generation copy of solve_branch_strong_multi for r=1."""
        node = self.node_list[node_id]
        if num_candidates is None:
            num_candidates = self.num_candidates

        binary_offset = self.nx * self.N + self.nu * self.N
        z_solution = node.var[binary_offset:]
        fractions = np.abs(z_solution - np.round(z_solution))
        ordered = sorted(
            range(len(fractions)),
            key=lambda index: fractions[index],
            reverse=True,
        )
        candidate_indices = np.asarray(
            [
                index
                for index in ordered
                if index not in node.branched_list
            ][:num_candidates],
            dtype=np.int64,
        )
        if candidate_indices.size == 0:
            raise RuntimeError(
                f"Node {node_id} has no unbranched strong-branching candidates"
            )

        r_combos = self.get_all_r_combinations(candidate_indices, 1)
        patterns_2r = self.get_all_2r_patterns(1)
        n_combos = len(r_combos)
        n_patterns = len(patterns_2r)
        num_problems = n_combos * n_patterns
        batch_size = self.current_batch_size
        if num_problems > batch_size:
            raise ValueError(
                f"Teacher needs {num_problems} problems but batch size is "
                f"{batch_size}"
            )

        device = torch.device(self.device)
        dtype = self.lisco.nlp.dtype
        n_binary = self.N * self.nz
        combo_variables = torch.as_tensor(
            np.asarray(r_combos, dtype=np.int64),
            device=device,
            dtype=torch.long,
        )
        patterns = torch.as_tensor(
            np.asarray(patterns_2r, dtype=np.bool_),
            device=device,
            dtype=torch.bool,
        )
        x0_tensor = torch.as_tensor(
            np.asarray(x0).reshape(-1),
            device=device,
            dtype=dtype,
        )
        node_lb = torch.as_tensor(
            np.asarray(DM(node.lbz)).reshape(-1),
            device=device,
            dtype=dtype,
        )
        node_ub = torch.as_tensor(
            np.asarray(DM(node.ubz)).reshape(-1),
            device=device,
            dtype=dtype,
        )
        z0_base = node.zk.clone()
        z0 = z0_base.unsqueeze(0).expand(batch_size, -1).clone()
        base_p = torch.cat((x0_tensor, node_lb, node_ub))
        p = base_p.unsqueeze(0).expand(batch_size, -1).clone()
        problem_variables = (
            combo_variables[:, None, :]
            .expand(n_combos, n_patterns, 1)
            .reshape(num_problems, 1)
        )
        problem_patterns = (
            patterns[None, :, :]
            .expand(n_combos, n_patterns, 1)
            .reshape(num_problems, 1)
        )
        lb_offset = self.nx
        ub_offset = self.nx + n_binary
        rows = torch.arange(
            num_problems,
            device=device,
        )[:, None].expand(num_problems, 1)
        current_lb = p[rows, lb_offset + problem_variables]
        current_ub = p[rows, ub_offset + problem_variables]
        target_columns = torch.where(
            problem_patterns,
            lb_offset + problem_variables,
            ub_offset + problem_variables,
        )
        target_values = torch.where(
            problem_patterns,
            current_ub,
            current_lb,
        )
        p[rows, target_columns] = target_values
        if num_problems < batch_size:
            p[num_problems:] = p[num_problems - 1]

        z_solved, _ = self.lisco.solve_batch(
            p_batch=p,
            max_iter=max_iter,
            alpha=1.0,
            #predictor=self.predictor,
            z0_batch=z0,
            update_mode="full",#"safeguarded_armijo",#"full",
            return_trajectory=False,
        )
        Tk = self.lisco.nlp.Tk_batch_func(z_solved, p).reshape(-1)
        feasible = Tk[:num_problems] <= 1e-5

        x_end = self.N * self.nx
        u_end = x_end + self.N * self.nu
        z_end = u_end + self.N * self.nz
        x_batch = z_solved[:num_problems, :x_end].reshape(num_problems,self.N, self.nx)[:,:, :2]#z_solved[:num_problems, :x_end].reshape(
            #num_problems,
            #self.N,
            #self.nx,
        #)
        u_batch = z_solved[:num_problems, x_end:u_end].reshape(
            num_problems,
            self.N,
            self.nu,
        )
        z_batch = z_solved[:num_problems, u_end:z_end].reshape(
            num_problems,
            self.N,
            self.nz,
        )
        c_k = torch.tensor(
            [0.1, 0.2],
            dtype=z_batch.dtype,
            device=z_batch.device,
        )
        x0_batch = p[:num_problems, :self.nx - 2]
        #z_batch_old = z_solved[:num_problems, u_end:z_end - self.nz].reshape(
        #    num_problems,
        #    self.N-1,
        #    self.nz,
        #)
        #z_batch_old = torch.concatenate((torch.zeros(num_problems, 1, self.nz,device=self.device), z_batch_old),axis=1)
        costs = (
                1 * x_batch.square().sum(dim=(1, 2))  # 0.2
                + 0.1 * u_batch.square().sum(dim=(1, 2))  # 0.1
                + 1 * x0_batch.square().sum(dim=1)  # 0.2
                + (z_batch * c_k).sum(dim=(1, 2))
                #+ 0.1 * (z_batch - z_batch_old).square().sum(dim=(1, 2))
        )
        #costs = (
        #        0.8 * (x_batch).square().sum(dim=(1, 2))  # 0.2
        #        + 0.1 * u_batch.square().sum(dim=(1, 2))  # 0.1
        #        + 0.8 * (x0_batch).square().sum(dim=1)  # 0.2
        #        + (z_batch * 0.0).sum(dim=(1, 2))
        #)
        costs = costs.masked_fill(~feasible, -torch.inf)
        cost_matrix = costs.reshape(n_combos, n_patterns)
        cost_change_matrix = torch.clamp(
            cost_matrix - node.lb,
            min=0.0,
        )

        alpha = 0.1 if np.isfinite(self.upper) else 0.7

        scores = (
                alpha * cost_change_matrix.amin(dim=1)
                + (1.0 - alpha) * cost_change_matrix.amax(dim=1)
        )
        best_combo_idx = int(torch.argmax(scores).item())
        best_problem_indices = torch.arange(
            best_combo_idx * n_patterns,
            (best_combo_idx + 1) * n_patterns,
            device=device,
        )
        best_params = p[best_problem_indices]
        primal_size = self.N * (self.nx + self.nu + self.nz)

        best_lb_array = (
            best_params[:, lb_offset:ub_offset]
            .detach()
            .cpu()
            .numpy()
        )
        best_ub_array = (
            best_params[:, ub_offset:ub_offset + n_binary]
            .detach()
            .cpu()
            .numpy()
        )
        best_cost_array = (
            cost_matrix[best_combo_idx]
            .detach()
            .cpu()
            .numpy()
        )
        zk=z_solved[best_problem_indices]
        best_solution_array = (
            z_solved[best_problem_indices, :primal_size]
            .detach()
            .cpu()
            .numpy()
        )
        best_combo_bounds = [
            (
                DM(best_lb_array[index].reshape(-1, 1)),
                DM(best_ub_array[index].reshape(-1, 1)),
            )
            for index in range(n_patterns)
        ]
        best_combo_costs = [
            float(best_cost_array[index])
            for index in range(n_patterns)
        ]
        best_combo_solutions = [
            best_solution_array[index].reshape(-1, 1)
            for index in range(n_patterns)
        ]

        lower_bounds = np.asarray(DM(node.lbz)).reshape(-1)
        upper_bounds = np.asarray(DM(node.ubz)).reshape(-1)
        x0_flat = np.asarray(x0, dtype=np.float64).reshape(-1)
        unfixed = np.count_nonzero(lower_bounds < upper_bounds)
        node_bound = float(node.lb) if np.isfinite(float(node.lb)) else 0.0
        if np.isfinite(float(self.upper)):
            gap_scale = max(1.0, abs(float(self.upper)))
            relative_gap = (
                max(float(self.upper) - node_bound, 0.0) / gap_scale
            )
        else:
            relative_gap = 0.0

        feature_rows = []
        for variable_index in candidate_indices:
            relaxed_value = float(
                np.clip(z_solution[int(variable_index)], 0.0, 1.0).item()
            )
            fractionality = min(relaxed_value, 1.0 - relaxed_value)
            stage = int(variable_index) // self.nz
            mode = int(variable_index) % self.nz
            feature_rows.append([
                relaxed_value,
                fractionality,
                float(lower_bounds[int(variable_index)]),
                float(upper_bounds[int(variable_index)]),
                stage / max(self.N - 1, 1),
                mode / max(self.nz - 1, 1),
                node.depth / max(n_binary, 1),
                node_bound,
                relative_gap,
                unfixed / max(n_binary, 1),
                *x0_flat.tolist(),
            ])

        return {
            "candidate_indices": candidate_indices,
            "features": np.asarray(feature_rows, dtype=np.float32),
            "teacher_scores": scores.detach().cpu().numpy(),
            "best_var_combo": tuple(r_combos[best_combo_idx]),
            "best_combo_bounds": best_combo_bounds,
            "best_combo_costs": best_combo_costs,
            "best_combo_solutions": best_combo_solutions,
            "zk":zk,
            "best_combo_feasible": (
                feasible.reshape(n_combos, n_patterns)[best_combo_idx]
                .detach()
                .cpu()
                .numpy()
            ),
        }

    def make_branching_training_example(
        self,
        node_id,
        x0,
        teacher_scores,
        instance_id,
    ):
        """Package externally evaluated strong-branching scores for training."""
        candidate_indices, features = self.get_branching_features(node_id, x0)
        teacher_scores = np.asarray(teacher_scores, dtype=np.float32).reshape(-1)
        if teacher_scores.shape[0] != candidate_indices.shape[0]:
            raise ValueError(
                "teacher_scores must contain one score for every candidate "
                "returned by get_branching_features()"
            )
        return {
            "features": torch.as_tensor(features, dtype=torch.float32),
            "teacher_scores": torch.as_tensor(
                teacher_scores,
                dtype=torch.float32,
            ),
            "candidate_indices": torch.as_tensor(
                candidate_indices,
                dtype=torch.long,
            ),
            "instance_id": instance_id,
            "x0": torch.as_tensor(
                np.asarray(x0).reshape(-1),
                dtype=torch.float32,
            ),
            "node_depth": int(self.node_list[node_id].depth),
            "feature_names": self.get_branching_feature_names(),
        }

    def solve_branch_and_bound_nn(self, x0):
        """Branch-and-bound with NN branching and batched down/up solves."""
        if self.branch_policy is None:
            raise RuntimeError(
                "No branching policy is loaded. Call load_branching_policy() "
                "or set_branching_policy() before solve_branch_and_bound_nn()."
            )

        self.node_list = []
        self.upper = inf
        self.upper_var = 0
        self.nodes_visited = 0
        self.start_time = time.time()

        self.node_list.append(
            Node(
                np.zeros((self.N * self.nz, 1)),
                np.ones((self.N * self.nz, 1)),
            )
        )
        self.solve_local_prob(x0, 0, lisco=True)
        self.generate_upper_bound(x0, lisco=True)
        self.check_current_sol(len(self.node_list) - 1)
        child_batch_size = 2
        if self.current_batch_size != child_batch_size:
            self.lisco.nlp.setup_for_batch(
                child_batch_size,
                force_rebuild=False,
            )
            self.current_batch_size = child_batch_size

        device = torch.device(self.device)
        dtype = self.lisco.nlp.dtype
        n_binary = self.N * self.nz
        lb_offset = self.nx
        ub_offset = self.nx + n_binary
        primal_size = self.N * (self.nx + self.nu + self.nz)
        x_end = self.N * self.nx
        u_end = x_end + self.N * self.nu
        x0_tensor = torch.as_tensor(
            np.asarray(x0).reshape(-1),
            device=device,
            dtype=dtype,
        )

        while True:
            node_id = self.decide_next_node()
            if node_id is None:
                break

            self.nodes_visited += 1
            node = self.node_list[node_id]
            branch_variable = self.decide_next_branch_variable_nn(node_id, x0)
            branched_list = node.branched_list.copy()
            branched_list.append(branch_variable)

            binary_offset = self.nx * self.N + self.nu * self.N
            branch_value = float(
                node.var[binary_offset + branch_variable][0]
            )

            node_lb = torch.as_tensor(
                np.asarray(DM(node.lbz)).reshape(-1),
                device=device,
                dtype=dtype,
            )
            node_ub = torch.as_tensor(
                np.asarray(DM(node.ubz)).reshape(-1),
                device=device,
                dtype=dtype,
            )
            z0_base = node.zk.clone()
            z0 = z0_base.unsqueeze(0).expand(self.current_batch_size, -1).clone()
            base_p = torch.cat((x0_tensor, node_lb, node_ub))
            p_batch = base_p.unsqueeze(0).expand(
                child_batch_size,
                -1,
            ).clone()

            p_batch[0, ub_offset + branch_variable] = p_batch[
                0,
                lb_offset + branch_variable,
            ]
            p_batch[1, lb_offset + branch_variable] = p_batch[
                1,
                ub_offset + branch_variable,
            ]

            z_solved, _ = self.lisco.solve_batch(
                p_batch=p_batch,
                max_iter=self.max_iter,
                alpha=1.0,
                #predictor=self.predictor,
                z0_batch=z0,
                update_mode="full",#"safeguarded_armijo",#"full",
                return_trajectory=False,
            )
            residuals = self.lisco.nlp.Tk_batch_func(
                z_solved,
                p_batch,
            ).reshape(-1)
            feasible = residuals <= 1e-5


            u_batch = z_solved[:, x_end:u_end].reshape(
                child_batch_size,
                self.N,
                self.nu,
            )

            x_end = self.N * self.nx
            u_end = x_end + self.N * self.nu
            z_end = u_end + self.N * self.nz
            x_batch = z_solved[:, :x_end].reshape(child_batch_size, self.N, self.nx)[:, :, :2]  # z_solved[:num_problems, :x_end].reshape(
            # num_problems,
            # self.N,
            # self.nx,
            # )

            z_batch = z_solved[:, u_end:z_end].reshape(
                child_batch_size,
                self.N,
                self.nz,
            )
            c_k = torch.tensor(
                [0.1, 0.2],
                dtype=z_batch.dtype,
                device=z_batch.device,
            )
            x0_batch = p_batch[:, :self.nx - 2]
            #z_batch_old = z_solved[:child_batch_size, u_end:z_end - self.nz].reshape(
            #    child_batch_size,
            #    self.N-1,
            #    self.nz,
            #)
            #z_batch_old = torch.concatenate((torch.zeros(child_batch_size, 1, self.nz,device=self.device), z_batch_old),axis=1)
            costs = (
                    1 * x_batch.square().sum(dim=(1, 2))  # 0.2
                    + 0.1 * u_batch.square().sum(dim=(1, 2))  # 0.1
                    + 1 * x0_batch.square().sum(dim=1)  # 0.2
                    + (z_batch * c_k).sum(dim=(1, 2))
                    #+ 0.1 * (z_batch - z_batch_old).square().sum(dim=(1, 2))
            )
            #costs = (
            #        0.8 * (x_batch).square().sum(dim=(1, 2))  # 0.2
            #        + 0.1 * u_batch.square().sum(dim=(1, 2))  # 0.1
            #        + 0.8 * (x0_batch).square().sum(dim=1)  # 0.2
            #        + (z_batch * 0.0).sum(dim=(1, 2))
            #)
            costs = costs.masked_fill(~feasible, torch.inf)

            child_lb_arrays = (
                p_batch[:, lb_offset:ub_offset]
                .detach()
                .cpu()
                .numpy()
            )
            child_ub_arrays = (
                p_batch[:, ub_offset:ub_offset + n_binary]
                .detach()
                .cpu()
                .numpy()
            )
            child_costs = costs.detach().cpu().numpy()
            child_solutions = (
                z_solved[:, :primal_size]
                .detach()
                .cpu()
                .numpy()
            )

            for child_index in range(child_batch_size):
                self.node_list.append(
                    Node(
                        DM(child_lb_arrays[child_index].reshape(-1, 1)),
                        DM(child_ub_arrays[child_index].reshape(-1, 1)),
                        depth=node.depth + 1,
                        parent_id=node_id,
                        branch_var=branch_variable,
                        branch_value=branch_value,
                        branched_list=branched_list,
                    )
                )
                child = self.node_list[-1]
                child.var = child_solutions[child_index].reshape(-1, 1)
                child.lb = float(child_costs[child_index])
                child.zk=z_solved[child_index]
                child.Tk=residuals[child_index]
                self.check_current_sol(len(self.node_list) - 1)

            node.branched = True
            self.prune_nodes()

        solve_time = time.time() - self.start_time
        #tk_values = self.get_node_tk_values()
        return self.upper_var, self.upper, self.nodes_visited, solve_time,self.upper_tk#tk_values

    def solve_branch_and_bound(self,x0):

        self.start_time = time.time()
        self.node_list.append(Node(np.zeros((self.N*self.nz,1)),np.ones((self.N*self.nz,1))))#self.lbg[(self.nx*2+self.nu*3+1)*self.N+self.nx:],self.ubg[(self.nx*2+self.nu*3+1)*self.N+self.nx:]))
        id=0
        self.solve_local_prob(x0,id)
        self.generate_upper_bound(x0)
        #root_z = self.node_list[0].var[self.nx*(self.N+1)+self.nu*self.N:]
        #self.feasibility_pump(x0, root_z, max_iterations=20)
        while True:
            id = self.decide_next_node()

            if id is None:
                break

            self.nodes_visited += 1
            node = self.node_list[id]
            branched_list=node.branched_list.copy()
            ind_z, ubz_lower_branch, lbz_upper_branch, lb_down, lb_up, sol_down, sol_up = self.solve_branch_strong(
                id, x0, num_candidates=40)
            branched_list.append(ind_z)
            branch_value = node.var[ind_z + self.nx * (self.N) + self.nu * self.N][0]
            lbz_lower_branch = DM(node.lbz)
            # ubz_lower_branch = DM(node.ubz)
            # ubz_lower_branch[ind_z] = max(floor(branch_value), lbz_lower_branch[ind_z])
            self.node_list.append(
                Node(lbz_lower_branch, ubz_lower_branch, depth=node.depth + 1, parent_id=id, branch_var=ind_z,
                     branch_value=branch_value,branched_list=branched_list))
            self.node_list[-1].var = sol_down
            self.node_list[-1].lb = lb_down
            # self.solve_local_prob(x0, len(self.node_list) - 1)
            self.check_current_sol(id=len(self.node_list) - 1)
            # lbz_upper_branch = DM(node.lbz)
            ubz_upper_branch = DM(node.ubz)
            # lbz_upper_branch[ind_z] = min(ceil(branch_value), ubz_upper_branch[ind_z])
            self.node_list.append(
                Node(lbz_upper_branch, ubz_upper_branch, depth=node.depth + 1, parent_id=id, branch_var=ind_z,
                     branch_value=branch_value,branched_list=branched_list))
            # self.solve_local_prob(x0, len(self.node_list) - 1)
            self.node_list[-1].var = sol_up
            self.node_list[-1].lb = lb_up
            self.check_current_sol(id=len(self.node_list) - 1)
            self.node_list[id].branched = True
            self.prune_nodes()
        
        solve_time = time.time() - self.start_time
        return self.upper_var, self.upper, self.nodes_visited, solve_time


def main():
    opt=Four_Tank()#Linearized_Four_Tank()#Four_Tank()#Linearized_Four_Tank()#Nonlinear_double()#Lokta_Volterra()#Linearized_Four_Tank()#Optimization_Problem()
    b_b=branch_and_bound_solver(opt.lbx,opt.ubx,opt.lbg,opt.ubg,opt.lbz,opt.ubz,opt.prob,opt.discrete,opt.model_gurobi,opt.model_gurobi_vars,opt.nx,opt.nu,opt.nz,opt.N,r=1)
    x0 = np.array([[0.5],[0.5],[0.5],[0.5]])#np.array([-0.6,-0.6]).reshape(2,1)#[-2,2,1,-1]
    #x1=time.time()
    #J_gurobi,x=b_b.solve_gurobi(x0)
    #x2=time.time()
    #print('Gurobi Cost:',J_gurobi)
    #print('Gurobi Time:', x2-x1)
    x1 = time.time()
    J_bonmin,x=b_b.solve_bonmin(x0)#solve_gurobi(x0)#solve_bonmin(x0)
    x2 = time.time()
    print(x[0:80].squeeze().tolist())
    print(x[80:120].squeeze().tolist())
    print(x[120:160].squeeze().tolist())
    print('Bonmin cost:', J_bonmin)
    print('Bonmin time:',x2-x1)
    b_b.get_cost_from_z(torch.tensor(x).reshape(1, -1), torch.tensor(x0).reshape(1, -1), 1)
    #x,J,nodes_visited,solve_time=b_b.solve_branch_and_bound(x0)
    #print(f"Nodes visited: {nodes_visited}")
    #print(f"Solve time: {solve_time:.4f} seconds")
    #b_b.write_tree_summary('tree_summary.txt')
    #print("Tree summary written to tree_summary.txt")
    #liste = x[120:160].squeeze().tolist()
    #print([int(liste[k]) for k in range(len(liste))])
    #print('My cost (solve_lisco):', J)
    
    #print("\n" + "="*80)
    #print("Testing solve_lisco_parallel with r=2")
    #print("="*80 + "\n")

    max_iter=40
    b_b = branch_and_bound_solver(opt.lbx, opt.ubx, opt.lbg, opt.ubg, opt.lbz, opt.ubz, opt.prob, opt.discrete,
                               opt.model_gurobi, opt.model_gurobi_vars, opt.nx, opt.nu, opt.nz, opt.N, r=1,num_candidates=40,device="cuda",max_iter=max_iter)
    #b_b.load_branching_policy("branching/branching/results/branching_policy_new.pt")

    x_parallel, J_parallel, nodes_visited_parallel, solve_time_parallel = b_b.solve_lisco_parallel(x0, r=1)#solve_branch_and_bound_nn(x0)#solve_lisco_parallel(x0, r=1)#solve_branch_and_bound_nn(x0)#solve_lisco_parallel(x0, r=1)##solve_branch_and_bound_nn(x0)#solve_lisco_parallel(x0, r=1)
    print(f"Nodes lvisited (parallel): {nodes_visited_parallel}")
    print(f"Solve time (parallel): {solve_time_parallel:.4f} seconds")
    print(x_parallel[0:80].squeeze().tolist())
    print(x_parallel[80:120].squeeze().tolist())
    liste=x_parallel[120:160].squeeze().tolist()
    print([round(liste[k]) for k in range(len(liste))])
    b_b.write_tree_summary(f'tree_summary_parallel_{max_iter}.txt')
    print('My cost (solve_lisco_parallel):', J_parallel)

if __name__ == '__main__':
   main()


