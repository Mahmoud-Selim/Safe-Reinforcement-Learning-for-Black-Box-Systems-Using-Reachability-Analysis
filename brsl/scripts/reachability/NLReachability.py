from .Zonotope import Zonotope
from .MatZonotope import MatZonotope
import numpy as np 
from .read_matlab import read_matlab
#from .reachability_analysis import concat_traj, get_AB
import joblib
from .utils.Options import Options
from .utils.Params import Params
from .reachability_nonlinear import *
import time
import sys
import os 

class NLReachability:
    def __init__(self, path=""):
        self.dim_x = 2
        self.dim_u = 8
        self.dim_pose = self.dim_u
        self.dim_a = 2

        self.dt = 0.015
        self.params = Params(tFinal = self.dt * 5, dt = self.dt)
        self.options = Options()
        #print("dude", sys.path)

        self.options.params["dim_x"] = self.dim_x
        self.options.params["dim_u"] = self.dim_u
        #Number of trajectories
        self.initpoints = 1000
        #Number of time steps
        self.steps = 10

        # Totoal number of samples
        self.totalsamples = 500#steps * initpoints

        #noise zonotope
        self.wfac = 0.001

        self.W = Zonotope(np.array(np.zeros((self.options.params["dim_x"], 1))), self.wfac * np.ones((self.options.params["dim_x"], 1)))
        self.params.params["W"] = self.W
        self.GW = []
        for i in range(self.W.generators().shape[1]):
            vec = np.reshape(self.W.Z[:, i + 1], (self.dim_x, 1))
            dummy = []
            dummy.append(np.hstack((vec, np.zeros((self.dim_x, self.totalsamples - 1)))))
            #print(vec.shape, W.Z.shape, dummy[i][:, 2:].shape, dummy[0][:, 0].shape)
            for j in range(1, self.totalsamples, 1):
                right = np.reshape(dummy[i][:, 0:j], (self.dim_x, -1))
                left = dummy[i][:, j:]
                dummy.append(np.hstack((left, right)))
            self.GW.append(np.array(dummy))

        self.GW = np.array(self.GW[0])
        #print("->"*10, np.array(GW).shape)

        self.params.params["Wmatzono"] = MatZonotope(np.zeros((self.dim_x, self.totalsamples)), self.GW)

        self.options.params["zonotopeOrder"] = 100
        self.options.params["tensorOrder"] = 2
        self.options.params["errorOrder"] = 5

        self.u = np.load(os.path.join(path, 'U_azure_500.npy'), allow_pickle=True).T 
        self.x_meas_vec_0 = np.load(os.path.join(path, 'X0_azure_500.npy'), allow_pickle=True).T
        self.x_meas_vec_1 = np.load(os.path.join(path, 'X1_azure_500.npy'), allow_pickle=True).T
        #print(u.shape, x_meas_vec_0.shape, "xasd")
        self.X_0T = self.x_meas_vec_0
        self.X_1T = self.x_meas_vec_1
        self.options.params["U_full"] = self.u

        L = 0
        for i in range(self.totalsamples):
            z1 = np.hstack((self.x_meas_vec_0[:, i].flatten(), self.u.flatten(order='F')[i]))
            f1 = self.x_meas_vec_1[:, i]
            #print("dude", z1)
            for j in range(self.totalsamples):
                z2 = np.hstack((self.x_meas_vec_0[:, j].flatten(), self.u.flatten(order='F')[j]))
                f2 = self.x_meas_vec_1[:, j]
                #print("TG", np.linalg.norm(z1 - z2))
                new_norm = np.linalg.norm(f1 - f2) / np.linalg.norm(z1 - z2)

                if (new_norm > L):
                    L = new_norm
                    eps = L * np.linalg.norm(z1 - z2)
        
        self.options.params["Zeps"] = Zonotope(np.array(np.zeros((self.dim_x, 1))),eps * np.diag(np.ones((self.options.params["dim_x"], 1)).T[0]))
        #Z_eps_gen = np.load("ZepsZ.npy")#np.save("ZepsZ", eps * np.diag(np.ones((options.params["dim_x"], 1)).T[0]))
        self.options.params["ZepsFlag"] = True
        self.options.params["Zeps_w"] = self.params.params["W"] + self.options.params["Zeps"]
        #self.options.params["Zeps"] = Zonotope(np.array(np.zeros((self.dim_x, 1))), self.Z_eps_gen)
        #print(options.params["Zeps"])
        #print(z1)
        #print(f1)

        
        self.options.params["X_0T"] = self.x_meas_vec_0
        self.options.params["X_1T"] = self.x_meas_vec_1 

    def run_reachability(self, r = [0, 0, 0, 0], u = [0, 0]):
        #U = Zonotope(np.array(np.array(u).reshape((self.dim_u, 1))), np.diag([.001] * self.dim_u))

        R0 = Zonotope(np.array(r).reshape((self.dim_x, 1)), np.diag([.001] * self.dim_x))

        #self.params.params["U"] = U
        self.params.params["R0"] = R0
        
        #start_t = time.time()
        R_data, derivatives = self.reach_DT(self.params, self.options, u)
        #print("predicted pose", R_data[-1].reduce('girard', 1).Z[:2, 0])
        #print("Operation took {}".format(time.time() - start_t))
        return R_data, derivatives


    def reach_DT(self, params, options, u, *varargin):
        options = self.params2options(params,options)
        #options = self.checkOptionsReach(options,0)

        spec = []
        if(len(varargin) > 0):
            spec = varargin[0]

        steps = len(u)#len(t) - 1

        R_data = [params.params["R0"]]
        #R_data[0] = R_data[0].reduce('girard', 1)
        derivatives = []
        for i in range(steps):
            if('uTransVec' in options.params):
                options.params['uTrans'] = options.params["uTransVec"][:, i]


            start_t = time.time()
            U = Zonotope(np.array(np.array(u[i]).reshape((self.dim_u, 1))), np.diag([.001] * self.dim_u))
            self.params.params["U"] = U

            
            uTrans = U.center()
            #print(options.params["U"])
            self.options.params["U"] = U  - uTrans
            #print(options.params["U"])
            self.options.params["uTrans"] = uTrans

            new_state, dc_dr, dc_du = self.linReach_DT(R_data[i] ,options)
            #print("Took {}".format(time.time() - start_t))
            R_data.append(new_state.reduce('girard', 1))
            derivatives.append([dc_dr, dc_du])
            #print("early info", R_data)
            #R_data.append(dc_du)
        return R_data, derivatives

    def params2options(self, params, options):
        for key, value in params.params.items():
            options.params[key] = value
        return options 


    def set_inputSet(self, options):
        """
        This function is greatly modified and different from the one in CORA. 
        Mostly removed parts that are irrelevant to our reachability analysis techniques, that is the check for "checkOptionsSimulate"
        """
        uTrans = options.params["U"].center()
        #print(options.params["U"])
        options.params["U"] = options.params["U"]  - uTrans
        #print(options.params["U"])
        options.params["uTrans"] = uTrans

        return options


    def checkOptionsReach(self, options, hyb):
        #checkName = 'checkOptionsReach'
        #options.params["tStart"] = 0
        #options.params["reductionTechnique"] = 'girard'
        #options.params["verbose"] = 0
        #options = self.set_inputSet(options)
        return options


    def linReach_DT(self, R_data ,options):
        """
        This function calculates teh next state for the reachability analysis of a non linear system.
        """
        #print("*" * 80)
        
    # print(options.params["U"] , "\n", options.params["uTrans"])
        options.params["Uorig"] = options.params["U"] + options.params["uTrans"]
        xStar = R_data.center()
        uStar = options.params["Uorig"].center()

        xStarMat = matlib.repmat(xStar, 1, options.params["X_0T"].shape[1])
        uStarMat  = matlib.repmat(uStar, 1, options.params["U_full"].shape[1])
        oneMat    = matlib.repmat(np.array([1]), 1, options.params["U_full"].shape[1])
        num_mat = np.vstack([oneMat, options.params["X_0T"] + (-1 * xStarMat), options.params["U_full"] + -1 * uStarMat])
        #options.params["X_0T"] + (-1 * xStarMat), options.params["U_full"] + -1 * uStarMat
        start_t = time.time()
        IAB = np.dot(options.params["X_1T"], pinv(num_mat))
        #print(num_mat.shape)
        start_t2 = time.time()
        #dummy = options.params["Wmatzono"] + np.dot(IAB, np.vstack([oneMat, options.params["X_0T"]+(-1*xStarMat), options.params["U_full"]+ -1 * uStarMat]))
        #print("GEEZ \n", np.dot(IAB, np.vstack([oneMat, options.params["X_0T"]+(-1*xStarMat), options.params["U_full"]+ -1 * uStarMat])).shape)
        V =  -1 * (options.params["Wmatzono"] + np.dot(IAB, num_mat)) + options.params["X_1T"]
        #print("IAB is {}".format(IAB.shape)) Current shape is 2*11
        #print("V is: \n", V)
        start_t3 = time.time()
        VInt = V.interval_matrix()
        leftLimit = VInt.Inf
        rightLimit = VInt.Sup
        
        #print(leftLimit.shape)
        V_one = Zonotope(Interval(leftLimit.min(axis=1).T, rightLimit.max(axis=1).T))

        #print("Vone is {}".format(V_one))
        x = R_data+(-1*xStar)
        #print("Move center:", np.dot(IAB[:, -2:], uStar), IAB[:, 3:])
        result = (x.cart_prod(options.params["Uorig"] + (-1 * uStar)).cart_prod([1]) * IAB) +  V_one + options.params["Zeps_w"]
        #print("wow {}".format(x.cart_prod(options.params["Uorig"] + (-1 * uStar)).cart_prod([1])))
        #print("derivative is {}".format(IAB[:2, -2:]))
        #print("done {} {} {}".format(time.time() - start_t, time.time() - start_t2, time.time() - start_t3))xx3
        return result, IAB[:self.dim_x, 1: 1 + self.dim_x], IAB[:self.dim_x, 1 + self.dim_x: 1 + self.dim_x + self.dim_a]