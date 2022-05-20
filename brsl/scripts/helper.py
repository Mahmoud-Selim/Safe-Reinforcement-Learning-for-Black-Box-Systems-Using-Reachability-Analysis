from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import linprog
import numpy as np
import numpy as np
from reachability.Zonotope import Zonotope
import math
import matplotlib.pyplot as plt
import time

def check_zono_intersection(_input):
    #print("Hi")
    zon1, obj = _input
    #for obj in objs:
    c_1 = zon1.center()
    G_1 = zon1.generators()

    c_2 = obj.center()
    G_2 = obj.generators()

    c, G, A, b = zon1.intersect_zonotopes(obj)

    #c, G, A, b = zon1.intersect_zonotopes(obj)

    LP_f,LP_A_ineq,LP_b_ineq,LP_A_eq,LP_b_eq = zon1.make_con_zono_empty_check_LP(A, b)
    #print("{} \n {} \n {} \n {} \n {}".format(LP_f, LP_A_ineq,LP_b_ineq,LP_A_eq,LP_b_eq))
    #print(LP_f.shape, LP_A_ineq.shape, LP_b_ineq.shape, LP_A_eq.shape, LP_b_eq.shape)
    """
    def eq_con(x):
        global LP_A_eq
        global LP_b_eq 
        return A * x - b

    def ineq_con(x):
        global LP_A_ineq
        global LP_b_ineq

        return LP_A_ineq * x - LP_b_ineq 
    """
    def cost(x, *args):
        A = args[0] 
        result = np.dot(x.reshape((1, len(x))), A)[0]
        return result


    eq_con = LinearConstraint(LP_A_eq, LP_b_eq.flatten(), LP_b_eq.flatten())
    ineq_con = LinearConstraint(LP_A_ineq, np.array([np.NINF] * 8), LP_b_ineq.flatten())
    #cons = [{'type':'eq', 'fun': eq_con},
    #        {'type':'ineq', 'fun': ineq_con}]
    cons = [eq_con, ineq_con]
    #print(cons)
    #print(LP_f.shape, LP_A_ineq.shape, LP_b_ineq.shape, LP_A_eq.shape, LP_b_eq.shape)
    #print(LP_A_eq)
    #start_t = time.time()
    
    res = linprog(LP_f, A_ub = LP_A_ineq, b_ub=LP_b_ineq.flatten(), \
                  A_eq=LP_A_eq, b_eq=LP_b_eq.flatten(), bounds=(None,None), \
                  options = {'maxiter':50}, method = 'highs-ipm')
    #print("res", res)
    #return res
    #print("After min", time.time() - start_t)
    #print("z_opt", res["x"])
    z_opt = res["x"]
    lm = res["ineqlin"]["marginals"]
    nu = res["eqlin"]["marginals"]
    #print(lagrange_multipliers)
    if(z_opt[-1] > 1):
        #print("returning")
        return False, 0
    else:
        M_Q = np.zeros((LP_A_ineq.shape[1], LP_A_ineq.shape[1]))
        M_GT = LP_A_ineq.T
        M_AT = LP_A_eq.T 
        M_DlmG = np.dot(np.diag(lm), LP_A_ineq) 
        M_DGzh = np.diag((np.dot(LP_A_ineq, z_opt) - LP_b_ineq.flatten()))
        M_A = LP_A_eq
        #print(np.dot(LP_A_ineq, z_opt))
        row_1 = np.hstack((M_Q, M_GT, M_AT))

        #print("row 1 \n", row_1, "*"*80)
        #print(M_DlmG.shape, M_DGzh.shape, np.zeros((M_DGzh.shape[0], M_AT.shape[1])).shape)
        row_2 = np.hstack((M_DlmG, M_DGzh, np.zeros((M_DGzh.shape[0], M_AT.shape[1]))))
        #print("row 2 \n", row_2, "*"*80)
        row_3 = np.hstack((M_A, np.zeros((M_A.shape[0], M_DGzh.shape[1] + M_AT.shape[1]))))
        LHS = np.vstack((row_1,
                            row_2,
                            row_3))
        #print(LHS)
        db = np.eye(LP_b_eq.shape[0])

        RHS = np.vstack((np.zeros((LHS.shape[0] - db.shape[0], db.shape[1])), db))

        J = np.dot(np.linalg.pinv(LHS), RHS) 
        #print(J)
        dz_opt_d_c_2 = J[: len(z_opt), :]
        #if(len(args) != 0):
        #    #(args)
        #    dz_opt_d_c_2 = np.dot(dz_opt_d_c_2, args[0])
        #print("dz/dc2", dz_opt_d_c_2)
        con = 1 - z_opt[-1] * z_opt[-1]

        d_con = -2 * z_opt[-1] * dz_opt_d_c_2[-1, :]
        #print("con", con, np.linalg.pinv(d_con.reshape((1, -1))))
        delta_center = np.linalg.pinv(d_con.reshape((1, -1))) * con
        #print(delta_center)
        #print("Returning")
        return True, delta_center
    
