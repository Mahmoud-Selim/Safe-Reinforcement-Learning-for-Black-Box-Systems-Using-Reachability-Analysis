from Zonotope import Zonotope
from MatZonotope import MatZonotope
import numpy as np 
from read_matlab import read_matlab
from reachability_analysis import concat_traj, get_AB
import joblib

dim_x = 2
A = np.array([[-1, -4, 0, 0, 0], [4, -1, 0, 0, 0], [0, 0, -3, 1, 0], [0, 0, -1, -3, 0], [0, 0, 0, 0, -2]])
B_ss = np.ones([5, 1])
C = np.array([1,0,0,0,0])
D = 0

initpoints =100
steps = 1
totalsamples = initpoints*steps
X0 = Zonotope(np.array(np.ones((dim_x, 1))), 0.1 * np.diag(np.ones((dim_x, 1)).T[0]))
U = Zonotope(np.array(np.ones((dim_x, 1))),0.25 * np.diag(np.ones((dim_x, 1)).T[0]))
W = Zonotope(np.array(np.zeros((dim_x, 1))), 0.003 * np.ones((dim_x, 1)))
print(X0)
print(U)
print(W)


"""index=1;
for i=1:size(W.generators,2)
    vec=W.Z(:,i+1);
    GW{index}= [ vec,zeros(dim_x,totalsamples-1)];
    for j=1:totalsamples-1
        GW{j+index}= [GW{index+j-1}(:,2:end) GW{index+j-1}(:,1)];
    end
    index = j+index+1;
end
"""

GW = []
for i in range(W.generators().shape[1]):
    vec = np.reshape(W.Z[:, i + 1], (dim_x, 1))
    dummy = []
    dummy.append(np.hstack((vec, np.zeros((dim_x, totalsamples - 1)))))
    print(vec.shape, W.Z.shape, dummy[i][:, 2:].shape, dummy[0][:, 0].shape)
    for j in range(1, totalsamples, 1):
        right = np.reshape(dummy[i][:, 0:j], (dim_x, -1))
        left = dummy[i][:, j:]
        dummy.append(np.hstack((left, right)))
    GW.append(np.array(dummy))

GW = np.array(GW)
#print(np.array(GW).shape)

Wmatzono = MatZonotope(np.zeros((dim_x, totalsamples)), GW)

#print(Wmatzono)
u = read_matlab('D:\\KTH\\u.mat', 'u')
x0 = X0.center()

print(x0)

utraj = read_matlab('D:\\KTH\\utraj.mat', 'utraj')
x = read_matlab('D:\\KTH\\x.mat', 'x')
#print(utraj.shape)
sysd_A = read_matlab('D:\\KTH\\A.mat', 'A_mat')
sysd_B = read_matlab('D:\\KTH\\B.mat', 'B_mat')

index_0 =1
index_1 =1
#print(W.rand_point())
U_full, X_0t, X_1t = concat_traj(X0, x, utraj, dim_x, steps, initpoints)
print("UT_full")
print(U_full)
print()
AB = get_AB(U_full, X_0t, X_1t, Wmatzono)
intAB11 = AB.interval_matrix()
intAB1 = intAB11.int
print(intAB1)

total_steps = 5

X_model = [X0]
X_data = [X0]
#print("A", sysd_A)
#print(sysd_B)
print("Doing Steps")

for i in range(total_steps):
    print("Doing Step", i)
    X_model[i] = X_model[i].reduce('girard', 10)
    #print(X_model[i])
    #print(U * sysd_B)
    X_model.append(X_model[i] * sysd_A + U * sysd_B + W ) #+ sysd_B * U + W


    X_data[i] = X_data[i].reduce('girard', 1)
    print(X_data[i])#.Z.shape
    X_data.append(AB * X_data[i].cart_prod(U) + W)
    
joblib.dump(AB, 'AB.ra')
