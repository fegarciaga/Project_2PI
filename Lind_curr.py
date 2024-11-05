import numpy as np
from qutip import *

def J_w(w, W0 =1, Lambda =1, T=0.1):
    if np.abs(w)>1e-7:
        return 1/W0*w*np.exp(-np.abs(w)/Lambda)/(np.exp(w/T)-1)
    else:
        return T/W0

def Build_H(N, hx, J, sx_list, sy_list, sz_list):
    H = 0
    for n in range(N):
        H += -hx * sz_list[n]
    for n in range(N-1):
        H += J * sx_list[n] * sx_list[n+1]
        H += J * sy_list[n] * sy_list[n+1]
        H += J * sz_list[n] * sz_list[n+1]
    return H

def Build_spinops(S):
    D = int(2*S+1)
    Sx = np.zeros((D,D))
    Sy = np.zeros((D,D), dtype=complex)
    Sz = np.zeros((D,D))

    for i in range(D):
        Sz[i,i] = S-i

    for i in range(D-1):
        m = -S+i
        Sx[i,i+1]=0.5*np.sqrt(S*(S+1)-m*(m+1))
        Sx[i+1,i]=0.5*np.sqrt(S*(S+1)-m*(m+1))
        Sy[i,i+1]=-0.5*1j*np.sqrt(S*(S+1)-m*(m+1))
        Sy[i+1,i]= 0.5*1j*np.sqrt(S*(S+1)-m*(m+1))

    return Qobj(Sx), Qobj(Sy), Qobj(Sz)

def Build_manybody(N, si, sx, sy, sz):
    sx_list = []
    sy_list = []
    sz_list = []

    for n in range(N):
        op_list = []
        for m in range(N):
            op_list.append(si)

        op_list[n] = sx
        sx_list.append(tensor(op_list))

        op_list[n] = sy
        sy_list.append(tensor(op_list))

        op_list[n] = sz
        sz_list.append(tensor(op_list))


    return sx_list, sy_list, sz_list

def Build_currents(N, sx_list, sy_list):
    Js_list = []
    for i in range(N-1):
        Js_list.append(2*(sx_list[i]*sy_list[i+1]-sy_list[i]*sx_list[i+1]))
    return Js_list

def Build_XXZ_currents(N, sx_list, sy_list, sz_list):
    JXXZ = []
    for i in range(1,N-1):
        JXXZ.append(2*((sy_list[i-1]*sz_list[i]*sx_list[i+1]-sx_list[i-1]*sz_list[i]*sy_list[i+1])+(sz_list[i-1]*sx_list[i]*sy_list[i+1]-sy_list[i-1]*sx_list[i]*sz_list[i+1])+(sx_list[i-1]*sy_list[i]*sz_list[i+1]-sz_list[i-1]*sy_list[i]*sx_list[i+1])))
    return JXXZ

def Build_Bcurrents(N, sx_list, sy_list, hx):
    JB = []
    for i in range(1, N-1):
        JB.append(hx*((sx_list[i-1]*sy_list[i]-sy_list[i-1]*sx_list[i])+(sx_list[i]*sy_list[i+1]-sy_list[i]*sx_list[i+1])))
    return JB

def Construct_collapse(Ekets, Evals, S, T, Lambda, bathS, W0):
    L = 0
    N = len(Evals)
    for i in range(N):
        for j in range(N):
            Xmn = Ekets[i].dag()*S*Ekets[j]
            L += np.sqrt(2*np.pi*bathS*J_w(Evals[j]-Evals[i], T=T, Lambda= Lambda, W0=W0))*Xmn*(Ekets[i]*Ekets[j].dag())
    return L

def Integrate(N, hx, J, bath_params, tlist, T, spin):
    bathS, Lambda, W0 = bath_params
    T1, T2 = T

    print(bathS)

    D_spin = int(2*spin+1)
    si = qeye(D_spin)
    sx, sy, sz = Build_spinops(spin)

    sx_list, sy_list, sz_list = Build_manybody(N, si, sx, sy, sz)

    H = Build_H(N, hx, J, sx_list, sy_list, sz_list)

    # Current operators
    Js = Build_currents(N, sx_list, sy_list)
    JXXZ = Build_XXZ_currents(N, sx_list, sy_list, sz_list)
    JB = Build_Bcurrents(N, sx_list, sy_list, hx)
    Jz = Build_currents(N, sz_list, sy_list)
    
    Evals, Ekets = H.eigenstates()

    c_list = []

    c_list.append(Construct_collapse(Ekets, Evals, sx_list[0], T1, Lambda, bathS, W0))
    c_list.append(Construct_collapse(Ekets, Evals, sy_list[0], T1, Lambda, bathS, W0))
    c_list.append(Construct_collapse(Ekets, Evals, sz_list[0], T1, Lambda, bathS, W0))

    c_list.append(Construct_collapse(Ekets, Evals, sx_list[N-1], T2, Lambda, bathS, W0))
    c_list.append(Construct_collapse(Ekets, Evals, sy_list[N-1], T2, Lambda, bathS, W0))
    c_list.append(Construct_collapse(Ekets, Evals, sz_list[N-1], T2, Lambda, bathS, W0))

    psi_list = []
    for i in range(N):
        if i%2==0:
            theta = np.pi/2
            phi = np.pi
        else:
            theta = np.pi/2
            phi = 0
        psi_list.append(Build_single(theta, phi, sx, sy, sz))

    psi_init = tensor(psi_list)

    sz_list.extend(sy_list)
    sz_list.extend(sx_list)
    sz_list.extend(Js)
    sz_list.extend(JXXZ)
    sz_list.extend(JB)
    sz_list.extend(Jz)

    options = Options(store_states = False)
    result = mesolve(H, psi_init, tlist, c_list, sz_list, options=options)

    return result.expect

def Build_single(theta, phi, sx, sy, sz):
    Sop = np.cos(theta)*sz+np.sin(theta)*np.cos(phi)*sx+np.sin(theta)*np.sin(phi)*sy
    evals, ekets = Sop.eigenstates()
    return ekets[0]

N = 4
h = 1
J = 1
bath_params = [0.1, 3, 1]
T = [5, 0.1]
spin = 0.5
al = np.linspace(0.01,1,100)

tlist = np.linspace(0,100,2000)
for i in range(100):
    bath_params[0]= al[i]
    data = Integrate(N, h, J, bath_params, tlist, T, spin)
    np.savetxt(f"LindbladResults/D_N_{N}_h_{h}_J_{J}_al_{bath_params[0]:.2f}_Wc_{bath_params[1]}.txt", data)
