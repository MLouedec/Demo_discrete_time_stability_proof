# Numerical method to prove the exponential stability of a nonlinear discrete system

from lib import *

# ---------------------------------------------------------------------------------------------
# Definition of the discrete system
# x(k+1) = A*x(k) + B*tanh(x(k)) = h(x(k))
#
# Jacobian matrix # J(x) = A + B*diag(1-tanh^2(x)), J(0) = A + B
# ---------------------------------------------------------------------------------------------
paper_system = True  # if True, use the system presented in the paper
if paper_system:
    N = 2  # dimension of the state space
    A = np.array([[-0.8,2],[-1,1.6]])
    B = np.array([[0.1,0.01],[0.01,0.1]])
    J = A + B  # Jacobian matrix of h at the origin
    print("Stability of the system presented in the paper")
    print("Jacobian matrix J=\n", J)
else:
    # random definition of the system
    N = 5  # dimension of the state space
    system_defined = False
    while not system_defined:
        # select A as a random Schur matrix
        A = 2 * (np.random.rand(N, N) - 0.5 * np.ones((N, N)))
        A = 0.9 * A / np.max(np.abs(np.linalg.eigvals(A)))

        # select B as a small random matrix
        B = 0.01 * (np.random.rand(N, N))

        J = A + B  # Jacobian matrix of h at the origin
        if np.all(np.abs(np.linalg.eigvals(J)) < 1):  # the system must have J Schur
            system_defined = True
sqrtm_op = SQRTM_OPERATOR(N)
epsilon = 10e-3
print("the system has been defined")

# ---------------------------------------------------------------------------------------------
# Numerical method to prove the exponential stability of the system
# ---------------------------------------------------------------------------------------------

# step 1 - solve the Lyapunov equation
P = scipy.linalg.solve_discrete_lyapunov(J.T, J.T @ J)

# step 2 - test the inclusion Ph(E) < E
alpha = 1
alpha_max = 10
res = False
while alpha < alpha_max:
    Q = cdc.IntervalMatrix(10 ** alpha * np.array(P)) # Q = Gamma**(-2)
    Q_inv = matrix_inversion(Q)
    G = sqrtm_op.sqrtm(Q_inv)
    E_box = E_box_(G)
    J_box = J_box_(A,B,E_box)
    Q_out, rho = propagate_ellipsoid(Q,cdc.IntervalMatrix(J),J_box)
    Delta_Q = Q_out - Q
    res, _ = test_positive_definite_by_cholesky(Delta_Q)
    print("for alpha =", alpha, " res is ", res, " with rho =", rho)
    if res:
        break
    alpha += 1

if res:
    print("the system is exponentially stable")
else:
    print("the method is not able to conclude on the stability of the system")

# ---------------------------------------------------------------------------------------------
# draw results
# ---------------------------------------------------------------------------------------------
scale = 1.1* max(E_box[0].ub(), E_box[1].ub())
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.set_title("Propagation of the ellipsoid (projection on the first 2 dimensions)")
ax2.set_ylabel("x2")
ax2.set_xlabel("x1")
ax2.set_aspect('equal')
ax2.set_xlim(-scale, scale)
ax2.set_ylim(-scale, scale)
ax2.grid()

Q_mid = mid_matrix(Q)
Q_out_mid = mid_matrix(Q_out)
draw_box_projected(ax2, E_box, col='grey', label="$[\mathscr{E}]$")
draw_ellipse_projected(ax2, Q_mid, col='red', in_col='#ff000080',label="$\mathscr{E}$")
draw_ellipse_projected(ax2, Q_out_mid, col='green',in_col='#00ff0080',label="$\mathscr{E}_{out}$")
ax2.legend()
plt.show()
print("end")
