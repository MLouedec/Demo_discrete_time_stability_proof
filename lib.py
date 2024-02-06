# toolbox for the main file

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy
# from module.roblib_morgan import sqrtm
import codac as cdc


# ---------------------------------------------------------------------------------------------
# numerical methods
# ---------------------------------------------------------------------------------------------

def transpose(A: cdc.IntervalMatrix):
    n, m = A.shape()
    res = cdc.IntervalMatrix(m, n)
    for i in range(n):
        for j in range(m):
            res[j][i] = A[i][j]
    return res


def J_box_(A, B, E_box):
    N = A.shape[0]
    A_box = cdc.IntervalMatrix(A)
    B_box = cdc.IntervalMatrix(B)
    T_box = cdc.IntervalMatrix(N, N)  # [tanh](E_box)
    for i in range(N):
        for j in range(N):
            if i == j:
                T_box[i][j] = cdc.Interval(1, 1) - cdc.sqr(cdc.tanh(E_box[i]))
            else:
                T_box[i][j] = cdc.Interval(0, 0)

    J_box = A_box + B_box * T_box
    return J_box


def E_box_(G):
    if type(G) is np.ndarray:
        N = G.shape[0]
        E_box = cdc.IntervalVector(N, cdc.Interval(-1, 1))
        for i in range(N):
            E_box[i] = np.linalg.norm(G[i]) * E_box[i]

    if type(G) is cdc.IntervalMatrix:
        N, _ = G.shape()
        E_box = cdc.IntervalVector(N, cdc.Interval(-1, 1))
        for i in range(N):
            gi_norm = 0
            for j in range(N):
                gi_norm += cdc.sqr(G[i][j]).ub()
            gi_norm = cdc.sqrt(gi_norm)
            E_box[i] = gi_norm * E_box[i]
    return E_box


def propagate_ellipsoid(Q: cdc.IntervalMatrix, J: cdc.IntervalMatrix, J_box: cdc.IntervalMatrix):
    Q_mid = mid_matrix(Q)
    G_inv = cdc.IntervalMatrix(scipy.linalg.sqrtm(Q_mid))  # set as interval matrix for guaranteed operation
    # TODO guaranteed SQRTM
    J_inv = matrix_inversion(J)
    G = matrix_inversion(G_inv)
    N, _ = Q.shape()

    U_box = cdc.IntervalVector(N, cdc.Interval(-1, 1))  # unit box
    b_box = (G_inv * J_inv * J_box * G - cdc.IntervalMatrix(
        np.eye(N))) * U_box
    b_box_sqr = np.array([cdc.sqr(b_box[i]).ub() for i in range(N)])
    rho = np.sum(b_box_sqr) ** 0.5

    Q_out = 1 / (1 + rho) ** 2 * transpose(J_inv) * Q * J_inv
    return Q_out, rho


def mid_matrix(A: cdc.IntervalMatrix):
    n, m = A.shape()
    res = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            res[i][j] = A[i][j].mid()
    return res


def mid_vector(v: cdc.IntervalVector):
    n = v.size()
    res = np.zeros(n)
    for i in range(n):
        res[i] = v[i].mid()
    return res


def matrix_inversion(A: cdc.IntervalMatrix):
    # A: n*n Interval matrix
    # compute the Interval matrix B such that AB=I using a Gaussian elimination
    # use preconditioning to inverse A

    # center of A
    Ac = mid_matrix(A)
    B0 = cdc.IntervalMatrix(np.linalg.inv(Ac))

    M = A * B0
    N = fast_matrix_inversion(M)
    B = B0 * N
    return B


def fast_matrix_inversion(A: cdc.IntervalMatrix):
    # A: n*n Interval matrix
    # compute the Interval matrix B such that AB=I using a Gaussian elimination
    # it works well when A is close to identity
    n, _ = A.shape()
    B = cdc.IntervalMatrix(n, n)
    for i in range(n):
        ei = cdc.IntervalVector(np.zeros(n))
        ei[i] = cdc.Interval(1, 1)
        bi = fast_interval_gaussian_elimination(A, ei)
        for j in range(n):
            B[j][i] = bi[j]
    return B


def fast_interval_gaussian_elimination(A: cdc.IntervalMatrix, b: cdc.IntervalVector):
    # A: n*n Interval matrix
    # b: n*1 Interval vector
    # compute the Interval vector x such that Ax=b using a Gaussian elimination
    # it works well when A is close to identity

    n, _ = A.shape()
    A_ = cdc.IntervalMatrix(A)
    x = cdc.IntervalVector(b)
    for i in range(n):
        # the pivot is A[i,i]
        p = A_[i][i]
        for j in range(i + 1, n):
            fi = A_[j][i] / p
            for k in range(n):
                A_[j][k] = A_[j][k] - fi * A_[i][k]
            x[j] = x[j] - fi * x[i]
    # back substitution
    for i in range(n - 1, -1, -1):
        p = A_[i][i]
        for j in range(i - 1, -1, -1):
            fi = A_[j][i] / p
            for k in range(n):
                A_[j][k] = A_[j][k] - fi * A_[i][k]
            x[j] = x[j] - fi * x[i]

    # normalize
    for i in range(n):
        x[i] = x[i] / A_[i][i]
    # print("A_=",A_)
    # print("x=",x)
    return x


def test_positive_definite_by_cholesky(A: cdc.IntervalMatrix):  # A = L@L.T, square symetric matrix
    n, m = A.shape()
    L = cdc.IntervalMatrix(n, m)  # initialisation with same size
    if n != m:
        print("A is not square")
        return False, L

    for j in range(n):  # for every column
        S = cdc.Interval(0, 0)

        # diagonal element
        for k in range(0, j):
            S += cdc.sqr(L[j][k])
        U = A[j][j] - S
        if U.lb() < 0:
            return False, L
        L[j][j] = cdc.sqrt(U)

        # the rest of the column
        for i in range(j + 1, n):
            S = cdc.Interval(0, 0)
            for k in range(0, j):
                S += L[j][k] * L[i][k]
            L[i][j] = (A[i][j] - S) / L[j][j]
            L[j][i] = cdc.Interval(0, 0)
    return True, L


# ---------------------------------------------------------------------------------------------
# Guaranteed square root (using newton contractor)
# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
# symmetric square root
# ---------------------------------------------------------------------------------------------
import sympy as sp


class SQRTM_OPERATOR:
    def __init__(self, n):
        self.n = n
        self.y, self.Jf, self.x = sqrtm_symbolic(n)
        self.y_str = sym_2_str(self.y)
        self.Jf_str = sym_2_str(self.Jf)

    def sqrtm(self, Q, epsilon=1e-3):
        # Q: n*n positive definite symmetric matrix
        # epsilon: inflation gain to initialise the newton contractor
        # return the n*n positive definite symmetric matrix A such that A*A=Q
        # as a cdc.IntervalMatrix
        A_nonguaranteed = scipy.linalg.sqrtm(mid_matrix(Q))
        q = cdc.IntervalVector(sym_matrix_2_vector(Q))
        a = cdc.IntervalVector(sym_matrix_2_vector(A_nonguaranteed)).inflate(epsilon)
        for i in range(10):
            a = newton_contractor(a, q, self.Jf_str, self.y_str)
        A = vector_2_sym_matrix(a)
        return A


def sym_matrix_2_vector(M):
    # M: n*n symetric matrix
    # return the vector x of the n*(n+1)/2 unique elements of M =
    # [x1 x2 ... xn]
    # [x2 xn+1 xn+2 ... x2n-1]
    # [x3 xn+2 x2n ... x3n-2]
    # ...
    # [xn x2n-1 x3n-2 ... x(n+1)n/2]

    if type(M) is np.ndarray:
        n, _ = M.shape
        m = int(n * (n + 1) / 2)
        x = np.zeros(m)
        k = 0
        for i in range(n):
            for j in range(i, n):
                x[k] = M[i, j]
                k = k + 1

    if type(M) is sp.Matrix:
        n, _ = M.shape
        m = int(n * (n + 1) / 2)
        x = sp.zeros(m, 1)
        k = 0
        for i in range(n):
            for j in range(i, n):
                x[k] = M[i, j]
                k = k + 1

    if type(M) is cdc.IntervalMatrix:
        n, _ = M.shape()
        m = int(n * (n + 1) / 2)
        x = cdc.IntervalVector(m)
        k = 0
        for i in range(n):
            for j in range(i, n):
                x[k] = M[i][j]
                k = k + 1
    return x


def vector_2_sym_matrix(x):
    # x : n*(n+1)/2 vector
    # return the n*n symetric matrix A defined from x
    # inverse transformation of sym_matrix_2_vector
    if type(x) is np.ndarray:
        m, _ = x.shape
        n = int((np.sqrt(1 + 8 * m)) / 2)
        A = np.zeros((n, n))
    if type(x) is cdc.IntervalVector:
        m = x.size()
        n = int((np.sqrt(1 + 8 * m)) / 2)
        A = cdc.IntervalMatrix(n, n)
    if type(x) is tuple:
        m = len(x)
        n = int((np.sqrt(1 + 8 * m)) / 2)
        A = sp.zeros(n, n)
    for i in range(n):
        for j in range(i, n):
            if type(x) is cdc.IntervalVector:
                A[i][j] = x[int(i * n - (i - 1) * i / 2 + j - i)]
                A[j][i] = A[i][j]
            else:
                A[i, j] = x[int(i * n - (i - 1) * i / 2 + j - i)]
                A[j, i] = A[i, j]
    return A


def sqrtm_symbolic(n):
    # return the symbolic expression of f(x) and its jacobian
    # where x defines a n*n symetric
    # and f(x) defines the square of A, B = A*A

    # define a symbolic vector of m=n*(n+1)/2 variables
    m = int(n * (n + 1) / 2)
    x = sp.symbols('x:' + str(m))
    A = vector_2_sym_matrix(x)
    B = A * A  # the square of A
    y = sym_matrix_2_vector(B)

    # differentiate y with respect to x
    Jf = y.jacobian(x)
    return y, Jf, x


def sym_2_str(M):
    # M: sympy matrix of vector
    # return an array of str with the shape of M
    # in the expression of M, replace the symbolic variables xi by x[i] for future interval evaluation
    m, n = M.shape
    res = []
    for i in range(m):
        res_i = []
        for j in range(n):
            Mij = str(M[i, j])
            for k in range(m):
                Mij = Mij.replace("x" + str(m - 1 - k), "x[" + str(m - 1 - k) + "]")
            res_i.append(Mij)
        res.append(res_i)
    return res


def str_2_interval(M_str, x: cdc.IntervalVector):
    # M_str: array of str
    # x: IntervalVector of the symbolic variables
    # output - M_box : IntervalMatrix of the Jacobian matrix
    # replace the symbolic variables in Jf_str by their values in x_val and evaluate the result
    m = len(M_str)
    n = len(M_str[0])
    if n > 1:
        res = cdc.IntervalMatrix(m, n)
    else:
        res = cdc.IntervalVector(m)
    for i in range(m):
        for j in range(n):
            # print(Jf_str[i][j])
            if n > 1:
                res[i][j] = cdc.Interval(eval(M_str[i][j]))
            else:
                res[i] = cdc.Interval(eval(M_str[i][j]))
    return res


def newton_contractor(x: cdc.IntervalVector, q: cdc.IntervalVector, Jf_str, f_str):
    # Newton contractor contract x such that f(x)=q with
    # Jf_str the Jacobian matrix of f (string expression)
    # f_str the string expression of f
    x0 = mid_vector(x)
    Jf_box = str_2_interval(Jf_str, x)
    p = x - cdc.IntervalVector(x0)
    b = str_2_interval(f_str, cdc.IntervalVector(x0)) - q

    p_bis = fast_interval_gaussian_elimination(Jf_box, -b)
    # intersect p and p_bis
    p = p & p_bis

    if p.is_empty() or p_bis.width() == np.inf:
        raise ValueError("Newton contractor failed to contract")
    else:
        x = x & (p + cdc.IntervalVector(x0))
        return x


# ---------------------------------------------------------------------------------------------
# drawing functions
# ---------------------------------------------------------------------------------------------
def draw_box(ax, box, col, linewidth=3, linestyle="-", label=None):
    x1 = box[0].lb()
    x2 = box[0].ub()
    y1 = box[1].lb()
    y2 = box[1].ub()
    ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], col, linewidth=linewidth, linestyle=linestyle, label=label)


def draw_box_projected(ax, E_box, xi=0, xj=1, col='r', label=None):
    E_box_projected = cdc.IntervalVector(2)
    E_box_projected[0] = E_box[xi]
    E_box_projected[1] = E_box[xj]
    draw_box(ax, E_box_projected, col=col, label=label)


def draw_ellipse(ax_, Q_, mu_=np.zeros(2), linewidths=3, col="r", in_col=False, linestyle="-", label=None):
    # alternative to draw ellipse, use matplotlib.patches.Ellipse

    xy = (mu_[0], mu_[1])  # center of ellipse

    Gamma = np.linalg.inv(scipy.linalg.sqrtm(Q_))
    U, S, V = np.linalg.svd(Gamma)  # Singular Value Decomposition

    width = 2 * float(S[0])
    height = 2 * float(S[1])
    angle = np.arctan2(U[1, 0], U[0, 0]) * 180 / np.pi
    if in_col:
        E1 = patches.Ellipse(xy, width, height, angle, linewidth=linewidths, fill=True, color=in_col,
                             linestyle=linestyle)
        ax_.add_artist(E1)
    E = patches.Ellipse(xy, width, height, angle, linewidth=linewidths, fill=False, color=col, linestyle=linestyle,
                        label=label)
    ax_.add_artist(E)


def project_ellipsoid(mu, G, d, T):
    # project ellipsoid E(mu,Q) = {x in R^n | (x-mu).T*G.{-T}*G^{-1}*(x-mu)<1}
    # on the affine plan A = {x|x=d+Tt} [Pope -2008]
    # reduce the dimensions of mu and Q

    TTG = T.T @ G  # T.transpose() * Gamma
    U, Sc, _ = np.linalg.svd(TTG, full_matrices=True)
    E = np.diag(Sc)
    G = U @ E @ U.T
    mu = T.T @ (d + T @ T.T @ (mu - d))
    return mu, G


def draw_ellipse_projected(ax, Q, mu=None, xi=0, xj=1, linewidths=3, col="r", in_col=None, linestyle='-', label=None):
    # draw 2d ellipse from center vector mu and shape matrix G

    dim = len(Q)
    if not mu:
        mu = np.zeros(dim)

    # affine space of the projection
    d = np.zeros(dim)
    T = np.zeros((dim, 2))
    T[xi, 0] = 1
    T[xj, 1] = 1

    # project the ellipsoid
    G = np.linalg.inv(scipy.linalg.sqrtm(Q))
    Gp = G
    mup = mu
    mup, Gp = project_ellipsoid(mup, Gp, d, T)
    Gp_inv = np.linalg.inv(Gp)
    Qp = Gp_inv.T @ Gp_inv
    draw_ellipse(ax, Qp, mup, linewidths=linewidths, col=col, in_col=in_col, linestyle=linestyle, label=label)
