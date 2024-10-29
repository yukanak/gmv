import numpy as np
import matplotlib.pyplot as plt

N = 1000000
X_std = 5.0
Y_std = 10.0

# Case 1: Uncorrelated X and Y
print(f'Case 1: {N} uncorrelated X and Y samples, with X std of {X_std} and Y std of {Y_std}')
cov_uncorr = np.array([[X_std**2, 0],[0, Y_std**2]])
M_uncorr = np.random.multivariate_normal(np.array([0,0]), cov_uncorr, N)
X_uncorr = M_uncorr[:,0]
Y_uncorr = M_uncorr[:,1]

norm = 1/X_std**2 + 1/Y_std**2
Z_uncorr = 1/norm * (1/X_std**2 * X_uncorr + 1/Y_std**2 * Y_uncorr)
Z_uncorr_var_num = np.var(Z_uncorr)
Z_uncorr_var_an = 1/norm**2 * (1/X_std**2 + 1/Y_std**2)
print(f'The numerical variance of Z for uncorrelated X and Y is: {Z_uncorr_var_num}')
print(f'The analytic variance of Z for uncorrelated X and Y is: {Z_uncorr_var_an}')

# Case 2: Introduce correlation between X and Y, suboptimal
print(f'Case 2: {N} correlated X and Y samples, with X std of {X_std} and Y std of {Y_std}')
corr = -1*0.5
cov_corr = np.array([[X_std**2, X_std*Y_std*corr],[X_std*Y_std*corr, Y_std**2]])
M = np.random.multivariate_normal(np.array([0,0]), cov_corr, N)
X = M[:,0]
Y = M[:,1]

Z_subopt = 1/norm * (1/X_std**2 * X + 1/Y_std**2 * Y)
Z_subopt_var_num = np.var(Z_subopt)
print(f'The numerical variance of suboptimal Z for correlated X and Y is: {Z_subopt_var_num}')

# Case 3: Introduce correlation between X and Y, optimal
w, v = np.linalg.eig(cov_corr) # w are eigenvalues, v are eigenvectors
# np.transpose(v) @ cov_corr @ v is equal to np.diag(w)
M_new = np.transpose(v) @ [X,Y]
X_new = M_new[0,:]
Y_new = M_new[1,:]
print(f'The eigenvalues of the correlation matrix are {w[0]} and {w[1]}')
print(f'The variances of the noise random variables in the new basis are {np.var(X_new)} and {np.var(Y_new)}')
norm_new = 1/w[0] + 1/w[1]
Z_opt = 1/norm_new * (1/w[0] * X_new + 1/w[1] * Y_new)
Z_opt_var_num = np.var(Z_opt)
Z_opt_var_an = 1/norm_new**2 * (1/w[0] + 1/w[1])
print(f'The numerical variance of optimal Z for correlated X and Y is: {Z_opt_var_num}')
print(f'The analytic variance of optimal Z for correlated X and Y is: {Z_opt_var_an}')

# Plot
plt.figure(1)
plt.scatter(X_uncorr, Y_uncorr, marker='x', label='Uncorrelated')
plt.scatter(X, Y, marker='x', label='Correlated')
plt.scatter(X_new, Y_new, marker='x', label='Correlated, New Basis')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(loc='upper left')
plt.ylim(-60,60)
plt.xlim(-60,60)

fig = plt.figure(2)
ax = fig.add_subplot(projection='3d')
ax.scatter(X_uncorr, Y_uncorr, Z_uncorr, marker='x', s=1, label=f'Uncorrelated, Z Variance {Z_uncorr_var_num:.2f}')
ax.scatter(X, Y, Z_subopt, marker='x', s=1, label=f'Correlated, Suboptimal, Z Variance {Z_subopt_var_num:.2f}')
ax.scatter(X_new, Y_new, Z_opt, marker='x', s=1, label=f'Correlated, Optimal, Z Variance {Z_opt_var_num:.2f}')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.legend()

plt.figure(3)
plt.subplot(121)
max_Z_uncorr = max(Z_uncorr)  # Find the maximum y value
xs = [x for x in range(N) if Z_uncorr[x] > max_Z_uncorr/2.0]
print(Z_uncorr[max(xs)] -Z_uncorr[min(xs)])
plt.hist(Z_uncorr, orientation=u'horizontal')
plt.subplot(122)
plt.scatter(X_uncorr, Z_uncorr, marker='x', label='Uncorrelated')
plt.scatter(X_new, Z_opt, marker='x', label='Correlated, New Basis')
plt.show()
