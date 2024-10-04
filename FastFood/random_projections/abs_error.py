import numpy as np
import sklearn as sc
import projections
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.kernel_approximation import RBFSampler
from sklearn_extra.kernel_approximation import Fastfood

#dimension
d = 10 
#data points
num_data = 4000 
x = np.random.rand(num_data, d)

#universal scale
scale = 0.5

#exact rbf kernel
exact_rbf = rbf_kernel(x, gamma=(1/(2*scale**2)))

#new n dimension for rks and ff
dimensions = [16,32,64,128,512,1024,2048,4096,8192]

#rks error
rks_error = []
for dim in dimensions:
    #rks approx
    rks = RBFSampler(gamma=(1/(2*scale**2)),n_components=dim)
    rks.fit(x)
    rks_approx = rks.transform(x)

    rks_approx = rks_approx @ rks_approx.T
    
    difference = np.linalg.norm(exact_rbf-rks_approx, 'fro')
    rks_error.append(difference/num_data)


rks_error_mine = []
#rks error
for dim in dimensions:
    #rks approx
    rks = projections.rks(dim, scale)
    rks.fit(x)
    rks_approx = rks.transform(x)

    rks_approx = np.matmul(rks_approx, rks_approx.T)

    difference = np.linalg.norm(np.abs(exact_rbf-rks_approx), 'fro')
    rks_error_mine.append(difference/num_data)

#ff error
ff_error = []
for dim in dimensions:
    #ff approx
    ff = Fastfood(sigma=scale, n_components=dim)
    ff_approx = ff.fit_transform(x)

    ff_approx = ff_approx @ ff_approx.T

    difference = np.linalg.norm(exact_rbf-ff_approx, 'fro')
    ff_error.append(difference/num_data)


plt.plot(dimensions,rks_error, label='RKS_Approx', marker='o')
plt.plot(dimensions,rks_error_mine, label='RKS_Personal_Approx', marker='o')
plt.plot(dimensions,ff_error, label='FF_Approx', marker='o')
plt.xlabel('Dimension (n)')
plt.ylabel('Error')
plt.title('Approximation Error vs. RBF Kernel')
plt.legend()
plt.savefig('graphs/abs_error.png', bbox_inches='tight')
plt.show()

plt.loglog(dimensions,rks_error, label='RKS_Approx', marker='o')
plt.loglog(dimensions,rks_error_mine, label='RKS_Personal_Approx', marker='o')
plt.loglog(dimensions,ff_error, label='FF_Approx', marker='o')
plt.xlabel('Dimension (n)')
plt.ylabel('Error')
plt.title('Approximation Error vs. RBF Kernel')
plt.legend()
plt.savefig('graphs/abs_error_log.png', bbox_inches='tight')
plt.show()
