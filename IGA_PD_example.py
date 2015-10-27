
# coding: utf-8

# This notebook provides an example code for using the IGA2D class

# In[1]:

import IGA
import numpy as np
import matplotlib.pyplot as plt
#get_ipython().magic(u'matplotlib inline')


# In[2]:

def run_case_1(num_knots, order, delta, norm, quad_degree=10):

    h = 1.0 / num_knots

    if delta > h:
        num_boundary_elements = np.ceil(delta / h)
    else:
        num_boundary_elements = 1

    omega_p1 = np.linspace(-delta, 0, num=(num_boundary_elements + 1))
    omega = np.linspace(0, 1, num=(num_knots+1))
    omega_p2 = np.linspace(1, 1 + delta, num=(num_boundary_elements + 1))

    knot_vector = np.r_[-delta * np.ones(order), omega_p1[:-1], omega[:-1], omega_p2, np.ones(order) * (1 + delta)]

    iga = IGA.PD1D(knot_vector, order, delta)
    iga.degree = quad_degree
    
    u = lambda x: x * (1 - x)
    b = lambda x: np.ones(x.shape[0])

    iga.compute_solutions(u, b, num_boundary_elements)
    
    return iga.compute_error(norm=norm)



# In[ ]:

dofs = np.array([100,700])
errs = [ run_case_1(num_knots, order=1, delta=0.25, norm=2, quad_degree=4) for num_knots in dofs ]


# In[ ]:



# In[ ]:

#Fit a straight line
coefs = np.polyfit(np.log10(1.0 / dofs), np.log10(errs), 1)
y = 10 ** (coefs[0] * np.log10(1.0 / dofs) + coefs[1])
#Plot
plt.loglog(1.0 / dofs, y, 'b-')
plt.loglog(1.0 / dofs, errs, 'b^')
plt.xlabel("$\log_{10} h$")
plt.ylabel("$\log_{10} \Vert Error \Vert_{L_2}$");

