from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


class Bspline(object):
    """
       Numpy implementation of Cox - de Boor algorithm in 1D

       inputs:
           knot_vector: Python list or Numpy array containing knot vector 
                        entries
           order: Order of interpolation, e.g. 0 -> piecewise constant between 
                  knots, 1 -> piecewise linear between knots, etc.
       outputs:
           basis object that is callable to evaluate basis functions at given 
           values of knot span
    """
    
    def __init__(self, knot_vector, order):
        """Initialize attributes"""
        self.knot_vector = np.array(knot_vector)
        self.p = order

        
    def __basis0(self, xi):
        """Order zero basis"""
        return np.where(np.all([self.knot_vector[:-1] <=  xi[:, np.newaxis], 
            xi[:,np.newaxis] < self.knot_vector[1:]],axis=0), 1.0, 0.0)
    
    def __basis(self, xi, p, compute_derivatives=False):
        """
           Recursive Cox - de Boor function to compute basis functions and 
           optionally their derivatives.
        """
        
        if p == 0:
            return self.__basis0(xi)
        else:
            basis_p_minus_1 = self.__basis(xi, p - 1)
        
        first_term_numerator = xi[:, np.newaxis] - self.knot_vector[:-p] 
        first_term_denominator = self.knot_vector[p:] - self.knot_vector[:-p]
        
        second_term_numerator = self.knot_vector[(p + 1):] - xi[:, np.newaxis]
        second_term_denominator = (self.knot_vector[(p + 1):] - 
                                   self.knot_vector[1:-p])
                
        
        #Change numerator in last recursion if derivatives are desired
        if compute_derivatives and p == self.p:
            
            first_term_numerator = np.ones((len(xi), 
                                            len(first_term_denominator))) * p
            second_term_numerator = np.ones((len(xi), 
                                             len(second_term_denominator))) * -p
            
        #Disable divide by zero error because we check for it
        with np.errstate(divide='ignore', invalid='ignore'):
            first_term = np.where(first_term_denominator != 0.0, 
                                  (first_term_numerator / 
                                   first_term_denominator), 0.0)
            second_term = np.where(second_term_denominator != 0.0,
                                   (second_term_numerator / 
                                    second_term_denominator), 0.0)

        return  (first_term[:,:-1] * basis_p_minus_1[:,:-1] + 
                 second_term * basis_p_minus_1[:,1:])
            
    
    def __call__(self, xi):
        """
           Convenience function to make the object callable.
        """
        return self.__basis(xi, self.p, compute_derivatives=False)
    
    def d(self, xi):
        """
           Convenience function to compute derivate of basis functions.  
        """
        return self.__basis(xi, self.p, compute_derivatives=True)
    
    def plot(self):
        """
           Convenience function to plot basis functions over full 
           range of knots.
        """
        
        x_min = np.min(self.knot_vector)
        x_max = np.max(self.knot_vector)
        
        x = np.linspace(x_min, x_max, num=1000, endpoint=False)
        
        N = self(x).T
        
        for n in N:
            
            plt.plot(x,n)
            
        return plt.show()
    
    def dplot(self):
        """
           Convenience function to plot derivatives of basis functions over 
           full range of knots.
        """
        
        x_min = np.min(self.knot_vector)
        x_max = np.max(self.knot_vector)
        
        x = np.linspace(x_min, x_max, num=1000, endpoint=False)
        
        N = self.d(x).T
        
        for n in N:
            
            plt.plot(x,n)
            
        return plt.show()
    

class NURBS_2D_Shape_Functions(Bspline):


    def __init__(self, knot_vector_1, p_1, knot_vector_2, p_2, weights):

        self.N = Bspline(knot_vector_1, p_1)
        self.M = Bspline(knot_vector_2, p_2)

        self.p_1 = p_1
        self.p_2 = p_2
        self.weights = np.array(weights)


    def __call__(self, xi, eta, derivative=None):

        numerator = np.einsum('...i,...j', self.N(xi) * self.weights, 
                                           self.M(eta) * self.weights)

        W = (np.einsum('...i,i', self.N(xi), self.weights) *  
             np.einsum('...i,i', self.M(eta), self.weights))

        R = numerator / W[:, np.newaxis, np.newaxis]

        if derivative == 'xi':

            dW = (np.einsum('...i,i', self.N.d(xi), self.weights) *  
                  np.einsum('...i,i', self.M(eta), self.weights))

            R = (np.einsum('...i,...j', self.N.d(xi) * self.weights, 
                 self.M(eta) * self.weights) + dW[:, np.newaxis, np.newaxis] * 
                 R) / W[:, np.newaxis, np.newaxis]   

        if derivative == 'eta':

            dW = (np.einsum('...i,i', self.N(xi), self.weights) *  
                  np.einsum('...i,i', self.M.d(eta), self.weights))

            R = (np.einsum('...i,...j', self.N(xi) * self.weights, 
                 self.M.d(eta) * self.weights) + dW[:, np.newaxis, np.newaxis] *
                 R) / W[:, np.newaxis, np.newaxis]   
        
        return R

    def d_xi(self, xi, eta):

        return self.__call__(xi, eta, derivative='xi')

    def d_eta(self, xi, eta):

        return self.__call__(xi, eta, derivative='eta')
                

    def plot(self, shape_function_number=0, derivative=None):

        xi_min = np.min(self.N.knot_vector)
        xi_max = np.max(self.N.knot_vector)

        eta_min = np.min(self.M.knot_vector)
        eta_max = np.max(self.M.knot_vector)

        xi = np.linspace(xi_min, xi_max, num=50, endpoint=False)
        eta = np.linspace(eta_min, eta_max, num=50, endpoint=False)

        x, y = np.meshgrid(xi, eta)

        basis = self(x.flatten(), y.flatten(), derivative)

        z = [basis[:,i,j].reshape(x.shape) for i in range(3) for j in range(3)]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(x, y, z[shape_function_number], rstride=1, 
                               cstride=1, cmap=cm.coolwarm, linewidth=0, 
                               antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        


class IGA2D(NURBS_2D_Shape_Functions):


    def __init__(self, knot_vector_1, p_1, knot_vector_2, p_2, 
                 control_points, weights):

        self.R = NURBS_2D_Shape_Functions(knot_vector_1, p_1, 
                                          knot_vector_2, p_2,
                                          weights)

        self.p_1 = p_1
        self.p_2 = p_2


        self.num_of_basis_functions_1 = len(self.R.N.knot_vector) - p_1 - 1
        self.num_of_basis_functions_2 = len(self.R.M.knot_vector) - p_2 - 1

        self.num_of_global_basis_functions = (self.num_of_basis_functions_1 *
                                              self.num_of_basis_functions_2)


        self.num_of_elements = ((self.num_of_basis_functions_1 - p_1) *
                                (self.num_of_basis_functions_2 - p_2))


        self.nurbs_coords = self.__build_nurbs_coord_array()

        self.connectivity_array = self.__build_connectivity_array()
   

    def __build_nurbs_coord_array(self):
        """
           Builds an array of coordinates in index space where each global basis
           function begins, the position in the array is the global basis id
        """

        #Index arrays in each basis direction
        i_arr = np.arange(self.num_of_basis_functions_1, dtype=np.int)
        j_arr = np.arange(self.num_of_basis_functions_2, dtype=np.int)

        #Construct the coordinate array
        return np.array([ (i, j) for j in j_arr for i in i_arr ], dtype=np.int)

    def __build_connectivity_array(self):
        """
           Builds an array that relates the local basis function #'s for each 
           element to the global basis function #'s.  Ordering starts with the 
           lower-left hand corner of an element in index space and moves 
           backwards starting in the xi direction, followed by the eta 
           direction row-by-row
        """

        #The total # of basis functions
        number_of_basis_functions = len(self.nurbs_coords)

        #The global basis function id's
        global_basis_ids = np.arange(number_of_basis_functions, dtype=np.int)
        #Here we reshape the array to mimic the layout in basis index space, 
        #this makes finding the "lower left-hand" corner easier
        global_basis_ids.shape = (self.num_of_basis_functions_2,
                                  self.num_of_basis_functions_1)

        #i_arr and j_arr are convenience indices for iterating through the 
        #basis functions to determine the "lower left-hand" corner of the 
        #elements.  This procedure accounts for elements of zero measure due 
        #to possibly open knot vectors.
        i_arr = np.arange(self.p_1, self.num_of_basis_functions_1, dtype=np.int)
        j_arr = np.arange(self.p_2, self.num_of_basis_functions_2, dtype=np.int)
        
        #Array of element corner indices pairs, i.e. (i,j)
        elem_corner = [(i,j) for i in j_arr for j in i_arr]

        #Constructs the connectivity array.  This does a slice from the element
        #corner location (i,j), backwards by p_1 in the \xi direction and p_2
        #in the \eta direction to get all basis functions that have support on
        #each element. The it flatten's the matrix to make an array and reverses 
        #the order with the [::-1] to be consistent with the convention that
        #the arrays start with the corner basis id and move backwards in \xi
        #and \eta.  Excuse the 
        return  np.array([(global_basis_ids[(i-self.p_2):(i+1),(j-self.p_1):(j+1)].flatten())[::-1] 
            for i,j in elem_corner])
        

    def compute_jacobian_matrix_and_inverse(self, xi, eta):
            """
               Compute the Jacobian matrix, Det(J) and B for every element
            """
            
            ni = self.nurbs_coords[self.connectivity_array[:,0],0]
            nj = self.nurbs_coords[self.connectivity_array[:,0],1]

            xi = ((self.R.N.knot_vector[ni+1] - self.R.N.knot_vector[ni]) * xi[:, np.newaxis] + 
                  (self.R.N.knot_vector[ni+1] - self.R.N.knot_vector[ni])) / 2.0

            eta = ((self.R.M.knot_vector[ni+1] - self.R.M.knot_vector[ni]) * xi[:, np.newaxis] + 
                   (self.R.M.knot_vector[ni+1] - self.R.M.knot_vector[ni])) / 2.0


            #Understand we are broadcasting the dot product to every element
            #J11 = np.dot(x[con], self.dNdxi(xi))
            #J12 = np.dot(y[con], self.dNdxi(xi))
            #J21 = np.dot(x[con], self.dNdeta(eta))
            #J22 = np.dot(y[con], self.dNdeta(eta))
            
            ##detJ is a vector containing the Jacobian determinate for every element
            #self.detJ = J11 * J22 - J12 * J21
            
            #self.Jinv11 =  J22 / self.detJ
            #self.Jinv12 = -J12 / self.detJ
            #self.Jinv21 = -J21 / self.detJ
            #self.Jinv22 =  J11 / self.detJ
        
        
    #def compute_B_matrix(self, xi, eta):
        #"""Computes the B matrix for a given xi and eta"""
        
        ##Returns detJ and Jinv components for this xi and eta
        #self.compute_jacobian_matrix_and_inverse(xi, eta)
        
        
        #Nmat = np.zeros((3, 4), dtype=np.double)
        #Nmat[0,:] = self.dNdxi(xi)
        #Nmat[1,:] = self.dNdeta(eta)
        #Nmat[2,:] = self.N(xi, eta)
        
        #zero = np.zeros(len(self.detJ))
        #one = np.ones(len(self.detJ))
        
        #Jmat = np.array([[self.Jinv11, self.Jinv12, zero],
                         #[self.Jinv21, self.Jinv22, zero],
                         #[       zero,        zero,  one]])
