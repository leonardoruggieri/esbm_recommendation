#%%
import pandas as pd
import numpy as np 
import random
from numpy import matlib
from scipy.special import betaln
from scipy.special import gammaln
import time
import matplotlib.pyplot as plt
#%%
class esbmb():
    def __init__(self, prior_u = "DP", alpha_urn_u = 2.55,
                       prior_i ="DP", alpha_urn_i = 2.55,
                       a = 1, b = 1,
                       sigma = None, beta = None, gamma = None, components = None):
        '''
        Initialization of the parameters.
        The user can modify the the prior choices, the hyperparameters of urn schemes and the values of a and b

        Parameters:
        ----------
           (under construction)
        '''
        #self.prior_u = prior_u
        self.alpha_urn_u = alpha_urn_u
        self.prior_i = prior_i
        self.prior_u = prior_u
        self.alpha_urn_i = alpha_urn_i 
        self.a = a
        self.b = b 
        self.sigma = sigma 
        self.gamma = gamma 
        self.beta = beta
        self.components = components

        self.ll = []

    def fit(self, Y, its, 
            xu = None, xi = None, 
            alpha_xu = 1.5, alpha_xi = 1.5, 
            beta_xu = 1, beta_xi = 1, 
            cont_par = [0, 1, 1, 1],
            xu_type = None, xi_type = None, 
            verbose = False):
        
        self.Y = Y
        self.its = its
        self.xu = xu
        self.xi = xi
        self.alpha_xu = alpha_xu
        self.alpha_xi = alpha_xi
        self.beta_xu = beta_xu
        self.beta_xi = beta_xi
        self.cont_par = cont_par # parameters of a continuous covariate
        self.xu_type = xu_type
        self.xi_type = xi_type
        self.verbose = verbose

        self.__howmany_covariates()

        self.__check_covariates()
        self.__covariates_preprocessing()

        self.__check_prior_parameters_u()
        self.__check_prior_parameters_i()

        self.__gibbs()

        self.__compute_block_interactions()

    def __covariates_preprocessing(self):

        # Users covariates
        if self.xu is not None and self.n_cov_u == 1:
            print("------------------\nOne covariate for 'user' entity provided:")
            if self.xu_type == "categ":
                self.alpha0u, self.alpha_xu, self.beta_xu = self.hyperparameters_preprocessing_categ(self.xu, self.alpha_xu, self.beta_xu)
                self.XU = self.__covar_preprocess_categ(self.xu)
                print(" - Categorical covariate for users preprocessed.")
            if self.xu_type == "count":
                self.alpha0u, self.alpha_xu, self.beta_xu = self.hyperparameters_preprocessing_count(self.xu, self.alpha_xu, self.beta_xu)
                self.XU = self.__covar_preprocess_count(self.xu)
                print(" - Count-type covariate for users preprocessed.")
            
            # (No need to intervene for the parameters of a single continuous covariate)

        elif self.xu is not None and self.n_cov_u > 1:
            print(f"------------------\n{self.n_cov_u} covariates for 'user' entity provided:")
            self.XU = []
            self.alpha0u_t = []
            self.alpha_xu_t = []
            self.beta_xu_t = []

            # Questi due "if" per gestire caso estremo di non inserimento iperparametri:
            if type(self.alpha_xu) == int or type(self.alpha_xu) == float:
                self.alpha_xu = np.repeat(self.alpha_xu, self.xu.shape[0]) # uno per ogni covariata

            if type(self.beta_xu) == int or type(self.beta_xu) == float:
                self.beta_xu = np.repeat(self.beta_xu, self.xu.shape[0]) # uno per ogni covariata

            self.index_cont_cov = dict()

            for cov_u in range(self.xu.shape[0]): # per ogni covariata...
                if self.xu_type[cov_u] == "categ":
                    # NB: "self.XU[k][:,:]" per prendere XU della covariata k-esima.
                    alpha0u_temp, alpha_xu_temp, beta_xu_temp = self.hyperparameters_preprocessing_categ(self.xu[cov_u], self.alpha_xu[cov_u], self.beta_xu[cov_u])
                    XU_temp = self.__covar_preprocess_categ(self.xu[cov_u])
                    print(f" - Categorical covariate {cov_u} for users preprocessed.")
                if self.xu_type[cov_u] == "count":
                    alpha0u_temp, alpha_xu_temp, beta_xu_temp = self.hyperparameters_preprocessing_count(self.xu[cov_u], self.alpha_xu[cov_u], self.beta_xu[cov_u])
                    XU_temp = self.__covar_preprocess_count(self.xu[cov_u])
                    print(f" - Count-type covariate {cov_u} for users preprocessed.")
                
                self.cont_cov = 0 # index to use to pick the right parameters from cont_par
                if self.xu_type[cov_u] == "cont":
                    alpha0u_temp, alpha_xu_temp, beta_xu_temp = self.hyperparameters_preprocessing_continuous()
                    XU_temp = self.__covar_preprocess_continuous()
                    self.index_cont_cov[cov_u] = self.cont_cov
                    self.cont_cov += 1
                    print(f" - Continuous covariate {cov_u} for users preprocessed.")


                self.XU.append(XU_temp)
                self.alpha0u_t.append(alpha0u_temp)
                self.alpha_xu_t.append(alpha_xu_temp)
                self.beta_xu_t.append(beta_xu_temp)

            self.alpha0u = self.alpha0u_t
            self.alpha_xu = self.alpha_xu_t
            self.beta_xu = self.beta_xu_t



        # Items covariates
        if self.xi is not None and self.n_cov_i == 1:
            print("------------------\nOne covariate for 'item' entity provided:")
            if self.xi_type == "categ":
                self.alpha0i, self.alpha_xi, self.beta_xi = self.hyperparameters_preprocessing_categ(self.xi, self.alpha_xi, self.beta_xi)
                self.XI = self.__covar_preprocess_categ(self.xi)
                print(" - Categorical covariate for items preprocessed.")
            if self.xi_type == "count":
                self.alpha0i, self.alpha_xi, self.beta_xi = self.hyperparameters_preprocessing_count(self.xi, self.alpha_xi, self.beta_xi)
                self.XI = self.__covar_preprocess_count(self.xi)
                print(" - Count-type covariate for items preprocessed.")
        
        elif self.xi is not None and self.n_cov_i > 1:
            print(f"------------------\n{self.n_cov_i} covariates for 'item' entity provided:")
            self.XI = []
            self.alpha0i_t = []
            self.alpha_xi_t = []
            self.beta_xi_t = []

            # Questi due "if" per gestire caso estremo di non inserimento iperparametri:
            if type(self.alpha_xi) == int or type(self.alpha_xi) == float:
                self.alpha_xi = np.repeat(self.alpha_xi, self.xi.shape[0]) # uno per ogni covariata

            if type(self.beta_xi) == int or type(self.beta_xi) == float:
                self.beta_xi = np.repeat(self.beta_xi, self.xi.shape[0]) # uno per ogni covariata
            
            self.index_cont_cov_i = dict()

            for cov_i in range(self.xi.shape[0]): # per ogni covariata...
                if self.xi_type[cov_i] == "categ":
                    # NB: "self.XU[k][:,:]" per prendere XU della covariata k-esima.
                    alpha0i_temp, alpha_xi_temp, beta_xi_temp = self.hyperparameters_preprocessing_categ(self.xi[cov_i], self.alpha_xi[cov_i], self.beta_xi[cov_i])
                    XI_temp = self.__covar_preprocess_categ(self.xi[cov_i])
                    print(f" - Categorical covariate {cov_i} for users preprocessed.")
                if self.xi_type[cov_i] == "count":
                    alpha0i_temp, alpha_xi_temp, beta_xi_temp = self.hyperparameters_preprocessing_count(self.xi[cov_i], self.alpha_xi[cov_i], self.beta_xi[cov_i])
                    XI_temp = self.__covar_preprocess_count(self.xi[cov_i])
                    print(f" - Count-type covariate {cov_i} for users preprocessed.")
                
                self.cont_cov_i = 0 # index to use to pick the right parameters from cont_par (in case more parameter vectors can be provided -> possible extension, not implemented yet)
                if self.xi_type[cov_i] == "cont":
                    alpha0i_temp, alpha_xi_temp, beta_xi_temp = self.hyperparameters_preprocessing_continuous()
                    XI_temp = self.__covar_preprocess_continuous()
                    self.index_cont_cov_i[cov_i] = self.cont_cov_i
                    self.cont_cov_i += 1
                    print(f" - Continuous covariate {cov_i} for users preprocessed.")

                self.XI.append(XI_temp)
                self.alpha0i_t.append(alpha0i_temp)
                self.alpha_xi_t.append(alpha_xi_temp)
                self.beta_xi_t.append(beta_xi_temp)

            self.alpha0i = self.alpha0i_t # ok
            self.alpha_xi = self.alpha_xi_t # ok
            self.beta_xi = self.beta_xi_t

        self.__random_initialization()

        self.indu = list(range(self.U))
        self.indi = list(range(self.I))
        # -----------------

        temp0 = self.Y @ self.zi
        self.s_full = self.zu.T @ temp0 # - np.diag(0.5*np.sum(temp0 * z, axis = 0)) NO NEED FOR THIS CORRECTION

        self.__count_possible_edges(self.zu,self.zi)
        
        likel0 = self.__marginal_loglik_bipartite(self.Y, range(self.U), range(self.I), self.a, self.b, self.s_full, self.n_cardinality)
        print(f"------------------\nInitial log-likelihood: {likel0}")
        self.ll.append(likel0)


    def __random_initialization(self):

        self.U, self.I = self.Y.shape

        zu_init = np.array(range(self.U)) # random cluster assignment for users
        zi_init = np.array(range(self.I)) # random cluster assignment for items

        self.zu = self.__indicator_matrix(zu_init)
        self.zi = self.__indicator_matrix(zi_init)

    def __count_possible_edges(self, zu, zi):
        self.m_u = np.array(np.sum(zu, axis = 0)) # no. users in each cluster
        self.m_i = np.array(np.sum(zi, axis = 0)) # no. items in each cluster

        # no. of possible interactions (edges) between pairs of clusters:
        self.n_cardinality = np.zeros(shape = (self.m_u.shape[0], self.m_i.shape[0]))
        for i in range(self.n_cardinality.shape[0]):
            for j in range(self.n_cardinality.shape[1]):
                self.n_cardinality[i,j] = self.m_u[i] * self.m_i[j]

    def __count_possible_edges_return(self, zu, zi):
        m_u_plus = np.array(np.sum(zu, axis = 0)) # no. users in each cluster
        m_i_plus = np.array(np.sum(zi, axis = 0)) # no. items in each cluster

        # no. of possible interactions (edges) between pairs of clusters:
        n_cardinality_plus = np.zeros(shape = (m_u_plus.shape[0], m_i_plus.shape[0]))
        for i in range(n_cardinality_plus.shape[0]):
            for j in range(n_cardinality_plus.shape[1]):
                n_cardinality_plus[i,j] = m_u_plus[i] * m_i_plus[j]
        return n_cardinality_plus, m_u_plus, m_i_plus


    def __covar_preprocess_categ(self, x = None):
        x = np.array(x)
        X = self.__indicator_matrix(x)
        return X

    def __covar_preprocess_count(self, x = None):
        x = np.array(x)
        X = self.__vector_to_matrix(x)
        return X

    def __covar_preprocess_continuous(self):
        '''
        Returning zero as placeholder for continuous variables.
        '''
        return np.empty(0)

    def hyperparameters_preprocessing_categ(self, x, alpha_x = None, beta_x = None):
        '''
        Vectorizing the covariates hyperparameters if a number instead of a list is given.
        Only needed for 'categ'-type covariates.
        '''
        if type(beta_x) == float or type(beta_x) == np.float64 or type(beta_x) == int or type(beta_x) == np.int64:
            beta_x = list(np.repeat(beta_x, x.shape[0]))
            print(" - beta hyperparameter set to a list.")
            # print(f"Here's the beta_x: {beta_x}.")
        if type(alpha_x) == float or type(alpha_x) == np.float64 or type(beta_x) == int or type(alpha_x) == np.int64:
            alpha_x = list(np.repeat(alpha_x,x.shape[0]))
            print(" - alpha hyperparameter set to a list.")
            # print(f"Here's the alpha_x: {alpha_x}.")
        
        alpha0 = np.sum(alpha_x)

        return alpha0, alpha_x, beta_x

    def hyperparameters_preprocessing_count(self, x, alpha_x = None, beta_x = None):
        '''
        No need of intervening on alpha and beta hyperparameters of count-type covariates.
        alpha0 will be set to zero.
        '''
        return 0, alpha_x, beta_x

    def hyperparameters_preprocessing_continuous(self):
        '''
        Returning zeros as placeholders for continuous variables.
        '''
        return 0, 0, 0


    def __indicator_matrix(self, vector):
        V = vector.shape[0]
        H = int(np.max(vector)+1)
        self.z = np.zeros(shape = (V,H))
        for i in range(vector.shape[0]):
            self.z[i,int(vector[i])] = 1
        return self.z

    def __vector_to_matrix(self, vector):
        V = vector.shape[0]
        H = int(np.max(vector)+1)
        self.z = np.zeros(shape = (V,H))
        for i in range(vector.shape[0]):
            self.z[i,int(vector[i])] = vector[i]
        return self.z

    def __vector_to_matrix_continuous(self, vector):
        '''
        Creating a diagonal matrix with the coninuous values of the covariate
        '''
        V, H = vector.shape
        self.z = np.zeros(shape = (V,H))
        for i in range(vector.shape[0]):
            self.z[i,i] = vector[i]
        return self.z

    def __urn_DM_u(self, m_u, beta, components):
        H = m_u.shape[0]
        self.urn_values_DMu = np.append(np.array(m_u + beta), beta * (components - H) * (components > H)) # last factor works as an indicator function and activates the expression for the cluster that is not yet assigned, if H is still less than the prespecified components number.
        return self.urn_values_DMu
        
    def __urn_DP_u(self, m_u,alpha_urn_u):
        self.urn_values_DPu = np.append(np.array(m_u), alpha_urn_u)
        return self.urn_values_DPu

    def __urn_PY_u(self, m_u, alpha_urn_u, sigma):
        H = m_u.shape[0]
        self.urn_values_PYu = np.append(np.array(m_u - sigma), alpha_urn_u + H * sigma)
        return self.urn_values_PYu

    def __urn_GN_u(self, m_u, gamma):
        H = m_u.shape[0]
        U = Y.shape[0]
        self.urn_values_GNu = np.append(np.array((m_u+1) * (U - H + gamma)), H ** 2 - H * gamma)
        return self.urn_values_GNu

    def __urn_DM_i(self, m_i, beta, components):
        K = m_i.shape[0]
        self.urn_values_DMi = np.append(np.array(m_i + beta), beta * (components - K) * (components > K)) # last factor works as an indicator function and activates the expression for the cluster that is not yet assigned, if H is still less than the prespecified components number.
        return self.urn_values_DMi

    def __urn_DP_i(self, m_i,alpha_urn_i):
        self.urn_values_DPi = np.append(np.array(m_i), alpha_urn_i)
        return self.urn_values_DPi

    def __urn_PY_i(self, m_i, alpha_urn_i, sigma):
        K = m_i.shape[0]
        self.urn_values_PYi = np.append(np.array(m_i - sigma), alpha_urn_i + K * sigma)
        return self.urn_values_PYi

    def __urn_GN_i(self, m_i, gamma):
        K = m_i.shape[0]
        I = Y.shape[1]
        self.urn_values_GNi = np.append(np.array((m_i+1) * (I - K + gamma)), K ** 2 - K * gamma)
        return self.urn_values_GNi

    def __marginal_loglik_bipartite(self, Y, zu_labels, zi_labels, a, b, s_full, n_cardinality):

        # s_full = zu.T @ Y @ zi
        
        self.a_shk = a + self.s_full.flatten()
        self.b_mhk = b + n_cardinality.flatten()
        self.loglik = np.sum(gammaln(self.a_shk) - self.a_shk * np.log(self.b_mhk))

        return self.loglik

    def __cov_prob(self, mask_u, u):

        if self.xu is not None and self.n_cov_u == 1:

           # NON MODIFICARE QUESTO IF, FUNZIONA. 

            # For every type of covariate:
            self.x_u = self.xu[mask_u]

            log_covar_u = 0 # initialize
            
            if self.xu_type == "categ":
                self.X_u = self.XU[mask_u]
                self.vxu = self.zu_u.T @ self.X_u # rows of individuals with that cluster * columns of individuals with that covariate
                                # clusters x covariates (without user u)

                log_covar_u = np.log((self.vxu[:,self.xu[u]] + self.alpha_xu[self.xu[u]]) / (np.array(self.m_u) + self.alpha0u)) # pick hyperparameter alpha_x proper of that particular covariate value c, i.e. alpha_x[x[v]]
                log_covar_u_new = np.log(self.alpha_xu[self.xu[u]] / self.alpha0u)

            if self.xu_type == "count":
                self.X_u = self.XU[mask_u]
                self.vxu = self.zu_u.T @ self.X_u # rows of individuals with that cluster * columns of individuals with that covariate

                self.vxu_sum = self.vxu.sum(axis = 1)
                log_covar_u = gammaln(self.alpha_xu + self.vxu_sum + self.xu[u]) - gammaln(self.alpha_xu + self.vxu_sum) + (self.alpha_xu + self.vxu_sum) * np.log(self.beta_xu + np.array(self.m_u)) - (self.alpha_xu + self.vxu_sum + self.xu[u]) * np.log(self.beta_xu + np.array(self.m_u) + 1)
                log_covar_u_new = self.alpha_xu * (np.log(self.beta_xu) - np.log(self.beta_xu + 1))

            if self.xu_type == "cont":

                self.vxu_contvar = self.zu_u.T @ self.x_u
                self.xbar = self.vxu_contvar / self.m_u # mean(-)
                self.xval_cont_i = self.zu_u.T * self.x_u # Hadamard product to obtain in each row the elements of each separate cluster (i.e. cluster x 'values')
                self.xvar = self.__variance_unnorm(self.vxu_contvar, self.xval_cont_i, self.x_u, self.m_u) # computing self.xvar (no. clusters x 1)
                               
                self.x_u_plus = np.array(self.xu[:]) # not masking u
                self.vxu_contvar_plus = self.zu.T @ self.x_u_plus
                self.xbar_plus = self.vxu_contvar_plus / self.m_u_plus # mean(-)
                self.xval_cont_plus = self.zu.T * self.xu # Hadamard product to obtain in each row the elements of each separate cluster (i.e. cluster x 'values')
                self.xvar_plus = self.__variance_unnorm(self.vxu_contvar_plus, self.xval_cont_plus, self.x_u_plus, self.m_u_plus)

                self.kappa_nh = self.cont_par[1] + (self.m_u + 1)
                self.mu_nh = ((self.cont_par[1] * self.cont_par[0]) + ((self.m_u + 1) + self.xbar_plus)) / (self.kappa_nh)
                self.alpha_nh = self.cont_par[2] + (self.m_u + 1) / 2
                self.beta_nh = self.cont_par[3] + (0.5 * self.xvar_plus) + ((self.cont_par[1] * (self.m_u + 1) * (self.xbar_plus - self.cont_par[0])) / (2 * (self.cont_par[1] + (self.m_u + 1))))

                self.kappa_nhm = self.cont_par[1] + (self.m_u)
                self.mu_nhm = ((self.cont_par[1] * self.cont_par[0]) + ((self.m_u) + self.xbar)) / (self.kappa_nhm)
                self.alpha_nhm = self.cont_par[2] + (self.m_u) / 2
                self.beta_nhm = self.cont_par[3] + (0.5 * self.xvar) + ((self.cont_par[1] * self.m_u * (self.xbar - self.cont_par[0])) / (2 * (self.cont_par[1] + (self.m_u))))

                log_covar_u = gammaln(self.alpha_nh) - gammaln(self.alpha_nhm) + self.alpha_nhm * np.log(self.beta_nhm) - self.alpha_nh * np.log(self.beta_nh) + 0.5 * (np.log(self.kappa_nhm) - np.log(self.kappa_nh)) - (0.5 * np.log(2 * np.pi))
                
                self.kappa_nh_new = self.cont_par[1]
                self.mu_nh_new = (self.cont_par[1] * self.cont_par[0])
                self.alpha_nh_new = self.cont_par[2]
                self.beta_nh_new = self.cont_par[3]

                log_covar_u_new = - 0.5 * np.log(2 * np.pi)
            
            log_covar_u = np.append(log_covar_u,log_covar_u_new) # creating the final vector
        
        elif self.xu is not None and self.n_cov_u > 1:
            
            log_covar_u, log_covar_u_new = 0, 0 # initialize
            
            for cov1 in range(self.n_cov_u):

                self.x_u = np.array(self.xu[cov1, mask_u])

                if self.xu_type[cov1] == "categ":

                    self.X_u = np.array(self.XU[cov1][mask_u]) 
                    self.vxu = np.array(self.zu_u.T @ self.X_u) # rows of individuals with that cluster * columns of individuals with that covariate
                    index_u = int(self.xu[cov1][u])
                    log_covar_u += np.log((self.vxu[:,index_u] + self.alpha_xu[cov1][index_u]) / (np.array(self.m_u) + self.alpha0u[cov1])) # pick hyperparameter alpha_x proper of that particular covariate value c, i.e. alpha_x[x[v]]
                    log_covar_u_new += self.alpha_xu[cov1][index_u] * (np.log(self.beta_xu[cov1][index_u]) - np.log(self.beta_xu[cov1][index_u] + 1))

                if self.xu_type[cov1] == "count":

                    self.X_u = np.array(self.XU[cov1][mask_u]) 
                    self.vxu = np.array(self.zu_u.T @ self.X_u) # rows of individuals with that cluster * columns of individuals with that covariate

                    self.vxu_sum = self.vxu.sum(axis = 1)
                    log_covar_u += gammaln(self.alpha_xu[cov1] + self.vxu_sum + self.xu[cov1][u]) - gammaln(self.alpha_xu[cov1] + self.vxu_sum) + (self.alpha_xu[cov1] + self.vxu_sum) * np.log(self.beta_xu[cov1] + np.array(self.m_u)) - (self.alpha_xu[cov1] + self.vxu_sum + self.xu[cov1][u]) * np.log(self.beta_xu[cov1] + np.array(self.m_u) + 1)
                    log_covar_u_new += self.alpha_xu[cov1] * (np.log(self.beta_xu[cov1]) - np.log(self.beta_xu[cov1] + 1))


                if self.xu_type[cov1] == "cont":

                    self.vxu_contvar = self.zu_u.T @ self.x_u
                    self.xbar = self.vxu_contvar / self.m_u # mean(-)
                    self.xval_cont = self.zu_u.T * self.x_u # Hadamard product to obtain in each row the elements of each separate cluster (i.e. cluster x 'values')
                    self.xvar = self.__variance_unnorm(self.vxu_contvar, self.xval_cont, self.x_u, self.m_u) # computing self.xvar (no. clusters x 1)

                    self.x_u_plus = np.array(self.xu[cov1,:]) # not masking u
                    self.vxu_contvar_plus = self.zu.T @ self.x_u_plus
                    self.xbar_plus = self.vxu_contvar_plus / self.m_u_plus # mean(-)
                    self.xval_cont_plus = self.zu.T * self.x_u_plus # Hadamard product to obtain in each row the elements of each separate cluster (i.e. cluster x 'values')
                    self.xvar_plus = self.__variance_unnorm(self.vxu_contvar_plus, self.xval_cont_plus, self.x_u_plus, self.m_u_plus)

                    self.kappa_nh = self.cont_par[1] + (self.m_u + 1)
                    self.mu_nh = ((self.cont_par[1] * self.cont_par[0]) + ((self.m_u + 1) + self.xbar_plus)) / (self.kappa_nh)
                    self.alpha_nh = self.cont_par[2] + (self.m_u + 1) / 2
                    self.beta_nh = self.cont_par[3] + (0.5 * self.xvar_plus) + ((self.cont_par[1] * (self.m_u + 1) * (self.xbar_plus - self.cont_par[0])) / (2 * (self.cont_par[1] + (self.m_u + 1))))

                    self.kappa_nhm = self.cont_par[1] + (self.m_u)
                    self.mu_nhm = ((self.cont_par[1] * self.cont_par[0]) + ((self.m_u) + self.xbar)) / (self.kappa_nhm)
                    self.alpha_nhm = self.cont_par[2] + (self.m_u) / 2
                    self.beta_nhm = self.cont_par[3] + (0.5 * self.xvar) + ((self.cont_par[1] * self.m_u * (self.xbar - self.cont_par[0])) / (2 * (self.cont_par[1] + (self.m_u))))

                    log_covar_u += gammaln(self.alpha_nh) - gammaln(self.alpha_nhm) + self.alpha_nhm * np.log(self.beta_nhm) - self.alpha_nh * np.log(self.beta_nh) + 0.5 * (np.log(self.kappa_nhm) - np.log(self.kappa_nh)) - (0.5 * np.log(2 * np.pi))
                    
                    self.kappa_nh_new = self.cont_par[1]
                    self.mu_nh_new = (self.cont_par[1] * self.cont_par[0])
                    self.alpha_nh_new = self.cont_par[2]
                    self.beta_nh_new = self.cont_par[3]

                    log_covar_u_new += - 0.5 * np.log(2 * np.pi)

                    self.cov_diagn1 = log_covar_u
                    self.cov_diagn2 = log_covar_u_new


            log_covar_u = np.append(log_covar_u,log_covar_u_new) # creating the final vector

        else:
            log_covar_u = 0

        return log_covar_u

    def __cov_prob_i(self, mask_i, i):

        if self.xi is not None and self.n_cov_i == 1:
            # For every type of covariate:
            self.x_i = self.xi[mask_i]

            log_covar_i, log_covar_i_new = 0, 0 # initialize
            
            if self.xi_type == "categ":
                self.X_i = self.XI[mask_i]
                self.vxi = self.zi_i.T @ self.X_i
                log_covar_i = np.log((self.vxi[:,self.xi[i]] + self.alpha_xi[self.xi[i]]) / (np.array(self.m_i) + self.alpha0i)) # pick hyperparameter alpha_x proper of that particular covariate value c, i.e. alpha_x[x[v]]
                log_covar_i_new = np.log(self.alpha_xi[self.xi[i]] / self.alpha0i)

            if self.xi_type == "count":
                self.X_i = self.XI[mask_i]
                self.vxi = self.zi_i.T @ self.X_i # rows of individuals with that cluster * columns of individuals with that covariate
                self.vxi_sum = self.vxi.sum(axis = 1)
                log_covar_i = gammaln(self.alpha_xi + self.vxi_sum + self.xi[i]) - gammaln(self.alpha_xi + self.vxi_sum) + (self.alpha_xi + self.vxi_sum) * np.log(self.beta_xi + np.array(self.m_i)) - (self.alpha_xi + self.vxi_sum + self.xi[i]) * np.log(self.beta_xi + np.array(self.m_i) + 1)
                log_covar_i_new = self.alpha_xi * (np.log(self.beta_xi) - np.log(self.beta_xi + 1))

            if self.xi_type == "cont":

                self.vxi_contvar = self.zi_i.T @ self.x_i
                self.xbar_i = self.vxi_contvar / self.m_i # mean(-)
                self.xval_cont_i = self.zi_i.T * self.x_i # Hadamard product to obtain in each row the elements of each separate cluster (i.e. cluster x 'values')
                self.xvar_i = self.__variance_unnorm(self.vxi_contvar, self.xval_cont_i, self.x_i, self.m_i)

                self.x_i_plus = np.array(self.xi[:]) # not masking u
                self.vxi_contvar_plus = self.zi.T @ self.x_i_plus
                self.xbar_plus_i = self.vxi_contvar_plus / self.m_i_plus # mean(-)
                self.xval_cont_plus_i = self.zi.T * self.xi # Hadamard product to obtain in each row the elements of each separate cluster (i.e. cluster x 'values')
                self.xvar_plus_i = self.__variance_unnorm(self.vxi_contvar_plus, self.xval_cont_plus_i, self.x_i_plus, self.m_i_plus)

                self.kappa_nh = self.cont_par[1] + (self.m_i + 1)
                self.mu_nh = ((self.cont_par[1] * self.cont_par[0]) + ((self.m_i + 1) + self.xbar_plus_i)) / (self.kappa_nh)
                self.alpha_nh = self.cont_par[2] + (self.m_i + 1) / 2
                self.beta_nh = self.cont_par[3] + (0.5 * self.xvar_plus_i) + ((self.cont_par[1] * (self.m_i + 1) * (self.xbar_plus_i - self.cont_par[0])) / (2 * (self.cont_par[1] + (self.m_i + 1))))

                self.kappa_nhm = self.cont_par[1] + (self.m_i)
                self.mu_nhm = ((self.cont_par[1] * self.cont_par[0]) + ((self.m_i) + self.xbar_i)) / (self.kappa_nhm)
                self.alpha_nhm = self.cont_par[2] + (self.m_i) / 2
                self.beta_nhm = self.cont_par[3] + (0.5 * self.xvar_i) + ((self.cont_par[1] * self.m_i * (self.xbar_i - self.cont_par[0])) / (2 * (self.cont_par[1] + (self.m_i))))

                log_covar_i = gammaln(self.alpha_nh) - gammaln(self.alpha_nhm) + self.alpha_nhm * np.log(self.beta_nhm) - self.alpha_nh * np.log(self.beta_nh) + 0.5 * (np.log(self.kappa_nhm) - np.log(self.kappa_nh)) - (0.5 * np.log(2 * np.pi))

                self.kappa_nh_new = self.cont_par[1]
                self.mu_nh_new = (self.cont_par[1] * self.cont_par[0])
                self.alpha_nh_new = self.cont_par[2]
                self.beta_nh_new = self.cont_par[3]

                log_covar_i_new = - 0.5 * np.log(2 * np.pi)

            
            log_covar_i = np.append(log_covar_i,log_covar_i_new) # creating the final vector
        
        elif self.xi is not None and self.n_cov_i > 1:
            
            log_covar_i, log_covar_i_new = 0, 0 # initialize
            
            for cov2 in range(self.n_cov_i):

                self.x_i = np.array(self.xi[cov2, mask_i])

                if self.xi_type[cov2] == "categ":
                    self.X_i = np.array(self.XI[cov2][mask_i]) 
                    self.vxi = np.array(self.zi_i.T @ self.X_i)
                    index_i = int(self.xi[cov2][i])
                    log_covar_i += np.log((self.vxi[:,index_i] + self.alpha_xi[cov2][index_i]) / (np.array(self.m_i) + self.alpha0i[cov2])) # pick hyperparameter alpha_x proper of that particular covariate value c, i.e. alpha_x[x[v]]
                    log_covar_i_new += self.alpha_xi[cov2][index_i] * (np.log(self.beta_xi[cov2][index_i]) - np.log(self.beta_xi[cov2][index_i] + 1))
                
                if self.xi_type[cov2] == "count":

                    self.X_i = np.array(self.XI[cov2][mask_i]) 
                    self.vxi = np.array(self.zi_i.T @ self.X_i) # rows of individuals with that cluster * columns of individuals with that covariate

                    self.vxi_sum = self.vxi.sum(axis = 1)
                    log_covar_i += gammaln(self.alpha_xi[cov2] + self.vxi_sum + self.xi[cov2][i]) - gammaln(self.alpha_xi[cov2] + self.vxi_sum) + (self.alpha_xi[cov2] + self.vxi_sum) * np.log(self.beta_xi[cov2] + np.array(self.m_i)) - (self.alpha_xi[cov2] + self.vxi_sum + self.xi[cov2][i]) * np.log(self.beta_xi[cov2] + np.array(self.m_i) + 1)
                    log_covar_i_new += self.alpha_xi[cov2] * (np.log(self.beta_xi[cov2]) - np.log(self.beta_xi[cov2] + 1))


                if self.xi_type[cov2] == "cont":

                    self.vxi_contvar = self.zi_i.T @ self.x_i
                    self.xbar_i = self.vxi_contvar / self.m_i # mean(-)
                    self.xval_cont_i = self.zi_i.T * self.x_i # Hadamard product to obtain in each row the elements of each separate cluster (i.e. cluster x 'values')
                    self.xvar_i = self.__variance_unnorm(self.vxi_contvar, self.xval_cont_i, self.x_i, self.m_i)

                    self.x_i_plus = np.array(self.xi[cov2,:]) # not masking u
                    self.vxi_contvar_plus = self.zi.T @ self.x_i_plus
                    self.xbar_plus_i = self.vxi_contvar_plus / self.m_i_plus # mean(-)
                    self.xval_cont_plus_i = self.zi.T * self.x_i_plus # Hadamard product to obtain in each row the elements of each separate cluster (i.e. cluster x 'values')
                    self.xvar_plus_i = self.__variance_unnorm(self.vxi_contvar_plus, self.xval_cont_plus_i, self.x_i_plus, self.m_i_plus)

                    self.kappa_nh = self.cont_par[1] + (self.m_i + 1)
                    self.mu_nh = ((self.cont_par[1] * self.cont_par[0]) + ((self.m_i + 1) + self.xbar_plus_i)) / (self.kappa_nh)
                    self.alpha_nh = self.cont_par[2] + (self.m_i + 1) / 2
                    self.beta_nh = self.cont_par[3] + (0.5 * self.xvar_plus_i) + ((self.cont_par[1] * (self.m_i + 1) * (self.xbar_i - self.cont_par[0])) / (2 * (self.cont_par[1] + (self.m_i + 1))))

                    self.kappa_nhm = self.cont_par[1] + (self.m_i)
                    self.mu_nhm = ((self.cont_par[1] * self.cont_par[0]) + ((self.m_i) + self.xbar_i)) / (self.kappa_nhm)
                    self.alpha_nhm = self.cont_par[2] + (self.m_i) / 2
                    self.beta_nhm = self.cont_par[3] + (0.5 * self.xvar_i) + ((self.cont_par[1] * self.m_i * (self.xbar_i - self.cont_par[0])) / (2 * (self.cont_par[1] + (self.m_i))))

                    log_covar_i += gammaln(self.alpha_nh) - gammaln(self.alpha_nhm) + self.alpha_nhm * np.log(self.beta_nhm) - self.alpha_nh * np.log(self.beta_nh) + 0.5 * (np.log(self.kappa_nhm) - np.log(self.kappa_nh)) - (0.5 * np.log(2 * np.pi))
                    
                    self.kappa_nh_new = self.cont_par[1]
                    self.mu_nh_new = (self.cont_par[1] * self.cont_par[0])
                    self.alpha_nh_new = self.cont_par[2]
                    self.beta_nh_new = self.cont_par[3]

                    log_covar_i_new += - 0.5 * np.log(2 * np.pi)

                    self.cov_diagn1_i = log_covar_i
                    self.cov_diagn2_i = log_covar_i_new


            log_covar_i = np.append(log_covar_i,log_covar_i_new) # creating the final vector

        else:
            log_covar_i = 0

        return log_covar_i


    def __gibbs(self):

        if self.prior_u == "DM":
            def urn_u(m_u):
                return self.__urn_DM_u(m_u, self.beta, self.components)
        if self.prior_u == "DP":
            def urn_u(m_u):
                return self.__urn_DP_u(m_u, self.alpha_urn_u)
        if self.prior_u == "PY":
            def urn_u(m_u):
                return self.__urn_PY_u(m_u, self.alpha_urn_u, self.sigma)
        if self.prior_u == "GN":
            def urn_u(m_u):
                return self.__urn_GN_u(m_u, self.gamma)

        if self.prior_i == "DM":
            def urn_i(m_i):
                return self.__urn_DM_i(m_i, self.beta, self.components)
        if self.prior_i == "DP":
            def urn_i(m_i):
                return self.__urn_DP_i(m_i, self.alpha_urn_i)
        if self.prior_i == "PY":
            def urn_i(m_i):
                return self.__urn_PY_i(m_i, self.alpha_urn_i, self.sigma)
        if self.prior_i == "GN":
            def urn_i(m_i):
                return self.__urn_GN_i(m_i, self.gamma)      

        print("------------------\nGibbs Sampling simulation starts.")
        tic = time.clock()
 
        for it in range(self.its):
            for u in range(self.U):
                # STEP 1: Removing user u and sampling
                

                mask_u = self.indu[:u] + self.indu[u+1:]
                zu_minus = self.zu[mask_u]

                '''
                Deleting empty components, i.e. removing components which are empty after user u has been removed
                is removed from the network. Hence, we are removing:
                1. empty clusters;
                2. the cluster of u if it contained only u.
                '''
                if self.zu.shape[1] > 1: # if there is more than one cluster...
                    nonempty_u = [k for k in range(zu_minus.shape[1]) if np.sum(zu_minus, axis = 0)[k] > 0] # nonempty components indeces after removing user u
                    # nonempty = [2] # TESTing whether it creates a 2D array rather than 1D (we need 2D)
                    '''
                        -> N.B.: if u was not the only element of a cluster, its assignment is still there in zu!!! (This justifies h_u)
                    '''

                    self.zu = self.zu[:,nonempty_u] # actually removing all the empty components

                    if len(nonempty_u) == 1:
                        self.zu = np.array(self.zu)
                        
                    # z = np.array(z).T
                    self.zu_u = self.zu[mask_u,:] # cluster assignments without user u, after possibly removing u's empty cluster

                    # Note: s_full is not squared! dim=(clusters_u*clusters_i). Hence we only remove EMPTY ROWS (u clusters)
                    self.s_full = self.s_full[nonempty_u]  # Reducing ROW dimension of s_full matrix

                    '''
                    Note1: this operation already resets the indeces of the components.
                    Recall that we are retrieving the components up to permutations of indices.

                    Note2: zu will contain an EMPTY ROW, which is the raw of cluster assignment for user u,
                    which will be determined afterwards. zu_u removes this empty row of zu to perform matrix operations.
                    zu.shape    == (users,   nonempty clusters for users after removing u)
                    zu_u.shape  == (users-1, nonempty clusters for users after removing u)

                    '''
                self.H = self.zu.shape[1] # *occupied* users clusters (can be less than starting value)
                self.K = self.zi.shape[1] # items clusters

                '''
                See McDaid p.40: "...posterior, now that we have *observed* edges with total weight ykl between pkl pairs of nodes (pair of nodes = all possible edges)".
                '''

                r_u = self.zi.T @ self.Y[u] # sum of interactions of user u to each item cluster
                R_u = np.matlib.repmat(r_u,self.H,1) # We use the same row of links for each prob. computation. We use it once for each prob. = h


                '''
                Computing probabilities
                '''

                self.__count_possible_edges(self.zu_u,self.zi) 
                self.n_cardinality_plus, self.m_u_plus, self.m_i_plus = self.__count_possible_edges_return(self.zu,self.zi)
                
                M_i = np.matlib.repmat(self.m_i,self.H,1)

                log_prob_int = np.array(np.sum(gammaln(self.a + self.s_full + R_u) - gammaln(self.a + self.s_full) + (self.a + self.s_full) * np.log(self.b + self.n_cardinality) - (self.a + self.s_full + R_u) * np.log(self.b + self.n_cardinality + M_i), axis = 1)) 

                log_prob_int_new = np.sum(gammaln(self.a + r_u) - gammaln(self.a) + self.a * np.log(self.b) - (self.a + r_u) * np.log(self.b + self.m_i))
                log_prob_int = np.append(log_prob_int,log_prob_int_new)

                '''
                vx matrix : on the columns, we have the covariates; on the rows, we have the clusters of users (since we inverted zu_u)
                We are counting how many x's there are in each cluster of users.
                
                vx: picking the column relative to covariate x[u] to see how it is distributed across clusters.
                '''


                log_covar_u = self.__cov_prob(mask_u, u)
                
                log_prob = np.log(np.array(urn_u(self.m_u))) + log_prob_int + log_covar_u

                
                prob_unnormalized = np.exp(log_prob - np.max(log_prob)) 

                prob = prob_unnormalized/np.sum(prob_unnormalized) 

                
                '''Pick randomly a partition:'''
                zu_sampled = np.random.choice(a = range(self.H+1), p = prob) # range(H+1) samples from 0 to H value

                h_u = np.where(self.zu[u,:]==1)[0] # cluster assignment of the node v at stake. 
                                            # If cluster h was not empty, then a number shows up.
                                            # otherwise, the cluster has been already removed.

                if len(h_u) == 1: # Cleaning the row before new assignment.
                    correct = np.zeros(shape=(self.H,self.K))
                    correct[h_u,:] = r_u
                    self.s_full = self.s_full - correct
                    self.zu[u,h_u] = 0

                if zu_sampled == self.H: # ex.: H = 3 but index [0,2]. Hence, we do not increase H!
                    zu_new = np.zeros((self.zu.shape[0],self.zu.shape[1]+1))
                    zu_new[:,:-1] = self.zu
                    self.zu = zu_new
                    self.zu[u,zu_sampled] = 1

                    # n_full_new = np.zeros((n_full.shape[0]+1,n_full.shape[1])) # (H+1 x K)
                    s_full_new = np.zeros((self.s_full.shape[0]+1,self.s_full.shape[1])) # (H+1 x K)
                    # n_full_new[:-1,:] = n_full
                    s_full_new[:-1,:] = self.s_full
                    # n_full = n_full_new
                    self.s_full = s_full_new

                    # The interactions with items for this "new" user, trivially, has not changed! No update of r_u.
                    # We will update it after the update of item clusters.

                    self.H += 1
                else:
                    self.zu[u,zu_sampled] = 1
                        

                # Including the interactions of the "new" user u into the n_full and s_full matrices:
            
                # inter_n = np.zeros(shape = (H,K))
                inter_s_u = np.zeros(shape = (self.H,self.K))
                inter_s_u[zu_sampled,:] = r_u # update solo della riga perché network bipartito
                self.s_full = self.s_full + inter_s_u
                del inter_s_u # superfluo in teoria

                s_full1 = self.zu.T @ self.Y @ self.zi # superfluo, ma per assicurarsi che si usi la matrice corretta
                # sfull1 è uguale a s_full

                self.__count_possible_edges(self.zu,self.zi) # Computation of m_u, m_i, n_cardinality as object attributes.


                self.zu_labels = self.zu @ range(self.zu.shape[1])
                # print(f"User {u} updated.")
            

            for i in range(self.I):
                '''
                STEP 2: Removing user i and sampling
                '''
                mask_i = self.indi[:i] + self.indi[i+1:]
                zi_minus = self.zi[mask_i]

                '''
                Deleting empty components, i.e. removing components which are empty after user u has been removed
                is removed from the network. Hence, we are removing:
                1. empty clusters;
                2. the cluster of u if it contained only u.
                '''
                if self.zi.shape[1] > 1: # if there is more than one cluster...
                    nonempty_i = [k for k in range(zi_minus.shape[1]) if np.sum(zi_minus, axis = 0)[k] > 0] # nonempty components indeces after removing item i

                    self.zi = self.zi[:,nonempty_i] # actually removing all the empty components

                    if len(nonempty_i) == 1:
                        self.zi = np.array(self.zi)
                        
                    # z = np.array(z).T
                    self.zi_i = self.zi[mask_i,:] # cluster assignments without item i, after possibly removing i's empty cluster

                    # Note: s_full is not squared! dim=(clusters_u*clusters_i). Hence we only remove EMPTY ROWS (u clusters)
                    self.s_full = self.s_full[:,nonempty_i]  # Reducing *COLUMN* dimension of s_full matrix

                    '''
                    Note1: this operation already resets the indeces of the components.
                    Recall that we are retrieving the components up to permutations of indices.

                    Note2: zu will contain an EMPTY ROW, which is the raw of cluster assignment for user u,
                    which will be determined afterwards. zu_u removes this empty row of zu to perform matrix operations.
                    zu.shape    == (users,   nonempty clusters for users after removing u)
                    zu_u.shape  == (users-1, nonempty clusters for users after removing u)

                    '''
                self.H = self.zu.shape[1] # users clusters 
                self.K = self.zi.shape[1] # *occupied* items clusters (can be less than starting value)


                r_i = self.zu.T @ self.Y[:,i] # sum of interactions of item i to each users cluster
                R_i = np.matlib.repmat(r_i,self.K,1) # We use the same row of links for each prob. computation. We use it once for each prob. = k

                self.__count_possible_edges(self.zu,self.zi_i) # ORA M_I ETC. SONO CALCOLATI SULLA MATRICE "BUCATA"
                self.n_cardinality_plus, self.m_u_plus, self.m_i_plus = self.__count_possible_edges_return(self.zu,self.zi)

                M_u = np.matlib.repmat(self.m_u,self.K,1)

                '''
                Computing probabilities
                '''

                log_prob_int = np.array(np.sum(gammaln(self.a + self.s_full.T + R_i) - gammaln(self.a + self.s_full.T) + (self.a + self.s_full.T) * np.log(self.b + self.n_cardinality.T) - (self.a + self.s_full.T + R_i) * np.log(self.b + self.n_cardinality.T + M_u), axis = 1)) 

                log_prob_int_new = np.sum(gammaln(self.a + r_i) - gammaln(self.a) + self.a * np.log(self.a) - (self.a + r_i) * np.log(self.b + self.m_u))
                log_prob_int = np.append(log_prob_int,log_prob_int_new)
                            

                log_covar_i = self.__cov_prob_i(mask_i,i)

                log_prob = np.log(np.array(urn_i(self.m_i))) + log_prob_int + log_covar_i

                prob_unnormalized = np.exp(log_prob - np.max(log_prob)) 

                prob = prob_unnormalized/np.sum(prob_unnormalized) 

                '''Pick randomly a partition:'''
                zi_sampled = np.random.choice(a = range(self.K+1), p = prob) # range(K+1) samples from 0 to K value

                h_i = np.where(self.zi[i,:]==1)[0] 

                if len(h_i) == 1: # Cleaning the row before new assignment.
                    correct = np.zeros(shape=(self.H,self.K))
                    correct[:,h_i] = np.matrix(r_i).T
                    self.s_full = self.s_full - correct
                    self.zi[i,h_i] = 0

                if zi_sampled == self.K: # ex.: H = 3 but index [0,2]. Hence, we do not increase H!
                    zi_new = np.zeros((self.zi.shape[0],self.zi.shape[1]+1))
                    zi_new[:,:-1] = self.zi
                    self.zi = zi_new
                    self.zi[i,zi_sampled] = 1

                    s_full_new = np.zeros((self.s_full.shape[0],self.s_full.shape[1]+1)) # (H x K+1)
                    s_full_new[:,:-1] = self.s_full
                    self.s_full = s_full_new

                    # The interactions with items for this "new" user, trivially, has not changed! No update of r_u.
                    # We will update it after the update of item clusters.
                    self.K += 1
                else:
                    self.zi[i,zi_sampled] = 1
                
                '''
                Including the interactions of the "new" item i into the n_full and s_full matrices:
                '''
                inter_s_i = np.zeros(shape = (self.H,self.K))
                inter_s_i[:,zi_sampled] = r_i # update solo della colonna perché network bipartito
                self.s_full = self.s_full + inter_s_i

                del inter_s_i # superfluo in teoria

                s_full1 = self.zu.T @ self.Y @ self.zi # superfluo, ma per assicurarsi che si usi la matrice corretta
                # Ancora una volta, s_full1 è uguale a s_full

                self.zi_labels = self.zi @ range(self.zi.shape[1])
                self.zu_labels = self.zu @ range(self.zu.shape[1])
                
                # print(f"Item {i} updated.")

            self.__count_possible_edges(self.zu,self.zi)
            likel = self.__marginal_loglik_bipartite(self.Y, self.zu_labels, self.zi_labels, self.a, self.b, self.s_full, self.n_cardinality)
            self.ll.append(likel)

            
            if self.verbose == True:
                print(f"Iteration {it} complete. Log-likelihood: {likel}.")
            elif it % 10 == 0:
                print(f"Iteration {it} complete. Log-likelihood: {likel}.")

        toc = time.clock()
        print(f"Runtime: {toc-tic}")

    def __check_prior_parameters_u(self):
        if self.prior_u == "DM":
            if self.components is None or self.components < 1:
                raise ValueError(f"Invalid value for 'components': {self.components}. "
                                  "Dirichlet-Multinomial prior requires at least one component.")
            if self.beta is None:
                raise ValueError(f"Invalid value for 'beta': {self.beta}. "
                                  "Dirichlet-Multinomial prior requires the specification of beta.")

        if self.prior_u == "DP" or self.prior_u == "PY":
            if self.alpha_urn_u < 0:
                raise ValueError(f"Invalid value for 'alpha_urn_u': {self.alpha_urn_u}. "
                                  "For the Dirichlet Process prior, it must be positive.")

        if self.prior_u == "PY":
            if self.sigma is None:
                raise ValueError(f"Pitman-Yor prior requires the specification of 'sigma'")
            if self.sigma < 0.:
                raise ValueError(f"Invalid value for 'sigma': {self.sigma}. "
                                  "For the Pitman-Yor prior, it must be positive.")

        if self.prior_u == "GN":
            if self.gamma is None:
                raise ValueError(f"Gnedin process prior requires the specification of 'gamma'")
            if self.gamma < 0.:
                raise ValueError(f"Invalid value for 'gamma': {self.gamma}. "
                                  "For the Gnedin process prior, it must be positive.")

    def __check_prior_parameters_i(self):
        if self.prior_i == "DM":
            if self.components is None or self.components < 1:
                raise ValueError(f"Invalid value for 'components': {self.components}. "
                                  "Dirichlet-Multinomial prior requires at least one component.")
            if self.beta is None:
                raise ValueError(f"Invalid value for 'beta': {self.beta}. "
                                  "Dirichlet-Multinomial prior requires the specification of beta.")

        if self.prior_i == "DP" or self.prior_i == "PY":
            if self.alpha_urn_i < 0:
                raise ValueError(f"Invalid value for 'alpha_urn_u': {self.alpha_urn_u}. "
                                  "For the Dirichlet Process prior, it must be positive.")

        if self.prior_i == "PY":
            if self.sigma is None:
                raise ValueError(f"Pitman-Yor prior requires the specification of 'sigma'")
            if self.sigma < 0.:
                raise ValueError(f"Invalid value for 'sigma': {self.sigma}. "
                                  "For the Pitman-Yor prior, it must be positive.")

        if self.prior_i == "GN":
            if self.gamma is None:
                raise ValueError(f"Gnedin process prior requires the specification of 'gamma'")
            if self.gamma < 0.:
                raise ValueError(f"Invalid value for 'gamma': {self.gamma}. "
                                  "For the Gnedin process prior, it must be positive.")

    def __check_covariates(self):
        """
        Check the input data xu and xi and their perparameters.

        Parameters
        ----------
        xu : array-like, shape (n_features, n_samples)
        xi : array-like, shape (n_features, n_samples)
        """

        if self.xu is not None and self.n_cov_u > 1:
            if self.xu.shape[1] != self.Y.shape[0]:
                raise ValueError(f"Invalid dimension of 'xu'. 'xu' should be an array of dimension ({self.n_cov_u,Y.shape[0]}) instead of ({self.n_cov_u,self.xu.shape[1]}).")

            if len(self.xu_type) != self.n_cov_u:
                raise ValueError(f"Invalid dimension of 'xu_type'. 'xu_type' should be a list of dimension {self.n_cov_u} instead of {len(self.xu_type)}.")
        
        if self.xi is not None and self.n_cov_i > 1:
            if self.xi.shape[1] != self.Y.shape[1]:
                raise ValueError(f"Invalid dimension of 'xi'. 'xi' should be an array of dimension ({self.n_cov_i,self.Y.shape[1]}) instead of ({self.n_cov_i,self.xi.shape[1]}).")
            
            if len(self.xi_type) != self.n_cov_i:
                raise ValueError(f"Invalid dimension of 'xi_type'. 'xi_type' should be a list of dimension {self.n_cov_i} instead of {len(self.xi_type)}.")


        if self.xu is not None and self.n_cov_u == 1:
            if self.xu.shape[0] != self.Y.shape[0]:
                raise ValueError(f"Invalid dimension of 'xu'. 'xu' should be an array of dimension ({self.n_cov_u,Y.shape[0]}) instead of ({self.n_cov_u,self.xu.shape[0]}).")

        if self.xi is not None and self.n_cov_i == 1:
            if self.xi.shape[0] != self.Y.shape[1]:
                raise ValueError(f"Invalid dimension of 'xu'. 'xu' should be an array of dimension ({self.n_cov_i,self.Y.shape[1]}) instead of ({self.n_cov_i,self.xi.shape[0]}).")


    # [KeyError WIP]: 
        #if self.xu is not None and self.n_cov_u == 1:
        #    if len(self.xu_type) > 1:
        #        raise KeyError(f"Invalid dimension of 'xu_type'. If one covariate is provided, then the dimension must be 1 instead of {len(self.xu_type)}.")
        
        #if self.xi is not None and self.n_cov_i == 1:
        #    if len(self.xi_type) > 1:
        #        raise KeyError(f"Invalid dimension of 'xi_type'. If one covariate is provided, then the dimension must be 1 instead of {len(self.xi_type)}.")


    def __compute_block_interactions(self):
        self.__count_possible_edges(self.zu, self.zi)
        self.theta_est = (self.a + self.s_full) / (self.b + self.n_cardinality)
        if self.theta_est is not None:
            print("Block-interactions computed.")

    def __howmany_covariates(self):
        if self.xu is not None and len(self.xu.shape) > 1:
            self.n_cov_u = self.xu.shape[0]
        elif self.xu is None:
            self.n_cov_u = 0
        else:
            self.n_cov_u = 1

        if self.xi is not None and len(self.xi.shape) > 1:
            self.n_cov_i = self.xi.shape[0]
        elif self.xi is None:
            self.n_cov_i = 0
        else:
            self.n_cov_i = 1

    def __variance_unnorm(self, vx_contvar, xval_cont, x, m_):
        xvar = np.zeros(shape = (vx_contvar.shape))
        for cluster in range(vx_contvar.shape[0]):
            values_temp_u = np.array([xval_cont[cluster][e] for e in range(xval_cont[cluster].shape[0]) if xval_cont[cluster][e] != 0])
            if values_temp_u.sum() != 0:
                xvar[cluster] = values_temp_u.var() * m_[cluster]
        return xvar        
        
# %%
