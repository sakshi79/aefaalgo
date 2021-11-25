# Importing required dependencies
import random
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt


class aefa:

           

    def benchmark_range(self, func_num):
        '''
        This function gives boundaries and dimension of search space for test functions.
    
        If lower bounds of dimensions are the same, then 'lb' is a value.
        Otherwise, 'lb' is a vector that shows the lower bound of each dimension.
        This is also true for upper bounds of dimensions.
    
        Insert your own boundaries with a new func_num.
        '''
        lb = -100
        ub = 100
        D = 1
    
    
        if func_num == 1:
            lb=-100
            ub=100
            D=30
    
        if func_num == 2:
            lb=-10
            ub=10
            D=30
        
        if func_num == 3:
            lb=-100
            ub=100
            D=30
        
        if func_num == 4:
            lb=-100
            ub=100
            D=30
        
        if func_num == 5:
            lb=-30
            ub=30
            D=30
        
        if func_num == 6:
            lb=-100
            ub=100
            D=30
        
        if func_num == 7:
            lb=-1.28
            ub=1.28
            D=30
        
        if func_num == 8:
            lb=-500
            ub=500
            D=30
        
        if func_num == 9:
            lb=-5.12
            ub=5.12
            D=30
        
        if func_num == 10:
            lb=-32
            ub=32
            D=30
        
        if func_num == 11:
            lb=-600
            ub=600
            D=30
        
        if func_num == 12:
            lb=-50
            ub=50
            D=30
        
        if func_num == 13:
            lb=-50
            ub=50
            D=30
        
        if func_num == 14:
            lb=-65.536
            ub=65.536
            D=25
    
        if func_num == 15:
            lb=-5
            ub=5
            D=4
    
        if func_num == 16:
            lb=-5
            ub=5
            D=2
    
        if func_num == 17:
            lb=-5
            ub=5
            D=2
        
        if func_num == 18:
            lb=-2
            ub=2
            D=2
        
        if func_num == 19:
            lb=0
            ub=1
            D=3
        
        if func_num == 20:
            lb=0
            ub=1
            D=6
        
        if func_num == 21:
            lb=0
            ub=10
            D=4
        
        if func_num == 22:
            lb=0
            ub=10
            D=4
        
        if func_num == 23:
            lb=0
            ub=10
            D=4
        
        return lb, ub, D


    def Ufun(self, x,a,k,m):

        y = k*((x-a)**m)*(x>a) + k*((-x-a)**m)*(x<(-a))

        return y 

    def benchmark(self, X, func_num, D):
        '''
        This function defines different fitness functions for a user to choose from.
        You may add your own function to this list.
        '''
        fit=0
    
        if func_num==1:
            fit = np.sum(np.square(X))
        
        if func_num==2:
            fit = np.sum(np.absolute(X)) + np.prod(np.absolute(X))
        
        if func_num==3:
            for i in range(D+1):
                fit += (np.sum(X[:i]))**2
            
        if func_num==4:
            fit = np.amax(np.absolute(X))
        
        if func_num==5:
            fit = np.sum(np.square(100*(X[1:D] - np.square(X[0:D-1]))) + np.square(X[0:D-1]-1))
        
        if func_num==6:
            fit = np.sum(np.square(np.floor(X+0.5)))
        
        if func_num==7:
            fit = np.sum(X[:]*(X**4)) + random.random()
        
        if func_num==8:
            fit = np.sum(-X*(np.sin(np.absolute(X)**(0.5))))
        
        if func_num==9:
            fit = np.sum(X**2-10*np.cos(2*np.pi*X))+10*D
        
        if func_num==10:
            fit = -20*np.exp(-0.2*((np.sum(X**2)/D)**0.5))-np.exp(np.sum(np.cos(2*np.pi*X)/D))+20+np.exp(1)
        
        if func_num==11:
            fit = np.sum(X**2)/4000 - np.prod(np.cos(X/np.sqrt(np.arange(1,11).reshape(-1,1))))+1
        
        if func_num==12:
            fit = (np.pi/D)*(10*((np.sin(np.pi*(1+(X[1]+1)/4)))**2)
                + np.sum((((X[0:D-1]+1)/4)**2)*(1+10*((np.sin(np.pi*(1+(X[1:D]+1)/4)))**2)))
                + ((X[D-1]+1)/4)**2) + np.sum(self.Ufun(X,10,100,4))
        
        if func_num==13:
            # Note that the following code snippet can only work for N>D
            fit = 0.1*((np.sin(3*np.pi*X[0])))**2 + np.sum((X[0:D-1]-1)**2*(1+(np.sin(3*np.pi*X[1:D]))**2))+np.sum(self.Ufun(X,5,100,4))
        
        if func_num==14:
            aS=np.array([[-32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32],
                        [-32, -32, -32, -32, -32, -16, -16, -16, -16, -16, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32]])
            bS = np.zeros((2,25))
            for j in range(25):
                bS[:,j]=np.sum((X[j]-aS[:,j])**6)
            fit = (1/500 + np.sum(1./(np.arange(25)+bS)))**(-1)
        
        if func_num==15:
            aK = np.array([.1957, .1947, .1735, .16, .0844, .0627, .0456, .0342, .0323, .0235, .0246])
            bK = np.array([.25, .5, 1, 2, 4, 6, 8, 10, 12, 14, 16])
            bK = 1. / bK
            fit = np.sum((aK-((X[0]*(bK**2+X[1]*bK))/(bK**2+X[2]*bK+X[3])))**2)
        
        if func_num==16:
            fit = 4*(X[0]**2) - 2.1*(X[0]**4) + (X[0]**6)/3+X[0]*X[1]-4*(X[1]**2)+4*(X[1]**4)
            
        if func_num==17:
            fit = (X[1]-(X[0]**2)*5.1/(4*(np.pi**2))+5/np.pi*X[0]-6)**2+10*(1-1/(8*np.pi))*np.cos(X[0])+10
            
        if func_num==18:
            fit = (1+(X[0]+X[1]+1)**2*(19-14*X[0]+3*(X[0]**2)-14*X[1]+6*X[0]*X[1]+3*(X[1]**2)))*(30+(2*X[0]-3*X[1])**2*(18-32*X[0]+12*(X[0]**2)+48*X[1]-36*X[0]*X[1]+27*(X[1]**2)))
        
        if func_num==19:
            aH=np.array([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]])
            cH = np.array([1, 1.2, 3, 3.2])
            pH = np.array([[.3689, .117, .2673], [.4699, .4387, .747], [.1091, .8732, .5547], [.03815, .5743, .8828]])
            fit=0
            for i in range(4):
                fit = fit - cH[i]*np.exp(-np.sum(aH[i,:]*(X-pH[i,:]**2)))
                
        if func_num==20:
            aH=np.array([[10, 3, 17, 3.5, 1.7, 8], [.05, 10, 17, .1, 8, 14], [3, 3.5, 1.7, 10, 17, 8], [17, 8, .05, 10, .1, 14]])
            cH=np.array([1, 1.2, 3, 3.2])
            pH=np.array([[.1312, .1696, .5569, .0124, .8283, .5886], [.2329, .4135, .8307, .3736, .1004, .9991],
                         [.2348, .1415, .3522, .2883, .3047, .6650], [.4047, .8828, .8732, .5743, .1091, .0381]])
            fit=0
            for i in range(4):
                fit = fit - cH[i]*np.exp(-np.sum(aH[i,:]*(X-pH[i,:]**2)))
                
        aSH=np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7], [2, 9, 2, 9], [5, 5, 3, 3],
                         [8, 1, 8, 1], [6, 2, 6, 2], [7, 3.6, 7, 3.6]])
        cSH=np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
            
        if func_num==21:
            fit=0
            for i in range(5):
                fit = fit - (np.matmul((X-aSH[i,:]), np.transpose(X-aSH[i,:])) + cSH[i])**(-1)
                    
        if func_num==22:
            fit=0
            for i in range(7):
                fit = fit - (np.matmul((X-aSH[i,:]), np.transpose(X-aSH[i,:])) + cSH[i])**(-1)
                    
        if func_num==23:
            fit=0
            for i in range(10):
                fit = fit - (np.matmul((X-aSH[i,:]), np.transpose(X-aSH[i,:])) + cSH[i])**(-1)
                
            
        return fit

    
    def optimize(self, N, max_iter, func_num, tag=0, Rpower=1, FCheck=True, show_plot=False):
        '''
        Returns optimized fitness value and its corresponding coordinates in the search space.

        Keyword arguments:
        N: number of particles in search space

        max_iter: number of iterations

        func_num: Specifies the function to be optimized

        Optional Keyword Arguments: 
        tag: specifies whether we want maxima or minima.
        0 by default for maximization. Specify tag=1 for minimization.

        Rpower: exponent for the normalized distance between the particles.
        Default value 1

        FCheck: This factor ensures that only 2-6% charges apply force to others in the last iterations.
        Set to True by default. 

        show_plot: True if you want to visualize convergence to the optimum, False otherwise and default.
        
        '''
    
        RNorm = 2
        R = np.zeros(N)
    
        #Dimension and lower and upper bounds of the variables
        (lb,ub,D)=self.benchmark_range(func_num)
        Fbest = 0
        Lbest = np.zeros([N, D])
    
        # Random initialization of charge population 
        X = np.random.rand(N,D)*(ub-lb)+lb
    
        #create the best so far chart and average fitnesses chart.
        BestValues = []
        MeanValues = []
    
        V = np.zeros([N,D])
    
    
        for iteration in range(1,max_iter+1):
        
            #Evaluating fitnesses of charged particles
            fitness = np.zeros(N)
            for i in range(N):
                fitness[i] = self.benchmark(X[i,:],func_num,D)
        
            if tag == 1:
                #Minimization
                best = np.amin(fitness)
                best_X = np.argmin(fitness)
    
            else:
                #Maximization
                best = np.amax(fitness)
                best_X = np.argmax(fitness)
        
            if iteration == 1:
                Fbest = best #Best fitness value
                Lbest = X[best_X,:] #Parameters corresponding to best fitness values
            
            else:
                if tag == 1:
                    #Minimization
                    if best < Fbest:
                        Fbest = best
                        Lbest = X[best_X,:]
                else:
                    #Maximization
                    if best > Fbest:
                        Fbest = best
                        Lbest = X[best_X,:] 

            
            if show_plot == True:
                swarm = np.zeros([N,2])
                swarm[:,0] = X[:,0]
                swarm[:,1] = X[:,1]
                plt.clf()
                plt.plot(swarm[:,0], swarm[:,1], 'o')
                plt.plot(swarm[best_X,0], swarm[best_X,1], '*')
                plt.title("Iteration %i" % iteration)
                plt.show(block=False)
                plt.pause(0.05)
                  
            
            BestValues.append(Fbest)
            MeanValues.append(np.mean(fitness))
        
        
            # Charge
            Fmax = np.max(fitness)
            Fmin = np.min(fitness)
            Fmean = np.mean(fitness)
        
            Q = np.zeros(N)
            #M = np.zeros(N)
        
            if Fmax == Fmin:
                Q = np.ones(N)
                #M = np.ones(N)
            
            else:
                if tag == 1:
                    #Minimization
                    best = Fmin
                    worst = Fmax
                else:
                    #Maximization
                    best = Fmax
                    worst = Fmin
            
                for i in range(N):
                    Q[i] = np.exp((fitness[i]-worst)/(best-worst))
                
            Q = Q/np.sum(Q)
        
            # Total electric field calculation
            fper = 3
            if FCheck == True:
                cbest = fper + (1-iteration/max_iter)*(100-fper)
                cbest = round(N*cbest/100)
            else:
                cbest = N
        
            # Sorting charges from maximum to minimun
            Qs = np.sort(Q)   # sorted in ascending order
            Qs = np.flip(Qs)  # sorted in descending order
        
            s = np.argsort(Q)  # sorted array of indices (A.O.)
            s = s[::-1]        # sorted array of indices (D.O.)
        
            # Electric field
            E = np.zeros([N,D])
        
            for i in range(N):
                for ii in range(cbest):
                    j = s[ii]
                    if j != i:
                        R = LA.norm((X[i,:]-X[j,:]), RNorm)
                        R = R**Rpower
                        for k in range(D):
                            E[i,k] = E[i,k] + (random.random()) * (Q[j]) * ( (X[j,k]-X[i,k]) / (R+np.finfo(float).eps) )
                    
                
                    
        
            # Calculation of coulomb constant
            self.alpha = 30
            K0 = 500
            K = K0*np.exp(-self.alpha*iteration/max_iter)
        
            #Calculation of acceleration
            a = E*K
        
            #Charge movement
            V = np.random.rand(N,D)*V + a
            X = X + V
            X = np.maximum(X,lb)
            X = np.minimum(X,ub)

            

        plt.close()                      
        return Fbest, Lbest
        





             
            
           



