'''QUESTION 1'''
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
import cvxpy as cvx
from sklearn.model_selection import KFold
import sklearn.metrics as skmet
import scipy as sci
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import gurobipy
import time
from functools import reduce


class classifier():
    '''Contains functions for prototype selection'''

    def __init__(self, X, y, epsilon_, lambda_):
        self.X = X
        self.y = y
        self.epsilon_ = epsilon_
        self.lambda_ = lambda_

    def intersect(self, leftSet, rightSet):
        aset = set([tuple(x) for x in leftSet])
        bset = set([tuple(x) for x in rightSet])
        return (np.array([x for x in aset & bset]))

    def DM(self, array1, array2):
        return (sci.spatial.distance_matrix(array1, array2))

    def B_xj(self, array, j):
        distanceMatrix = self.DM(array, array)
        coveredPointsInd, = np.where(distanceMatrix[j] <= self.epsilon_)
        return (coveredPointsInd)

    def X_Xl(self, label_array, array, label):
        ind = np.where(label_array != label)
        return (array[ind])

    def card(self, array):
        return (len(array))

    def train_lp(self, verbose=False):
        '''initialization'''
        totalNumOfPoints = self.card(self.X)
        self.optimalValue = 0
        self.Prototype = []

        '''solve relaxed LP'''
        for label in np.unique(self.y):

            nPointsInLabel = self.card(self.X[np.where(self.y == label)])

            ## set variables
            if verbose:
                print("set var")
            start = time.time()
            alpha = cvx.Variable(totalNumOfPoints)
            xi = cvx.Variable(nPointsInLabel)
            if verbose:
                print("set var finished")
                print("time: ", time.time() - start)

            ## objective function
            if verbose:
                print("obj set")
            start = time.time()

            obj = 0

            notInLabel = self.X_Xl(self.y, self.X, label)
            for i in range(totalNumOfPoints):
                obj1 = self.lambda_ + self.card(self.intersect(self.X[self.B_xj(self.X, i)], notInLabel))
                obj2 = obj1 * alpha[i]
                obj += obj2

            sum_ = reduce((lambda x, y: x + y), xi)
            obj += sum_

            obj = cvx.Minimize(obj)

            if verbose:
                print("obj set finished")
                print("time: ", time.time() - start)

            ## constraints

            if verbose:
                print("const set")
            start = time.time()

            constraints = [alpha <= 1, alpha >= 0, xi >= 0]
            cons = []
            distanceMatrix = self.DM(self.X, self.X)
            for count, j in enumerate(np.array(np.where(self.y == label)).tolist()[0]):
                temp = distanceMatrix[j] <= self.epsilon_
                xtemp = temp.astype(int)
                cons += [sum(cvx.mul_elemwise(xtemp, alpha)) >= 1 - xi[count]]

            constraints += cons

            if verbose:
                print("const set finished")
                print("time: ", time.time() - start)

            ## problem
            if verbose:
                print("prob set")
            start = time.time()

            prob = cvx.Problem(obj, constraints)
            if verbose:
                print("prob set finished")
                print("time: ", time.time() - start)

            ## solve
            if verbose:
                print("solving")
            start = time.time()
            optimal = prob.solve(solver=cvx.GUROBI, verbose=False)
            if verbose:
                print("solved")
                print("time: ", time.time() - start)

            ## verbose = True
            if (verbose):
                print("status:", prob.status)
                print("optimal value", optimal)

            ''' randomize rounding'''
            if verbose:
                print("rounding started...")
            start = time.time()

            distanceMatrix = self.DM(self.X, self.X)
            objective = 2 * np.log(nPointsInLabel) * optimal + 100

            ## initialize A and S
            self.A = [0] * totalNumOfPoints
            self.S = [0] * nPointsInLabel
            objective = 0  # objective for ILP (approx.)

            for t in range(2 * int(np.floor(np.log(nPointsInLabel)))):

                # Af = []
                # Sf = []
                # for j in range(totalNumOfPoints):
                #    prob = alpha.value[j]
                #    prob = int(prob * 1000000)/1000000
                #    Af.append(sci.stats.bernoulli.rvs( abs(prob) ))    # draw
                # self.A = list(map(max, self.A, Af))                             # update

                flat = (alpha.value).tolist()
                flat_list = [item for sublist in flat for item in sublist]
                a = abs(np.array(list(map(int, (np.array(flat_list) * 1000000).tolist()))) / 1000000).tolist()
                Af = sci.stats.bernoulli.rvs(a).tolist()
                self.A = list(map(max, self.A, Af))

                # for i in range(nPointsInLabel):
                #    prob2 = xi.value[i]
                #    prob2 = int(prob2 * 1000000)/1000000
                #    Sf.append(sci.stats.bernoulli.rvs( abs(prob2) ))   # draw
                # self.S = list(map(max, self.S, Sf))                             # update

                flat2 = (xi.value).tolist()
                flat_list2 = [item for sublist in flat2 for item in sublist]
                b = abs(np.array(list(map(int, (np.array(flat_list2) * 1000000).tolist()))) / 1000000).tolist()
                Sf = sci.stats.bernoulli.rvs(b).tolist()
                self.S = list(map(max, self.S, Sf))

                feasible = True

                for count, jj in enumerate(np.array(np.where(self.y == label)).tolist()[0]):
                    temp = distanceMatrix[jj] <= self.epsilon_
                    xtemp = temp.astype(int)
                    if (np.dot(xtemp, self.A) < (1 - self.S[count])):
                        feasible = False
                        continue

                if (feasible):
                    print("if feasible")
                    st = time.time()
                    obj = []
                    notInLabel = self.X_Xl(self.y, self.X, label)
                    for i in range(totalNumOfPoints):
                        obj.append(self.lambda_ + self.card(self.intersect(self.X[self.B_xj(self.X, i)], notInLabel)))
                    objective = np.dot(obj, self.A) + sum(self.S)  # objective of ILP (approx.)
                    print('if feasible finished')
                    print(time.time() - st)
                else:
                    continue

                if (objective <= 2 * np.log(nPointsInLabel) * optimal):
                    prototypeInd, = np.where(np.array(self.A) == 1)
                    break

            self.Prototype.append(prototypeInd.tolist())

            # related ILP objective
            self.optimalValue += objective

            if verbose:
                print("rounding finished")
                print("time: ", time.time() - start)

        self.prots = sum([len(x) for x in self.Prototype])

    def predict(self, testSet):
        distanceMatrix = self.DM(self.X, testSet)

        predResult = np.array([-1] * len(testSet))

        for count, i in enumerate(self.Prototype):
            for j in i:
                coveredPointsInd, = np.where(distanceMatrix[j] <= self.epsilon_)
                predResult[coveredPointsInd] = np.unique(self.y)[count]

        return (predResult)

    def testError(self, testSet, testLabel):
        testerror = 0
        prototypelist = [item for sublist in self.Prototype for item in sublist]
        prototype = self.X[prototypelist]
        prototypeLabel = self.y[prototypelist]
        distanceMatrix = self.DM(testSet, prototype)
        for count, i in enumerate(distanceMatrix):
            minInd, = np.where(i == min(i.tolist()))
            minInd = minInd[0]  ## maybe many min, only take the first one
            predLabel = prototypeLabel[minInd]
            if predLabel != testLabel[count]:
                testerror += 1
        testerror = testerror / self.card(testSet)
        return (testerror)

    def coverError(self, testSet):
        covererror = 0
        prototypelist = [item for sublist in self.Prototype for item in sublist]
        prototype = self.X[prototypelist]
        distanceMatrix = self.DM(testSet, prototype)
        for count, i in enumerate(distanceMatrix):
            minInd, = np.where(i <= self.epsilon_)
            if len(minInd) == 0:
                covererror += 1
        covererror = covererror / self.card(testSet)
        return (covererror)

    def objective_value(self):
        return (self.optimalValue)

    def numOfProts(self):
        return (self.prots)
        
    def predict(self,testSet):
        distanceMatrix = self.DM(self.X,testSet)

        predResult = np.array([-1] * len(testSet))     
        
        for count,i in enumerate(self.Prototype):
            for j in i:
                coveredPointsInd, = np.where(distanceMatrix[j] <= self.epsilon_)
                predResult[coveredPointsInd] = np.unique(self.y)[count]
        
        return(predResult)
    
    
    def testError(self, testSet, testLabel):
        testerror = 0
        prototypelist = [item for sublist in self.Prototype for item in sublist]
        prototype = self.X[prototypelist]
        prototypeLabel = self.y[prototypelist]
        distanceMatrix = self.DM(testSet,prototype)
        for count,i in enumerate(distanceMatrix):
            minInd, = np.where(i == min(i.tolist()))
            minInd = minInd[0]                 ## maybe many min, only take the first one
            predLabel = prototypeLabel[minInd]
            if predLabel != testLabel[count]:
                testerror += 1
        testerror = testerror / self.card(testSet)
        return(testerror)
    
        
    def coverError(self, testSet):
        covererror = 0
        prototypelist = [item for sublist in self.Prototype for item in sublist]
        prototype = self.X[prototypelist]
        distanceMatrix = self.DM(testSet,prototype)
        for count,i in enumerate(distanceMatrix):
            minInd, = np.where(i <= self.epsilon_)
            if len(minInd)==0:
                covererror += 1
        covererror = covererror / self.card(testSet)
        return(covererror)
    
    
    
    def objective_value(self):
        return(self.optimalValue)
    
    def numOfProts(self):
        return(self.prots)
        
        
        


def cross_val(data, target, epsilon_, lambda_, k, verbose):
    kf = KFold(n_splits=k, random_state = 42)
    score = 0
    prots = 0
    obj_val = 0
    test_Error = 0
    cover_Error = 0
    for train_index, test_index in kf.split(data):
        ps = classifier(data[train_index], target[train_index], epsilon_, lambda_)
        ps.train_lp(verbose)
        obj_val += ps.objective_value()
        score += skmet.accuracy_score(target[test_index], ps.predict(data[test_index]))
        prots += ps.numOfProts()
        test_Error += ps.testError(data[test_index],target[test_index])
        cover_Error += ps.coverError(data[test_index])
        #print(ps.Prototype)
        #print("testError: ",ps.testError(data[test_index],target[test_index]))
        #print("coverError: ",ps.coverError(data[test_index]))
    score /= k    
    prots /= k
    obj_val /= k
    test_Error /= k
    cover_Error /= k
    test_Errorlist.append(test_Error)
    cover_Errorlist.append(cover_Error)
    num_Prots.append(prots)
    obj_list.append(obj_val)
    return print("score: ",score,"\n","# of prots: ",prots,
                 "\n","ILP_obj_val: ",obj_val,"\n", "testError: ",test_Error
                 ,"\n","coverError: ", cover_Error)

''' real tests'''
iris = load_iris(return_X_y = True)
X = iris[0]
y = iris[1]
test_Errorlist = []
cover_Errorlist = []
num_Prots = []
obj_list = []
cross_val(X,y,2,1/150,3,False)










'''QUESTION 2'''
cancer = load_breast_cancer(return_X_y = True)
X = cancer[0]
y = cancer[1]
test_Errorlist = []
cover_Errorlist = []
num_Prots = []
distM = sci.spatial.distance_matrix(X,X)
perc = np.percentile(distM,list(range(2,41,2)))
for epsilon_ in perc:             
    cross_val(X,y,epsilon_,1/(len(X)),4,True)


plt.plot(num_Prots, cover_Errorlist, 'r--', num_Prots, test_Errorlist, 'b')
plt.xlabel("Average # of Prototypes")
plt.ylabel("Errors")
red_patch = mpatches.Patch(color='red', label='Cover Error')
blue_patch = mpatches.Patch(color='blue', label='Test Error')
plt.legend(handles=[red_patch,blue_patch])
plt.show()








'''QUESTION 3'''
digits = load_digits(return_X_y=True)
X = digits[0]
y = digits[1]
obj_list = []
distM = sci.spatial.distance_matrix(X,X)
perc = np.percentile(distM,list(range(2,41,2)))
for epsilon_ in perc:             
    cross_val(X,y,epsilon_,1/(len(X)),4,False)
    
plt.plot(list(range(2,41,2)), obj_list)
plt.xlabel("Epsilon")
plt.ylabel("Objective of ILP")
plt.title("objective for different values of epsilon")
plt.show()    
    
    





'''QUESTION 4'''

class classifier2():
    '''Contains functions for prototype selection'''
    def __init__(self, X, y, epsilon_, lambda_):
        self.X = X
        self.y = y
        self.epsilon_ = epsilon_
        self.lambda_ = lambda_
        
    def intersect(self, leftSet,rightSet):
        aset = set([tuple(x) for x in leftSet])
        bset = set([tuple(x) for x in rightSet])
        return(np.array([x for x in aset & bset]))
    
    def DM(self,array1,array2):
        return(skmet.pairwise_distances(array1, array2,metric='correlation')) 
        #return(skmet.pairwise_distances(array1, array2,metric='chebyshev'))
        #‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule
    
    
    def B_xj(self, array, j):
        distanceMatrix = self.DM(array,array)
        coveredPointsInd, = np.where(distanceMatrix[j] <= self.epsilon_)
        return(coveredPointsInd)
    
    def X_Xl(self, label_array,array, label):
        ind = np.where(label_array != label)
        return(array[ind])
    
    def card(self, array):
        return(len(array))
    
    
    def train_lp(self, verbose = False):

        '''initialization'''
        totalNumOfPoints = self.card(self.X)
        self.optimalValue = 0
        self.Prototype = []
        
        '''solve relaxed LP'''
        for label in np.unique(self.y):

            
            nPointsInLabel = self.card(self.X[np.where(self.y==label)])
        
            ## set variables
            if verbose:
                print ("set var")
            start = time.time()
            alpha = cvx.Variable(totalNumOfPoints)
            xi = cvx.Variable(nPointsInLabel)
            if verbose:
                print("set var finished")
                print("time: ", time.time()-start)
            
            ## objective function
            if verbose:
                print("obj set")
            start = time.time()
            
            obj = 0
            
            notInLabel = self.X_Xl(self.y, self.X, label)
            for i in range(totalNumOfPoints):
                obj1 = self.lambda_ + self.card(self.intersect(self.X[self.B_xj(self.X,i)], notInLabel)) 
                obj2 = obj1*alpha[i]
                obj += obj2
        
            sum_ = reduce((lambda x, y: x + y), xi)
            obj += sum_

            obj = cvx.Minimize(obj) 
            
            if verbose:
                print("obj set finished")
                print("time: ",time.time()-start)
            
            
            
            ## constraints
            
            if verbose:
                print("const set")
            start = time.time()
            
            constraints = [alpha <= 1, alpha >=0 , xi >= 0]
            cons = []
            distanceMatrix = self.DM(self.X,self.X)
            for count,j in enumerate(np.array(np.where(self.y == label)).tolist()[0]):     
                temp = distanceMatrix[j] <= self.epsilon_
                xtemp = temp.astype(int)
                cons += [sum(cvx.mul_elemwise(xtemp, alpha)) >= 1 - xi[count]]
        
            constraints += cons
            
            if verbose:
                print("const set finished")
                print("time: ",time.time()-start)

            ## problem
            if verbose:
                print("prob set")
            start = time.time()
            
            prob = cvx.Problem(obj, constraints)
            if verbose:
                print("prob set finished")
                print("time: ",time.time()-start)

            ## solve
            if verbose:
                print("solving")
            start = time.time()
            optimal = prob.solve(solver = cvx.GUROBI, verbose = False)
            if verbose:
                print("solved")
                print("time: ",time.time()-start)
            
            
            ## verbose = True
            if (verbose):
                print("status:", prob.status)
                print("optimal value", optimal)
            
            
            ''' randomize rounding'''
            if verbose:
                print("rounding started...")
            start = time.time()
            
            distanceMatrix = self.DM(self.X,self.X)
            objective = 2*np.log(nPointsInLabel)*optimal + 100
        
            ## initialize A and S
            self.A = [0] * totalNumOfPoints
            self.S = [0] * nPointsInLabel
            objective = 0  # objective for ILP (approx.)        
        
            for t in range(2*int(np.floor(np.log(nPointsInLabel)))):
        
                #Af = []
                #Sf = []
                #for j in range(totalNumOfPoints):
                #    prob = alpha.value[j]
                #    prob = int(prob * 1000000)/1000000
                #    Af.append(sci.stats.bernoulli.rvs( abs(prob) ))    # draw
                #self.A = list(map(max, self.A, Af))                             # update
                
                flat = (alpha.value).tolist()
                flat_list = [item for sublist in flat for item in sublist]
                a = abs(np.array(list(map(int,(np.array(flat_list)*1000000).tolist())))/1000000).tolist()
                Af = sci.stats.bernoulli.rvs(a).tolist()
                self.A = list(map(max, self.A, Af))
        
                #for i in range(nPointsInLabel):
                #    prob2 = xi.value[i]
                #    prob2 = int(prob2 * 1000000)/1000000
                #    Sf.append(sci.stats.bernoulli.rvs( abs(prob2) ))   # draw
                #self.S = list(map(max, self.S, Sf))                             # update
                
                flat2 = (xi.value).tolist()
                flat_list2 = [item for sublist in flat2 for item in sublist]
                b = abs(np.array(list(map(int,(np.array(flat_list2)*1000000).tolist())))/1000000).tolist()
                Sf = sci.stats.bernoulli.rvs(b).tolist()
                self.S = list(map(max, self.S, Sf))
                
                
                feasible = True
        
                for count,jj in enumerate(np.array(np.where(self.y == label)).tolist()[0]):    
                    temp = distanceMatrix[jj] <= self.epsilon_
                    xtemp = temp.astype(int)
                    if(np.dot(xtemp, self.A) < (1 - self.S[count])):
                        feasible = False
                        continue
        
                if (feasible):
                    obj = []
                    notInLabel = self.X_Xl(self.y, self.X, label)
                    for i in range(totalNumOfPoints):
                        obj.append( self.lambda_ + self.card(self.intersect(self.X[self.B_xj(self.X,i)], notInLabel)) ) 
                    objective = np.dot(obj,self.A) + sum(self.S)  # objective of ILP (approx.)
                else:
                    continue
        
                if (objective <= 2*np.log(nPointsInLabel)*optimal):
                    prototypeInd, = np.where(np.array(self.A) == 1)
                    break
            
            self.Prototype.append(prototypeInd.tolist())
            
            # related ILP objective
            self.optimalValue += objective
            
            if verbose:
                print("rounding finished")
                print("time: ",time.time()-start)
            
            
            
        self.prots = sum([len(x) for x in self.Prototype])  
        
    def predict(self,testSet):
        distanceMatrix = self.DM(self.X,testSet)

        predResult = np.array([-1] * len(testSet))     
        
        for count,i in enumerate(self.Prototype):
            for j in i:
                coveredPointsInd, = np.where(distanceMatrix[j] <= self.epsilon_)
                predResult[coveredPointsInd] = np.unique(self.y)[count]
        
        return(predResult)
    
    
    def testError(self, testSet, testLabel):
        testerror = 0
        prototypelist = [item for sublist in self.Prototype for item in sublist]
        prototype = self.X[prototypelist]
        prototypeLabel = self.y[prototypelist]
        distanceMatrix = self.DM(testSet,prototype)
        for count,i in enumerate(distanceMatrix):
            minInd, = np.where(i == min(i.tolist()))
            minInd = minInd[0]                 ## maybe many min, only take the first one
            predLabel = prototypeLabel[minInd]
            if predLabel != testLabel[count]:
                testerror += 1
        testerror = testerror / self.card(testSet)
        return(testerror)
    
        
    def coverError(self, testSet):
        covererror = 0
        prototypelist = [item for sublist in self.Prototype for item in sublist]
        prototype = self.X[prototypelist]
        distanceMatrix = self.DM(testSet,prototype)
        for count,i in enumerate(distanceMatrix):
            minInd, = np.where(i <= self.epsilon_)
            if len(minInd)==0:
                covererror += 1
        covererror = covererror / self.card(testSet)
        return(covererror)
    
    
    
    def objective_value(self):
        return(self.optimalValue)
    
    def numOfProts(self):
        return(self.prots)
        
        


def cross_val2(data, target, epsilon_, lambda_, k, verbose):
    kf = KFold(n_splits=k, random_state = 42)
    score = 0
    prots = 0
    obj_val = 0
    test_Error = 0
    cover_Error = 0
    for train_index, test_index in kf.split(data):
        ps = classifier2(data[train_index], target[train_index], epsilon_, lambda_)
        ps.train_lp(verbose)
        obj_val += ps.objective_value()
        score += skmet.accuracy_score(target[test_index], ps.predict(data[test_index]))
        prots += ps.numOfProts()
        test_Error += ps.testError(data[test_index],target[test_index])
        cover_Error += ps.coverError(data[test_index])
        #print(ps.Prototype)
        #print("testError: ",ps.testError(data[test_index],target[test_index]))
        #print("coverError: ",ps.coverError(data[test_index]))
    score /= k    
    prots /= k
    obj_val /= k
    test_Error /= k
    cover_Error /= k
    test_Errorlist.append(test_Error)
    cover_Errorlist.append(cover_Error)
    num_Prots.append(prots)
    obj_list.append(obj_val)
    return score, prots, obj_val, test_Error, cover_Error




cancer = load_breast_cancer(return_X_y = True)
X = cancer[0]
y = cancer[1]
test_Errorlist = []
cover_Errorlist = []
num_Prots = []
obj_list = []
distM = skmet.pairwise_distances(X,X,metric='correlation') #metric='chebyshev'
perc = np.percentile(distM,list(range(2,41,2)))
for epsilon_ in perc:             
    cross_val2(X,y,epsilon_,1/(len(X)),4,True)


plt.plot(num_Prots, cover_Errorlist, 'r--', num_Prots, test_Errorlist, 'b')
plt.xlabel("Average # of Prototypes")
plt.ylabel("Errors")
red_patch = mpatches.Patch(color='red', label='Cover Error')
blue_patch = mpatches.Patch(color='blue', label='Test Error')
plt.legend(handles=[red_patch,blue_patch])
plt.show()





