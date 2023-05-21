
import matplotlib

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

import seaborn as sns


# lambda_U=1;
# lambda_V=1;
# K=2;

class MF():
    
    def __init__(self, X, K, learning_rate, iterations, copy, beta):
        self.X = X
        self.num_users, self.num_items = X.shape
        self.K = K
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.copy = copy
        self.beta = beta

        

    def train(self):
        # Initialize user and item weights
        self.U_users = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.V_items = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

        
        # Create a list of training samples
        self.samples = [                                    # Shuffle  data set 
                (i, j, self.X[i, j])                        # The sample cointain three values 
                for i in range(self.num_users)              # At the i and j (x,y-cordination) there is a shuffled training value (data-point)
                for j in range(self.num_items)
                if  np.isnan(self.X[i,j]) != True               
            ]
        

        training_process = []
        for i in range(self.iterations):                # in rage for number of iterations
            np.random.shuffle(self.samples)             # Random shuffle of the samples
            self.sgd()
            mse = self.mse()
            training_process.append((i, mse))
            if i !=0:
                if mse < 0.0001: #(i+1) % 10 == 0:
                    print("stop loop at iteration:", i+1, "mse", mse)
                    break

            if (i+1) % 50 == 0:                                         #Print error value at every x iteration 
                #print("Iteration: %d ; error = %.4f" % (i+1, mse))
                print(mse)

                if (i+1) == self.iterations:
                    x, y = np.array(training_process).astype(float).T  # Create plot for training process

                  
                    print(self.copy)
                    print("\n")
                    print(self.full_matrix())
                    
                    plt.scatter(x, y)
                    plt.show()

                   

    

    def compare(self):


        prediction = np.dot(self.U_users, self.V_items. T)
        self.findNAN = [                                    # Shuffle  data set 
                (i, j, self.X[i, j])                        # The sample cointain three values 
                for i in range(self.num_users)              # At the i and j (x,y-cordination) there is a shuffled training value (data-point)
                for j in range(self.num_items)
                if  np.isnan(self.X[i,j]) == True               
            ]
        #for i, j, l in (self.findNAN):
            #print(self.copy[i,j], prediction[i,j])
            
            
    
    def mse(self):  # Mean square error. Calculate the total error value compared to the original (copy)
        """
        A function to compute the total mean square error
        """
        #print('MF4')
        xs, ys = self.X.nonzero()
        predicted = self.full_matrix()              # call full matrix to get predicted value
        error = 0
        
        #print(predicted)

        for x, y in zip(xs, ys):
            # print (x)
            # print(y)
            if  np.isnan(self.X[x,y]) == True:
                pass
            else:
                error += pow(self.copy[x, y] - predicted[x, y], 2)
                #print(error)
        true_error = np.sqrt(error)

        return true_error                           #Return total mean square error
        #return np.sqrt(error)
    
    def sgd(self):                                  # Stochastic Gradient Descent calculates the error and updates the model for each example in the training dataset. 
        """
        Perform stochastic graident descent
        """
        #print('MF5')
        for i, j, r in self.samples:                #At the i and j (x,y-cordination) there is a shuffled training value r = (data-point) 
            # Computer prediction and error
            prediction = self.get_rating(i, j)
            
            
            e = (r - prediction)                    # Training error calculated
            
            
            # Create copy of row of U_users since we need to update it but use older values for update on V_items
            U_users_copy = self.U_users[i, :][:]
            
            
            # Update the weights
            self.U_users[i, :] += self.learning_rate * (e * self.V_items[j, :] - self.beta * self.U_users[i, :])
            self.V_items[j, :] += self.learning_rate * (e * U_users_copy - self.beta - self.V_items[j, :])
            

    
    def get_rating(self, i, j):
        """
        Get the predicted rating of user i and item j
        """
        #print('MF6')
        prediction = self.U_users[i, :].dot(self.V_items[j, :].T)   # Prediction rating
        #print('->',prediction)                      
        return prediction                               

    def full_matrix(self):
        """
        Computer the full matrix using the  P and Q
        """
       
        return self.U_users.dot(self.V_items.T)                     # Training data
    


def parse_data():
    """ Parse txt file and return MxN matrix"""

 
    X = np.loadtxt("test_small.txt")                                # Load txt file into function and return
    

    return X 


def get_nan(X):

    N,M=X.shape
    shape = X.shape
    # print(shape)

    tot = M * N
    print("Total data set:",int(tot))
    percentage = 0.90                                               # Percentage of NAN in array

    c = int(tot * percentage)                                       # Calculate percentage of NAN
    print("x percentage of total:", c)
    X.ravel()[np.random.choice(X.size, c, replace=False)] = np.nan  # Put NAN at rrandom location in matrix

    a = np.nan;



def main():

    # train ##
    X = parse_data()
    copy = parse_data()
    get_nan(X)

    learning_rate = 0.0000001
    iterations = 1200
    beta = 1
    K = 2

    mf = MF(X, K, learning_rate, iterations, copy, beta)        # Fill the class with desired parameters

    mf.train()                                                  # Function to train the weights
    mf.compare()                                                # Compare weights with the copy (original)
    
    
    print(mf.full_matrix())                                     
    print('\n')
    print(copy)


if __name__ == '__main__':
    main()