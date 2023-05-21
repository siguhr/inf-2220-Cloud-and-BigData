import matplotlib

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

import seaborn as sns






###


# Some fixed preliminary computations
#X = np.loadtxt("test_small.txt", dtype=float)
#print('test',test_data)
# print('Dataset')
# print (X)

# N,M=X.shape
# shape = X.shape
# # print(shape)

# tot = M * N
# print(int(tot))
# percentage = 0.01

# c = int(tot * percentage)
# print(c)

# X.ravel()[np.random.choice(X.size, c, replace=False)] = np.nan

# a = np.nan;

lambda_U=1;
lambda_V=1;
K=2;

class MF():
    #print('MF')
    def __init__(self, X, K, learning_rate, beta, iterations, copy):
        self.X = X
        self.num_users, self.num_items = X.shape
        self.K = K
        self.learning_rate = learning_rate
        self.beta = beta
        self.iterations = iterations
        self.copy = copy
        #print(copy)
        #print('MF2', X)

    def train(self):
        # Initialize user and item latent feature matrice
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))
        #print('MF3')
        
        
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

            if (i+1) % 50 == 0:
                #print("Iteration: %d ; error = %.4f" % (i+1, mse))
                print(mse)

                if (i+1) == self.iterations:
                    x, y = np.array(training_process).astype(float).T  # Create plot for TP

                    print(self.copy)
                    print("\n")
                    print(self.full_matrix())
                    
                    plt.scatter(x, y)
                    plt.show()


    
    def mse(self):  # Mean square error.  # totale feilen i arrayet fra sannet til prediction.
        """
        A function to compute the total mean square error
        """
        #print('MF4')
        xs, ys = self.X.nonzero()
        predicted = self.full_matrix()
        error = 0
        
        #print(predicted)

        for x, y in zip(xs, ys):
            # print (x)
            # print(y)
            if  np.isnan(self.X[x,y]) != True:
                pass
            else:
                error += pow(self.copy[x, y] - predicted[x, y], 2)
                #print(error)
                true_error = np.sqrt(error)

        return true_error
        #return np.sqrt(error)
    
    def sgd(self):                                  # Stochastic Gradient Descent calculates the error and updates the model for each example in the training dataset. 
        """
        Perform stochastic graident descent
        """
        #print('MF5')
        for i, j, r in self.samples:                #At the i and j (x,y-cordination) there is a shuffled training value r = (data-point) 
            # Computer prediction and error
            prediction = self.get_rating(i, j)
            #e = (r - prediction)                    # Training error.
            #print(e)
            
            e = (r - prediction)
            
            
            # Create copy of row of P since we need to update it but use older values for update on Q
            P_copy = self.P[i, :][:]
            
            
            # Update user and item latent feature matrices
            self.P[i, :] += (self.learning_rate * (e * self.Q[j, :]))
            self.Q[j, :] += (self.learning_rate * (e * P_copy))


            
    
    def get_rating(self, i, j):
        """
        Get the predicted rating of user i and item j
        """
        #print('MF6')
        prediction = self.P[i, :].dot(self.Q[j, :].T)   # Prediction rating
        #print('->',prediction)                      
        return prediction                               #e.g : -0.012519532091809775

    def full_matrix(self):
        """
        Computer the full matrix using the  P and Q
        """
        return self.P.dot(self.Q.T)                     # Training data
    
#def get_txt():
    # X = np.loadtxt("test_small.txt", dtype=float)
    # shape = X.shape
    # print(shape)

def parse_data():
    """ Parse txt file and return MxN matrix"""

    ####
    # data = []
    # with open("/dbfs/FileStore/tables/mytxt.txt") as csv_file:
    #     csv_reader = csv.reader(csv_file, delimiter=" ")
    #     for row in csv_reader:
    #         data.append(row)
    # X = np.array(data).astype(float)

    ######

    #X = pd.read_csv("/dbfs/FileStore/tables/test_small.csv")
    #print()
    X = np.loadtxt("test_small.txt")
    
    #test = np.loadtxt("test_small.txt")
    

    return X #test


def get_nan(X):

    N,M=X.shape
    shape = X.shape
    # print(shape)

    tot = M * N
    print("Total data set:",int(tot))
    percentage = 0.60                                               # persentage of NAN im array

    c = int(tot * percentage)
    print("x percentage of total:", c)
    X.ravel()[np.random.choice(X.size, c, replace=False)] = np.nan

    a = np.nan;





def main():
    X = parse_data()
    copy = parse_data()
    get_nan(X)

    iterations = 1000
    mf = MF(X, 2, 0.0000001, 5, iterations, copy)
    mf.train()
    print(mf.full_matrix())
    print(copy)
  
    #get_txt()
    

     

if __name__ == '__main__':
    main()