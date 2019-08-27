"""
Neural network learning using backpropagation and 
gradient descent
"""

import random
import numpy as np
import pickle

class Network(object):

    def __init__(self, sizes,is_from_file=False):
        """
        sizes is a list containing the number of neurons in each layer  
        from input to output, every layer must be specified
        no need to specify bias neurons
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        
        if(is_from_file):
            infile = open("weights",'rb')
            self.weights = pickle.load(infile)
            infile.close()

            infile = open("biases",'rb')
            self.biases = pickle.load(infile)
            infile.close()

            print("weights: ",self.weights)
            print("biases: ",self.biases)

        else:        
            self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
            #print(self.biases)
            self.weights = [np.random.randn(y, x)
                            for x, y in zip(sizes[:-1], sizes[1:])]
            #print(self.weights)


    def feedforward(self, a):
        """given a set of inputs a, calculates the output vector"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None,save_interval=100):
        """Train the model in stochastic gradient descent algorithm
            Training data is a tuple of (x,y) where x and y are input and output vectors
        """
        if test_data:
             n_test = len(test_data)

        n = len(training_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                #print(mini_batch)
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate2(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

            # TODO implement saving to a file function after every 1000 iterations
            if(j%save_interval==0):
                outfile = open("weights",'wb')
                pickle.dump(self.weights,outfile)
                outfile.close()

                outfile = open("biases",'wb')
                pickle.dump(self.biases,outfile)
                outfile.close()
                #np.savetxt("weights.txt",np.array(self.weights))
                #np.savetxt("biases.txt",np.array(self.biases))
                #print("weights: ",self.weights)
                #print("biases: ",self.biases)


    def update_mini_batch(self, mini_batch, eta):
        """do the forward propagation and backpropagation"""
        del_b = [np.zeros(b.shape) for b in self.biases]
        del_w = [np.zeros(w.shape) for w in self.weights]

        # this can be improved using matrix
        for x, y in mini_batch:
            #print(x.shape)
            #print(y.shape)
            delta_del_b, delta_del_w = self.backprop(x, y)
            del_b = [nb+dnb for nb, dnb in zip(del_b, delta_del_b)]
            del_w = [nw+dnw for nw, dnw in zip(del_w, delta_del_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, del_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, del_b)]

    def backprop(self, x, y):
        del_b = [np.zeros(b.shape) for b in self.biases]
        del_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass

        # calculate delta in very last layer
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        del_b[-1] = delta
        del_w[-1] = np.dot(delta, activations[-2].transpose())
        
        # backpropagate 
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            del_b[-l] = delta
            del_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (del_b, del_w)

    def evaluate(self, test_data):
        """how many classified correctly"""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        #print(test_res)
        return sum(int(x == y) for (x, y) in test_results)
        #return sum(int(y[:,x] == 1) for (x, y) in test_results)


    def evaluate2(self,test_data):
        """Confusion matrix"""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        n = self.sizes[-1]
        print("n: ",n)
        matrix = np.zeros((n,n))

        for (x,y) in test_results:
            matrix[x,y] += 1

        print("row index of an entry-> predicted class\n col index of an entry-> actual class\n")
        print(matrix)

    def cost_derivative(self, output_activations, y):
        """Error in last layer"""
        return (output_activations-y)


def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
