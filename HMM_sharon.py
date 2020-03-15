########################################
# CS/CNS/EE 155 2018
# Problem Set 6
#
# Author:       Sharon Chen, Andrew Kang
# Description:  Set 6 skeleton code
########################################

# You can use this (optional) skeleton code to complete the HMM
# implementation of set 6. Once each part is implemented, you can simply
# execute the related problem scripts (e.g. run 'python 2G.py') to quickly
# see the results from your code.
#
# Some pointers to get you started:
#
#     - Choose your notation carefully and consistently! Readable
#       notation will make all the difference in the time it takes you
#       to implement this class, as well as how difficult it is to debug.
#
#     - Read the documentation in this file! Make sure you know what
#       is expected from each function and what each variable is.
#
#     - Any reference to "the (i, j)^th" element of a matrix T means that
#       you should use T[i][j].
#
#     - Note that in our solution code, no NumPy was used. That is, there
#       are no fancy tricks here, just basic coding. If you understand HMMs
#       to a thorough extent, the rest of this implementation should come
#       naturally. However, if you'd like to use NumPy, feel free to.
#
#     - Take one step at a time! Move onto the next algorithm to implement
#       only if you're absolutely sure that all previous algorithms are
#       correct. We are providing you waypoints for this reason.
#
# To get started, just fill in code where indicated. Best of luck!

import random
import numpy as np

class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, A, O):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0. 
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.

        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.

            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.

        Parameters:
            L:          Number of states.
            
            D:          Number of observations.
            
            A:          The transition matrix.
            
            O:          The observation matrix.
            
            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''

        self.L = len(A)
        self.D = len(O[0])
        self.A = A
        self.O = O
        self.A_start = [1. / self.L for _ in range(self.L)]


    def viterbi(self, x):
        '''
        Uses the Viterbi algorithm to find the max probability state 
        sequence corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            max_seq:    State sequence corresponding to x with the highest
                        probability.
        '''

        M = len(x)      # Length of sequence.

        # The (i, j)^th elements of probs and seqs are the max probability
        # of the prefix of length i ending in state j and the prefix
        # that gives this probability, respectively.
        #
        # For instance, probs[1][0] is the probability of the prefix of
        # length 1 ending in state 0.
        probs = [[0. for _ in range(self.L)] for _ in range(M + 1)]
        seqs = [['' for _ in range(self.L)] for _ in range(M + 1)]

        #print('M: %d' % M)
        #print('self.L: %d' % self.L)
        for i in range(M + 1):
            for j in range(self.L):
                yj = j
                if i == 0:
                    probs[i][j] = self.A_start[yj]
                    seqs[i][j] = ''
                elif i == 1:
                    probs[i][j] = probs[i-1][j] * self.O[yj][x[i-1]]
                    seqs[i][j] = str(yj)
                else:
                    max_prob = float("-inf")
                    max_seq = ''
                    for k in range(self.L):
                        yi = k
                        prob = probs[i-1][k] * self.A[yi][yj]
                        seq = seqs[i-1][k]
                        if prob > max_prob:
                            max_prob = prob
                            max_seq = seq
                    #print(i,j,yi,yj,yj,x[i-1])
                    probs[i][j] = max_prob * self.O[yj][x[i-1]]
                    seqs[i][j] = max_seq + str(yj)

        # Set final max_seq
        max_prob = float("-inf")
        max_seq = ''
        for i in range(self.L):
            prob = probs[M][i]
            seq = seqs[M][i]
            if prob > max_prob:
                max_prob = prob
                max_seq = seq

        #print(probs)
        return max_seq


    def forward(self, x, normalize=False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            alphas:     Vector of alphas.

                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.

                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        '''

        M = len(x)      # Length of sequence.
        alphas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        for i in range(M + 1):
            for j in range(self.L):
                yj = j
                if i == 0:
                    alphas[i][j] = self.A_start[yj]
                elif i == 1:
                    alphas[i][j] = self.A_start[yj] * self.O[yj][x[i-1]]
                else:
                    sum_alphas = 0
                    for k in range(self.L):
                        yi = k
                        sum_alphas += alphas[i-1][k] * self.A[yi][yj]
                    alphas[i][j] = self.O[yj][x[i-1]] * sum_alphas
        if normalize:
            for i in range(1, M + 1):
                sum_row = sum(alphas[i])
                if sum_row > 0:
                    for j in range(self.L):
                        alphas[i][j] /= sum_row
        return alphas


    def backward(self, x, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            betas:      Vector of betas.

                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.

                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M+1:M, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        '''

        M = len(x)      # Length of sequence.
        betas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        for i in range(M, -1, -1):
            for j in range(self.L):
                yj = j
                if i == M:
                    betas[i][j] = 1.0
                else:
                    sum_betas = 0
                    for k in range(self.L):
                        yi = k
                        sum_betas += betas[i+1][k] * self.A[yj][yi] * self.O[yi][x[i]]
                    betas[i][j] = sum_betas
        if normalize:
            for i in range(M + 1):
                sum_row = sum(betas[i])
                if sum_row > 0:
                    for j in range(self.L):
                        betas[i][j] /= sum_row
        return betas


    def supervised_learning(self, X, Y):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to D - 1. In other words, a list of
                        lists.

            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to L - 1. In other words, a list of
                        lists.

                        Note that the elements in X line up with those in Y.
        '''
        N = len(X)

        # Calculate each element of A using the M-step formulas.
        for a in range(self.L):
            for b in range(self.L):
                numerator = 0
                for j in range(N):
                    yj = Y[j]
                    Mj = len(X[j])
                    for i in range(Mj-1):
                        if yj[i+1] == a and yj[i] == b:
                            numerator += 1.0
                denominator = 0
                for j in range(N):
                    yj = Y[j]
                    Mj = len(X[j])
                    for i in range(Mj-1):
                        if yj[i] == b:
                            denominator += 1.0
                self.A[b][a] = numerator / denominator

        # Calculate each element of O using the M-step formulas.
        for w in range(self.D):
            for z in range(self.L):
                numerator = 0
                for j in range(N):
                    xj = X[j]
                    yj = Y[j]
                    Mj = len(X[j])
                    for i in range(Mj):
                        if xj[i] == w and yj[i] == z:
                            numerator += 1.0
                denominator = 0
                for j in range(N):
                    yj = Y[j]
                    Mj = len(X[j])
                    for i in range(Mj):
                        if yj[i] == z:
                            denominator += 1.0
                self.O[z][w] = numerator / denominator


    def unsupervised_learning(self, X, N_iters):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.

            N_iters:    The number of iterations to train on.
        '''
        # X: contains many sequences, each sequence is of length M.
        # N_iters: we will train N_iters many iterations.

        for n in range(N_iters):
            if n % 10 == 0:
                print('iter_idx: %d' % n)

            A_num = [[0.0 for i in range(self.L)] for j in range(self.L)]
            A_den = [0.0 for j in range(self.L)]
            O_num = [[0.0 for i in range(self.D)] for j in range(self.L)]
            O_den = [0.0 for j in range(self.L)]
            
            for j, x in enumerate(X):
                M = len(x)

                alphas = self.forward(x, normalize=True)
                betas = self.backward(x, normalize=True)

                gamma = np.zeros((M+1, self.L))
                for t in range(1, M+1):
                    for i in range(self.L):
                        gamma[t][i] = alphas[t][i] * betas[t][i]

                for row in gamma[1:, :]:
                    norm = np.sum(row)
                    if norm > 0:
                        row /= norm

                chi = np.zeros((M, self.L, self.L))
                for t in range(1, M):
                    for i in range(self.L):
                        for j in range(self.L):
                            chi[t][i][j] = alphas[t][i] * self.A[i][j] * self.O[j][x[t]] * betas[t+1][j]
                for mat in chi[1:, :, :]:
                    norm = np.sum(mat)
                    if norm > 0:
                        mat /= norm

                for chi_t in chi[1:]:
                    A_num += chi_t
                for gamma_t in gamma[1:M]:
                    A_den += gamma_t
                for t in range(1, M+1):
                    O_den += gamma[t]
                    for i in range(self.L):
                        O_num[i][x[t-1]] += gamma[t][i]

            A = A_num / A_den[:, None]
            O = O_num / O_den[:, None]
            self.A = A
            self.O = O
            if n == 10 or n == 99:
                print(self.A)
                print(self.O)



    def generate_emission(self, M):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random. 

        Arguments:
            M:          Length of the emission to generate.

        Returns:
            emission:   The randomly generated emission as a list.

            states:     The randomly generated states as a list.
        '''

        emission = []
        states = []

        for t in range(M):
            if t == 0:
                prob = self.A_start
            else:
                prob = self.A[states[t-1]]
            y = np.random.choice(range(self.L), p=prob)
            x = np.random.choice(range(self.D), p=self.O[y])
            states.append(y)
            emission.append(x)

        return emission, states


    def probability_alphas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the forward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the state sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any state sequence, i.e.
        # the probability of x.
        prob = sum(alphas[-1])
        return prob


    def probability_betas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the backward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        betas = self.backward(x)

        # beta_j(1) gives the probability that the state sequence starts
        # with j. Summing this, multiplied by the starting transition
        # probability and the observation probability, over all states
        # gives the total probability of x paired with any state
        # sequence, i.e. the probability of x.
        prob = sum([betas[1][j] * self.A_start[j] * self.O[j][x[0]] \
                    for j in range(self.L)])

        return prob


def supervised_HMM(X, Y):
    '''
    Helper function to train a supervised HMM. The function determines the
    number of unique states and observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for supervised learning.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        Y:          A dataset consisting of state sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to L - 1. In other words, a list of lists.
                    Note that the elements in X line up with those in Y.
    '''
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Make a set of states.
    states = set()
    for y in Y:
        states |= set(y)
    
    # Compute L and D.
    L = len(states)
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with labeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.supervised_learning(X, Y)

    return HMM

def unsupervised_HMM(X, n_states, N_iters):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        n_states:   Number of hidden states to use in training.
        
        N_iters:    The number of iterations to train on.
    '''

    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)
    
    # Compute L and D.
    L = n_states
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    random.seed(2020)
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    # Randomly initialize and normalize matrix O.
    random.seed(155)
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X, N_iters)

    return HMM
