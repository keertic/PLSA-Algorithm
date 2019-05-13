import numpy as np
import math


def normalize(input_matrix):
    """
    Normalizes the rows of a 2d input_matrix so they sum to 1
    """

    row_sums = input_matrix.sum(axis=1)
    assert (np.count_nonzero(row_sums)==np.shape(row_sums)[0]) # no row should sum to zero
    new_matrix = input_matrix / row_sums[:, np.newaxis]
    return new_matrix

       
class Corpus(object):

    """
    A collection of documents.
    """

    def __init__(self, documents_path):
        """
        Initialize empty document list.
        """
        self.documents = []
        self.vocabulary = []
        self.likelihoods = []
        self.documents_path = documents_path
        self.term_doc_matrix = None 
        self.document_topic_prob = None  # P(z | d)
        self.topic_word_prob = None  # P(w | z)
        self.topic_prob = None  # P(z | d, w)

        self.number_of_documents = 0
        self.vocabulary_size = 0

    def build_corpus(self):
        """
        Read document, fill in self.documents, a list of list of word
        self.documents = [["the", "day", "is", "nice", "the", ...], [], []...]
        
        Update self.number_of_documents
        """
        # #############################
        # your code here
        # #############################
        file = open (self.documents_path)
        docs = [doc for doc in file]
        for doc in docs:
            tokens = doc.split ()
            self.documents.append (tokens)
        self.number_of_documents = len(docs)

    def build_vocabulary(self):
        """
        Construct a list of unique words in the whole corpus. Put it in self.vocabulary
        for example: ["rain", "the", ...]

        Update self.vocabulary_size
        """
        # #############################
        # your code here
        # #############################
        unique_words = set()
        for doc in self.documents:
            for word in doc:
                unique_words.add(word)
        self.vocabulary = list(unique_words)
        self.vocabulary_size = len(self.vocabulary)
       

    
    def build_term_doc_matrix(self):
        """
        Construct the term-document matrix where each row represents a document, 
        and each column represents a vocabulary term.

        self.term_doc_matrix[i][j] is the count of term j in document i
        """
        # ############################
        # your code here
        # ############################
        self.term_doc_matrix = np.zeros ( (self.number_of_documents, self.vocabulary_size) )
        for doc_index, doc in enumerate(self.documents, start=0):   # default is zero
            for word in doc:
               if word in self.vocabulary:
                   word_index = self.vocabulary.index(word)
                   self.term_doc_matrix[doc_index, word_index] = self.term_doc_matrix[doc_index, word_index] + 1

       # print ('term_doc_matrix::' )
        #print (self.term_doc_matrix)
                    


    def initialize_randomly(self, number_of_topics):
        """
        Randomly initialize the matrices: document_topic_prob and topic_word_prob
        which hold the probability distributions for P(z | d) and P(w | z): self.document_topic_prob, and self.topic_word_prob

        Don't forget to normalize!
        """
        # ############################
        # your code here
        # ############################
        self.document_topic_prob = np.random.random((self.number_of_documents, number_of_topics))
        self.document_topic_prob = normalize(self.document_topic_prob)

        self.topic_word_prob = np.random.random((number_of_topics, len(self.vocabulary)))
        self.topic_word_prob = normalize(self.topic_word_prob)

        

    def initialize_uniformly(self, number_of_topics):
        """
        Initializes the matrices: self.document_topic_prob and self.topic_word_prob with a uniform 
        probability distribution. This is used for testing purposes.

        DO NOT CHANGE THIS FUNCTION
        """
        self.document_topic_prob = np.ones((self.number_of_documents, number_of_topics))
        self.document_topic_prob = normalize(self.document_topic_prob)

        self.topic_word_prob = np.ones((number_of_topics, len(self.vocabulary)))
        self.topic_word_prob = normalize(self.topic_word_prob)

    def initialize(self, number_of_topics, random=False):
        """ Call the functions to initialize the matrices document_topic_prob and topic_word_prob
        """
        #print("Initializing...")

        if random:
            self.initialize_randomly(number_of_topics)
        else:
            self.initialize_uniformly(number_of_topics)

    def expectation_step(self):
        """ The E-step updates P(z | w, d)
        """
       # print("E step:")
        
        # ############################
        # your code here
        # ############################

        for doc_index, doc in enumerate(self.documents):   
            for word_index in range(self.vocabulary_size):
                prob = self.document_topic_prob[doc_index,:] * self.topic_word_prob[:,word_index]
                #normalize(prob)
                denom = sum (prob)
                if denom == 0:
                   denom=1
                for i in range(len(prob)):
                    prob[i] = prob[i] * 1.0 /denom
                # P(z | d, w)
                self.topic_prob [doc_index, :, word_index] = prob
          
            

    def maximization_step(self, number_of_topics):
        """ The M-step updates P(w | z)
        """
       # print("M step:")
        
        # update P(w | z)
        
        # ############################
        # your code here
        # ############################
        for z in range (number_of_topics):
            for word_index in range(self.vocabulary_size):
                sum = 0
                for doc_index, doc in enumerate(self.documents):   
                    count = self.term_doc_matrix[doc_index, word_index] 
                    sum = sum + count * self.topic_prob [doc_index, z, word_index] 
                self.topic_word_prob[z][word_index] = sum    
            #normalize(self.topic_word_prob[z])
            # Normalize
            denom = np.sum (self.topic_word_prob[z])
            if denom == 0.0:
               denom = 1
            for i in range(len(self.topic_word_prob[z])):
                self.topic_word_prob[z][i] = self.topic_word_prob[z][i] * 1.0 / denom
                 
        
        # update P(z | d)

        # ############################
        # your code here
        # ############################
        
        for doc_index, doc in enumerate(self.documents):   
            for z in range (number_of_topics):
                sum = 0
                for word_index in range(self.vocabulary_size):
                    count = self.term_doc_matrix[doc_index, word_index] 
                    sum = sum + count * self.topic_prob [doc_index, z, word_index] 
                self.document_topic_prob[doc_index][z] = sum    
            #normalize(self.document_topic_prob[doc_index])
            denom = np.sum (self.document_topic_prob[doc_index])
            for i in range(len(self.document_topic_prob[doc_index])):
                self.document_topic_prob[doc_index][i] = self.document_topic_prob[doc_index][i] * 1.0 / denom

    def calculate_likelihood(self, number_of_topics):
        """ Calculate the current log-likelihood of the model using
        the model's updated probability matrices
        
        Append the calculated log-likelihood to self.likelihoods

        """
        # ############################
        # your code here
        # ############################
        likelihood = 0
       # print ('likelihoods:............')
        for doc_index, doc in enumerate(self.documents):   
            for word_index in range(self.vocabulary_size):
                sum = 0
                for z in range (number_of_topics):
                    sum +=  self.topic_word_prob[z,word_index] * self.document_topic_prob[doc_index,z] 
                if sum > 0 :
                   likelihood +=  self.term_doc_matrix[doc_index, word_index] *np.log(sum)
      
        self.likelihoods.append(likelihood)
       # print (self.likelihoods)
        
        return

    def plsa(self, number_of_topics, max_iter, epsilon):

        """
        Model topics.
        """
       # print ("EM iteration begins...")
        
        # build term-doc matrix
        self.build_term_doc_matrix()
        
        # Create the counter arrays.
        
        # P(z | d, w)
        self.topic_prob = np.zeros([self.number_of_documents, number_of_topics, self.vocabulary_size], dtype=np.float)

        # P(z | d) P(w | z)
        self.initialize(number_of_topics, random=True)

        # Run the EM algorithm
        current_likelihood = 0.0

        for iteration in range(max_iter):
          #  print("Iteration #" + str(iteration + 1) + "...")

            # ############################
            # your code here
            # ############################
            last_likelihood = current_likelihood
            self.expectation_step()
            self.maximization_step(number_of_topics)
            self.calculate_likelihood( number_of_topics)
            current_likelihood = self.likelihoods[-1]
            if (abs(last_likelihood - current_likelihood)  < epsilon):
                break
            



def main():
    documents_path = 'data/test.txt'
    corpus = Corpus(documents_path)  # instantiate corpus
    corpus.build_corpus()
    corpus.build_vocabulary()
  #  print(corpus.vocabulary)
   # print("Vocabulary size:" + str(len(corpus.vocabulary)))
    #print("Number of documents:" + str(len(corpus.documents)))
    corpus.build_term_doc_matrix()
    #print(corpus.term_doc_matrix )
    number_of_topics = 2
    max_iterations = 200
    epsilon = 0.001
    corpus.plsa(number_of_topics, max_iterations, epsilon)



if __name__ == '__main__':
    main()
