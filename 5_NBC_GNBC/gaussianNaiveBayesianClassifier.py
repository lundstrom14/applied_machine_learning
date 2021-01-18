import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm




class GaussianNaiveBayesianClassifier:
    means = [] # list containing a list for each class. These list containts the mean value for each feature. 
    n_class = 0  
    prior = 0 
    cpt_gauss = 0

    def fit(self, train_features, train_labels, n_class, max_feature_value = 16):
        self.n_class = n_class
        feature_size = np.shape(train_features)[1]
        sorted_features = [[] for i in range(n_class)] # create 10 lists. Index in list will correspond to label
        self.means = [[] for i in range(n_class)] #
        self.stds = [[] for i in range(n_class)] #

        for tup in zip(train_features, train_labels):
            feature = tup[0]
            label = tup[1]
            sorted_features[label].append(list(feature))   
        self.prior = [len(cl)/len(train_features) for cl in sorted_features]  # list of prior for each class

        for label in range(self.n_class):  # construct means and std's over each feature for each class
            for pix in range(len(train_features[0])):
                l = [i[pix] for i in sorted_features[label]]
                self.means[label].append(np.mean(l))
                self.stds[label].append(np.std(l))
        return

    def predict(self, test_features):
        predicted_labels = []
        for feature in test_features:
            prob_classes = [[] for i in range(self.n_class)] # 10 classes
            for c, prob_class in enumerate(prob_classes):
                sigmas = np.array(self.stds[c])+0.01
                mu = np.array(self.means[c])
                prob_classes[c] = norm.pdf(feature, mu, sigmas) * self.prior[c] # Here we have the joint probability
            total_prob_class = [np.prod(prob_class) for prob_class in prob_classes] # Product over every joint probability, (each pixel and its priori)
            predicted_labels.append(np.argmax(total_prob_class))  # argmax. Find Maximum a priori. 
        return predicted_labels

    def visaulizeLabel(self, label, size):
        plt.imshow(np.reshape(self.means[label], size), cmap='gray')
        plt.show()

        #print(norms)
    
