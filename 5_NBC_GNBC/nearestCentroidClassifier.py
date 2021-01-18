import numpy as np
import matplotlib.pyplot as plt



class NearestCentriodClassifier:
    n_class = 0
    means = [] # list containing a list for each class. These list containts the mean value for each feature. 
            
    def fit(self, train_features, train_labels, n_class):
        self.train_features = train_features
        self.train_labels = train_labels
        self.n_class = n_class # 0-9
        sorted_features = [[] for i in range(n_class)] # create 10 lists. Index in list will correspond to label
        self.means = [[] for i in range(n_class)] #

        for tup in zip(self.train_features, self.train_labels):
            feature = tup[0]
            label = tup[1]
            sorted_features[label].append(feature)
        m = np.array(sorted_features)
        for label in range(self.n_class):
            for pix in range(len(self.train_features[0])):
                self.means[label].append(np.mean([i[pix] for i in m[label]]))
        return

    def predict(self, test_features):
        predicted_labels = np.array([])
        for feature in test_features:
            norms = []
            for mean in self.means:
                norms.append(np.linalg.norm(feature-mean))
            predicted_labels = np.append(predicted_labels, np.argmin(norms))
        #print(predicted_labels)
        return predicted_labels

    def visaulizeLabel(self, label, size):
        
        plt.imshow(np.reshape(self.means[label], size), cmap='gray')
        plt.show()

        #print(norms)
    