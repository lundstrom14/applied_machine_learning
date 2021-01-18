import numpy as np
import matplotlib.pyplot as plt



class NaiveBayesianClassifier:
    means = [] # list containing a list for each class. These list containts the mean value for each feature. 
    n_class = 0  
    prior = 0 
    cpt = 0

    def fit(self, train_features, train_labels, n_class, max_feature_value = 16):
        self.n_class = n_class
        feature_size = np.shape(train_features)[1]
        sorted_features = [[] for i in range(n_class)] # create 10 lists. Index in list will correspond to label
        self.means = [[] for i in range(n_class)] #

        for tup in zip(train_features, train_labels):
            feature = tup[0]
            label = tup[1]
            sorted_features[label].append(list(feature))    
        
        self.prior = [len(cl)/len(train_features) for cl in sorted_features]  # list of prior for each class

        m = np.array(sorted_features)
        self.cpt = np.zeros((feature_size,n_class, max_feature_value+1)) # 64 depth, 10 rows and 17 columns
        
        for i, cl in enumerate(sorted_features):
            cl = np.array(cl)
            for j, idx in enumerate(range(feature_size)):
                pixs = cl.transpose()[idx][:]
                for pix_v in range(max_feature_value+1):
                    self.cpt[j][i][pix_v] = len(np.where(pixs == pix_v)[0])/len(pixs)
        print(self.cpt[50][5])
        return

    def predict(self, test_features):
        predicted_labels = []
        for feature in test_features:   
            prod = self.prior.copy()
            for c in range(len(prod)):
                for idx, p in enumerate(feature):
                    cpt_prob = self.cpt[idx][c][int(p)]
                    prod[c] *= cpt_prob
            predicted_labels.append(np.argmax(prod))
        return predicted_labels

    def visaulizeLabel(self, label, size):
        
        plt.imshow(np.reshape(self.means[label], size), cmap='gray')
        plt.show()

        #print(norms)
    
