from sklearn import metrics, datasets
import MNIST
from nearestCentroidClassifier import NearestCentriodClassifier
from naiveBayesianClassifier import NaiveBayesianClassifier
from gaussianNaiveBayesianClassifier import GaussianNaiveBayesianClassifier
import time



def main_ncc() :
    mnist = MNIST.MNISTData('MNIST_Light/*/*.png')

    train_features, test_features, train_labels, test_labels = mnist.get_data()

    ncc = NearestCentriodClassifier()
    ncc.fit(train_features, train_labels, 10)

    #ncc.visaulizeLabel(3, (20,20)) 

    y_pred = ncc.predict(test_features)

    print("Classification report SKLearn GNB:\n%s\n"
      % (metrics.classification_report(test_labels, y_pred)))
    print("Confusion matrix SKLearn GNB:\n%s" % metrics.confusion_matrix(test_labels, y_pred))


def main_nbc() :
  
    digits = datasets.load_digits()

    digits.data
    digits.data.shape

    num_split = int(0.7*len(digits.data))
    train_features = digits.data[:num_split]
    train_labels =  digits.target[:num_split]
    test_features = digits.data[num_split:]
    test_labels = digits.target[num_split:]

    # mnist = MNIST.MNISTData('MNIST_Light/*/*.png')

    # train_features, test_features, train_labels, test_labels = mnist.get_data()

    # print(train_features)
    
    # train_features = train_features*255
    # test_features = test_features*255

    nbc = NaiveBayesianClassifier()
    nbc.fit(train_features, train_labels, 10, 255)
    
    
    

    #print("Classification report SKLearn GNB:\n%s\n"
    #  % (metrics.classification_report(test_labels, y_pred)))
    print("Confusion matrix SKLearn GNB:\n%s" % metrics.confusion_matrix(test_labels, y_pred))

    #mnist.visualize_wrong_class(y_pred, 8)

    #mnist.visualize_wrong_class(y_pred, 8)
def main_gnb() :
    digits = datasets.load_digits()

    digits.data
    digits.data.shape

    num_split = int(0.7*len(digits.data))
    train_features = digits.data[:num_split]
    train_labels =  digits.target[:num_split]
    test_features = digits.data[num_split:]
    test_labels = digits.target[num_split:]

    gnb = GaussianNaiveBayesianClassifier()
    gnb.fit(train_features, train_labels, 10)
    #ncc.visaulizeLabel(3, (20,20)) 
    
    tic = time.clock()
    y_pred = gnb.predict(test_features)
    toc = time.clock()
    print(toc-tic)

    print("Classification report SKLearn GNB:\n%s\n"
      % (metrics.classification_report(test_labels, y_pred)))
    print("Confusion matrix SKLearn GNB:\n%s" % metrics.confusion_matrix(test_labels, y_pred))

if __name__ == "__main__": main_gnb()