import ID3 as ID3
from collections import OrderedDict




from sklearn import tree, metrics, datasets
import numpy as np


def main():
    classes = (0, 1, 2, 3 , 4, 5, 6, 7, 8, 9)
    attr = ('dark', 'gray', 'light')
    attributes = OrderedDict()
    for i in range(64):
        attributes[i] = attr
    #print(attributes)


        
    digits = datasets.load_digits()
    #print(digits.data)

    digits_data_list = digits.data.tolist()

    data = []
    # maping function did not work out, 
    for row in enumerate(digits_data_list):
        l = []
        i = 0
        for x in row[1]:
            if x < 5:
                l.append('dark')
            elif x > 10:
                l.append('light')
            else:
                l.append('gray')
        data.append(l)
        

    num_split = int(0.7*len(digits.data))
    train_features = data[:num_split]
    train_labels =  digits.target[:num_split]
    test_features = data[num_split:]
    test_labels = digits.target[num_split:]
    #print(train_features)

    id3 = ID3.ID3DecisionTreeClassifier()

    myTree = id3.fit(train_features, train_labels, attributes, classes)
    # #print(myTree)
    plot = id3.make_dot_data()
    plot.render("id3_tree_digits")
    predicted_labels = id3.predict(test_features, myTree)

    print(metrics.classification_report(test_labels, predicted_labels))
    print(metrics.confusion_matrix(test_labels, predicted_labels))
    #print(predicted)
    #print(target2)


if __name__ == "__main__": main()