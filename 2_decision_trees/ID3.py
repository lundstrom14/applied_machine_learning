from collections import Counter
from graphviz import Digraph
import math
import numpy as np


class ID3DecisionTreeClassifier :

    def __init__(self, minSamplesLeaf = 1, minSamplesSplit = 2) :

        self.__nodeCounter = 0

        # the graph to visualise the tree
        self.__dot = Digraph(comment='The Decision Tree')

        # suggested attributes of the classifier to handle training parameters
        self.__minSamplesLeaf = minSamplesLeaf
        self.__minSamplesSplit = minSamplesSplit


    # Create a new node in the tree with the suggested attributes for the visualisation.
    # It can later be added to the graph with the respective function
    def new_ID3_node(self):
        node = {'id': self.__nodeCounter, 'value': None, 'label': None, 'attribute': None, 'entropy': None, 'samples': None,
                         'classCounts': None, 'nodes': None}

        self.__nodeCounter += 1
        return node

    # adds the node into the graph for visualisation (creates a dot-node)
    def add_node_to_graph(self, node, parentid=-1, branch_label = None):
        nodeString = ''
        for k in node:
            if ((node[k] != None)): #and (k != 'nodes')):
                if (k == 'nodes'):
                    nodeString += "\n" + str(k) + ": " + str(len(node[k]))
                else:
                    nodeString += "\n" + str(k) + ": " + str(node[k])

        self.__dot.node(str(node['id']), label=nodeString)
        if (parentid != -1):
            self.__dot.edge(str(parentid), str(node['id']))
            nodeString += "\n" + str(parentid) + " -> " + str(node['id'])

        #print(nodeString)
        return


    # make the visualisation available
    def make_dot_data(self) :
        return self.__dot


    # For you to fill in; Suggested function to find the best attribute to split with, given the set of
    # remaining attributes, the currently evaluated data and target. 
    def find_split_attr(self, data, target, attributes):
        # for all attribute set, test
        IS = self.entropy(target)
        IG_with_attr = (-9999,"") 

        # for every remaning attributes, we want to get the information Gain 
        attr = list(global_attributes.keys())
        for data_col, a in zip(zip(*data),attr): # zip data to only get ordered value from tuples and the attribute, ex ('color')
            entropy = self.subset_entropy(data_col, target, global_attributes[a])
            IG = IS - entropy
            if (IG > IG_with_attr[0]):
                IG_with_attr = (IG, a, IS)
        return IG_with_attr  # returns [IG, a, entropy]
        
    # Return entropy for target set. 
    def entropy(self, target):
        t = len(target)
        counts = Counter(target)
        IS = 0
        for c in counts:
            p_class_i = Counter(target)[c]/t
            IS -= (p_class_i) * math.log2(p_class_i)
        return IS
    
    # Return subset entropy for the subeset. Weighed. data_target = column of data correspoding to an attribute
    def subset_entropy(self, data_target, target, attr_values):
        eps =  7./3 - 4./3 - 1
        IS_s = 0
        #IS_all_s = []
        t = len(data_target)
        l = [] # empty list, will hold the unique, count lists
        d_l = [[d,t] for d,t in list(zip(data_target, target))] # pairs of data-labels
        pairs = np.array(d_l)  # convert to ndarray pairs
        unique, counts = np.unique(pairs, return_counts=True, axis=0)
        for u,c in zip(unique, counts): 
            l.append([list(u),c])
        for v in attr_values:
            IS_sv = 0
            t_v = data_target.count(v)
            for e in l: #for every element in the list of unique-count
                if v in e[0]:
                    fraq = e[1]/(t_v+eps) # len of data with attribute v
                    if fraq == 0: 
                        IS_sv -= 0
                    else:
                        IS_sv -= fraq * math.log2(fraq+eps) #subset of IS, for example color_green

            IS_s +=  t_v/(t+eps) * IS_sv
        return IS_s
                
    # the entry point for the recursive ID3-algorithm, you need to fill in the calls to your recursive implementation
    def fit(self, data, target, attributes, classes):

        global global_attributes # need a global reference to all attributes in id3..
        global_attributes = attributes.copy()

        return self.id3(None, data, target, attributes, None)

    # id3 algorithm
    def id3(self, root, data, target, attributes, target_attribute):
        if (root is None):
            root = self.new_ID3_node() 

        # If all samples belong belong to one class, return the single node tree with class as label
        c = target[0]
        for i in range(len(target)):
            if (c != target[i]):
                break
            if (i == len(target)-1):
                root.update({'label': c})
                return root

        # If Attributes is empty, then return the single node tree with label = most common class value
        if (len(attributes) == 0):
            most_common = Counter(target).most_common(1)[0][0] # counts the most common values in target
            root.update({'label': most_common, 'samples': len(data), 'entropy': self.entropy(target), 'nodes': []})
            return root
        else:
            # target_attribute = attribute that generates the maximum information on tree split
            find_split_attr = self.find_split_attr(data, target, attributes)
            A = find_split_attr[1] #stored in postion 1

            if (root['id'] == 0):
                root.update({'classCounts': Counter(target), 'samples': len(data), 'entropy': self.entropy(target), 'attribute': A, 'nodes': []})
                self.add_node_to_graph(root)
            root.update({'attribute': A})

            attr_index = list(global_attributes.keys()).index(A)  # what position in data corresponds to the target_attribute value. 

            # for each v in values, create a new tree branch below the root node
            # rem_attr = attributes.copy() # remaining attributes
            values = attributes[A]  ############# ERROR IN DIGITS HERE, WE CANT SPLIT ON THE SAME ATTRIBUTE AGAIN
            for v in values:

                # let samples(v) be subset of samples that have value v. 
                samples = []
                for d,t in zip(data,target):
                    if v == d[attr_index]: #############################################################################################
                        samples.append([d,t])  #samples is on the form [[('y', 's', 'r'), '+'], [..]], ie nested tuples.

                root.update({"value": v})
                # if samples is empty, add leaf node with label most common class in samples
                if (len(samples) == 0):
                    most_common = Counter(target).most_common(1)[0][0]

                    leaf_node = self.new_ID3_node()
                    leaf_node.update({'label': most_common, 'samples': len(samples), "value": root["value"], 'nodes': []})
                    root['nodes'].append(leaf_node)
                    self.add_node_to_graph(leaf_node, root['id'])
                else:
                    subnode = self.new_ID3_node()
                    data_next = [s[0] for s in samples]
                    target_next = [s[1] for s in samples]
                    
                    rem_attr = attributes.copy()
                    del rem_attr[A]
                    subnode.update({'nodes': []})
                    node = self.id3(subnode, data_next, target_next, rem_attr, A)
                    node.update({'samples': len(data), 'entropy': self.entropy(target_next), 'classCount': Counter(target_next), "value": v})
                    root['nodes'].append(node)
                    self.add_node_to_graph(node, root['id'])
        return root

    def predict(self, data, tree):
        predicted = list()

        #or x in data:
        for d in data:
            prediction = self.predict_rek(tree, d)
            predicted.append(prediction)

        # fill in something more sensible here... root should become the output of the recursive tree creation
        #return predicted
        #return data
        return predicted
        #return self.predict_rek(tree, data)

    def predict_rek(self, node, x):
        # if node is leaf
        if (self.is_leaf(node)):
            return node['label'] # return the class label of node
        else:
            child_node = self.find_child(node, x)  # find the child c among the children of node 
            return self.predict_rek(child_node, x)

    def is_leaf(self, node):
        if len(node['nodes']) == 0: 
            return True
        return False

    def find_child(self, node, x):
        # x = ('y', 's', 'r')
        A = node['attribute'] # what attribute to check for
        attr_index = list(global_attributes.keys()).index(A)  # what postion/column in the data corresponds to the target_attribute value. 

        for n in node['nodes']:
            if (n['value'] == x[attr_index]): # if the child nodes value matches with what the data, with respect to attribute. (If 's' == x[1])
                return n
        return None #somethind bad happened, or childnodes have not been trained to the input feature
       

