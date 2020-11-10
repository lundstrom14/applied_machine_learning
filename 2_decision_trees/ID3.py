from collections import Counter
from graphviz import Digraph



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
    def add_node_to_graph(self, node, parentid=-1):
        nodeString = ''
        for k in node:
            if ((node[k] != None) and (k != 'nodes')):
                nodeString += "\n" + str(k) + ": " + str(node[k])

        self.__dot.node(str(node['id']), label=nodeString)
        if (parentid != -1):
            self.__dot.edge(str(parentid), str(node['id']))
            nodeString += "\n" + str(parentid) + " -> " + str(node['id'])

        print(nodeString)

        return


    # make the visualisation available
    def make_dot_data(self) :
        return self.__dot


    # For you to fill in; Suggested function to find the best attribute to split with, given the set of
    # remaining attributes, the currently evaluated data and target. 
    def find_split_attr(self, attributes):

        # Change this to make some more sense
        # USE INFORMATION GAIN
        return attributes.copy().popitem()[0]


    # the entry point for the recursive ID3-algorithm, you need to fill in the calls to your recursive implementation
    def fit(self, data, target, attributes, classes):
        # root should become the output of the recursive tree creation
        return self.id3(None, data, target, attributes, None)

    # id3 algorithm
    def id3(self, root, data, target, attributes, target_attribute):
        if (root is None):
            root = self.new_ID3_node() 

        root.update({'classCounts': Counter(target), 'samples': len(data), 'attribute': target_attribute})
        self.add_node_to_graph(root)
        
        # If all samples belong belong to one class, return the single node tree with class as label
        c = target[0]
        for i in range(len(target)):
            if (c != target[i]):
                break
            if (i == len(target)):
                root.update({'label': c})
                return root

        # If Attributes is empty, then return the single node tree with label = most common class value
        if (len(attributes) == 0):
            most_common = Counter(target).most_common(1)[0][0] # counts the most common values in target
            root.update({'label': most_common})
            return root
        else:
            # target_attribute = attribute that generates the maximum information on tree split
            target_attribute = self.find_split_attr(attributes)
            attr_index = list(attributes.keys()).index(target_attribute)  #what position in data corresponds to the target_attribute value. 

            # for each v in values, create a new tree branch below the root node
            rem_attr = attributes.copy() # remaining attributes
            values = rem_attr.pop(target_attribute) # pops the attribute. 
            for v in values:

                # let samples(v) be subset of samples that have value v. 
                samples = []
                for d,t in zip(data,target):
                    if v in d[attr_index]:
                        samples.append([d,t])  #samples is on the form [[('y', 's', 'r'), '+'], [..]], ie nested tuples.
                

                # if samples is empty, add leaf node with label most common class in samples
                if (len(samples) == 0):
                    most_common = Counter(target).most_common(1)[0][0]
                    
                    leaf_node = self.new_ID3_node()
                    leaf_node.update({'label': most_common})
                    self.add_node_to_graph(leaf_node, root['id']) # unsure where to place this
                    return leaf_node
                else:
                    # add subtree below this branch -> call fit recursivly? 
                    # target attribute?
                    subnode = self.new_ID3_node()
                    data_next = [s[0] for s in samples]
                    target_next = [s[1] for s in samples]

                    self.add_node_to_graph(subnode, root['id']) # unsure where to place this
                    self.id3(subnode, data_next, target_next, rem_attr, target_attribute) 
        return root
   
            
                
        


    def predict(self, data, tree) :
        predicted = list()

        # fill in something more sensible here... root should become the output of the recursive tree creation
        return predicted