# LDAH2V

*LDAH2V* an efficient computational framework for predicting potential lncRNA-disease associations.


## How to use?

To run *HIN2Vec* on LDAH2V_network, execute the following command from the repository home directory:<br/>

    python main.py res/LDAH2V_network/HIN.txt node_vectors.txt metapath_vectors.txt -l 1280 -d 64 -w 4

### Options

See help for the other available options to use with *HIN2Vec*.<br/>

    python main.py --help

### Input HIN format

The supported input format is an edgelist (separated by TAB):

    node1_name node2_type node2_name node2_type edge_type
                    
The input graph is assumed to be directed by default, which means that for an edge in a undirected graph, you need to add two directed edges. For example:

   1   L   11  L   L-L 
   ...
   11   L   1  L   L-L 
   ...

### Output

Outputs two files: node representations and metapath representations.

The node representation file has *n+1* lines for a graph with *n* nodes with the following format. 

    num_of_nodes dim_of_representation
    node_id dim1 dim2 ... dimd
    ...

where dim1, ... , dimd is the *d*-dimensional node representation learned by *HIN2Vec*.

The metapath representation file has *k+1* lines for a graph with *k* targeted metapath relationships with the following format. 

    num_of_metapaths dim_of_representation
    metapath1 dim1 dim2 ... dimd

where dim1, ... , dimd is the *d*-dimensional metapath representation learned by *HIN2Vec*. The number of target metapaths is related to the window size set for learning and the schema of the given graph.

### Calculate the AUC value

After learning the node representation by *HIN2Vec*, to run *GBT* on LDAH2V_sample, execute the following command from the repository home directory:<br/>
    python GBT.py

Please send any questions you might have about the code and/or the algorithm to <liwenkai@csu.edu.cn>.
