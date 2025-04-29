# eecs182-homework-5-solved
**TO GET THIS SOLUTION VISIT:** [EECS182 Homework 5 Solved](https://www.ankitcodinghub.com/product/eecs182-solved-2/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;116345&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;2&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (2 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;EECS182 Homework 5  Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (2 votes)    </div>
    </div>
1. Directed and Undirected Graphs

Figure 1: Simple Undirected Graph

Figure 1 shows a simple undirected graph whose adjacency matrices we want to make sure you can write down. Generally, an unnormalized adjacency matrix between the nodes of a directed or undirected graph is given by:

(

1 : if there is an edge between node i and node j,

Ai,j = (1) 0 : otherwise.

This will be a symmetric matrix for undirected graphs. For a directed graph, we have:

(

1 : if there is an edge from node i to node j,

Ai,j = (2) 0 : otherwise.

This need not to be symmetric for a directed graph, and is in fact typically not a symmetric matrix when we are thinking about directed graphs (otherwise, we’d probably be thinking of them as undirected graphs).

Similarly, the degree matrix of an undirected graph is a diagonal matrix that contains information about the degree of each vertex. In other words, it contains the number of edges attached to each vertex and it is given by:

(

deg(vi) : if i == j,

Di,j = (3)

0 : otherwise.

where the degree deg(vi) of a vertex counts the number of times an edge terminates at that vertex.

For directed graphs, the degree matrix could be In-Degree when we count the number of edges coming into a particular node and Out-Degree when we count the number of edges going out of the node. We’ll use the terms in-degree matrix or out-degree matrix to make it clear which one we are invoking.

In that sense, a normalized adjacency matrix is given by:

ANormalized = AD−1 (4)

and a symmetrically normalized adjacency matrix is given by

ASymNorm = D−1/2AD−1/2 (5)

Additionally, the Laplacian matrix relates many useful properties of a graph. In fact, the spectral decomposition of the Laplacian matrix of a graph allows for the construction of low-dimensional embeddings that appear in many machine learning applications. In other words, there is a relation between the properties of a graph and the spectra (eigenvalues and eigenvectors) of matrices associated with the graph, such as its adjacency matrix or Laplacian matrix.

Given a simple graph G with n vertices v1,…,vn, its unnormalized Laplacian matrix Ln×n is defined

element-wise as:

deg(vi) : if i == j,



Li,j = −1 : if i != j and vi is adjacent to vj,

 0 : otherwise.

or equivalently by the matrix: (6)

L = D − A (7)

where D is the degree matrix and A is the adjacency matrix of the graph.

We could also compute the symmetrically normalized Laplacian which is inherited from the adjacency matrix normalization scheme as shown below:

LSymNorm= I − ASymNorm (8)

where I is the identity matrix, A is the unnormalized adjacency matrix, and L is the unnormalized Laplacian.

(a) Show that LSymNorm could also be written as:

LSymNorm = D−1/2LD−1/2 (9)

where D is the dregree matrix, and L is the unnormalized Laplacian.

(b) Write the unnormalized adjacency A, the degree matrix, D, and the symmetrically normalized adjacency matrix, ASymNorm, of the graph in Figure. 1.

(c) Write the symmetrically normalized Laplacian matrix of the graph in Figure. 1.

(d) Compute A2, A3

Figure 2: Simple Directed Graph

The intersections are labeled a to h. The road segments are labeled 1 to 22. The arrows indicate the direction of traffic.

Hint: think about the best way to represent the road network in terms of matrices, vectors, etc.

(e) Write the unnormalized adjacency matrix of the graph in Figure 2.

(f) Write the In-degree Din and Out-degree Dout matrix of the graph in Figure. 2.

(g) Write both of the symmetrically normalized In-degree and Out-degree Laplacian matrix of the graph in Figure. 2.

2. Graph Dynamics

This problem is designed to:

• show connections between these methods.

• show that for a positive integer k, the matrix Ak has an interesting interpretation. That is, the entry in row i and column j gives the number of walks of length k (i.e., a collection of k edges) leading from vertex i to vertex j.

To do this, let’s consider a very simple deep linear network that is built on an underlying graph with n vertices. In the 0-th layer, each node has a single input with weight 1 that is fed a one-hot encoding of its own identity — so node i in the graph has a direct input which is an n−dimensional vector that has a 1 in position i and 0s in all other positions. You can view these as n channels if you want.

The weights connecting node i in layer k to node j in layer k+1 are simply 1 if vertices i and j are connected in the underlying graph and are 0 if those vertices are not connected in the underlying graph. At each layer, the operation at each node is simply to sum up the weighted sum of its inputs and to output the resulting n-dim vector to the next layer. You can think of these as being depth-wise operations if you’d like.

(a) Let A be the n×n size adjacency matrix for the underlying graph where the entry Ai,j = 1 if vertices i and j are connected in the graph and 0 otherwise. Write the output of the j-th node at layer k in this network in terms of the matrix A.

(Hint: This output is an n-dimensional vector since there are n output channels at each layer.)

(b) Here is some helpful notation: Let V (i) be the set of vertices that are connected to vertex i in the graph. Let Lk(i,j) be the number of distinct paths that go from vertex i to vertex j in the graph where the number of edges traversed in the path is exactly k. Recall that a path from i to j in a graph is a sequence of vertices that starts with i, ends with j, and for which every successive vertex in the sequence is connected by an edge in the graph. The length of the path is 1 less than the number of vertices in the corresponding sequence. Show that the i-th output of node j at layer k in the network above is the count of how many paths there are from i to j of length k, where by convention there is exactly 1 path of length 0 that starts at each node and ends up at itself.

(Hint: Can applying induction on k help?)

(c) The structure of the neural network in this problem is compatible with a straightforward linear graph neural network since the operations done (just summing) are locally permutation-invariant at the level of each node and can be viewed as essentially doing the exact same thing at each vertex in the graph based on inputs coming from its neighbors. This is called “aggregation” in the language of graph neural nets. In the case of the computations in previous parts, what is the update function that takes the aggregated inputs from neighbors and results in the output for this node?

(d) The simple GNN described in the previous parts counts paths in the graph. If we were to replace sum aggregation with max aggregation, what is the interpretation of the outputs of node j at layer k?

3. The power of the graph perspective in clustering (Coding)

Implement all the TODOs in the hw5_graph_clustering.ipynb (colab link) notebook. Answer the written questions below and include your completed notebook with your submission.

(a) We used the KMeans algorithm implementation of sklearn, and showed our attempt to cluster this dataset into 3 classes. Comment on the output the KMeans algorithm? Did it work? If so explain why, if not, explain not.

(b) As given, the data points in our dataset are represented simply with their 2D Cartesian coordinates. Let’s now interpret every single point as a node in a graph. Our goal is to find a way to relate every node in the graph in such way that the points that are closer together and points that are far apart maintain that relationship explicitly.

That is, we will choose to look at every point in the dataset as a vertex in a graph where the edge connection between two vertexes is determined by the weighted distances between them. Write a function that takes in the input dataset and some coefficient gamma and returns the adjacency matrix A. Is this a directed or an undirected graph?

Ai,j = e−γ||xi−xj||2 (10)

(c) The degree matrix of an undirected graph is a diagonal matrix that contains information about the degree of each vertex. In other words, it contains the number of edges attached to each vertex and it is given by Eq (4) in problem 3. Note that in the traditional definition of the adjacency matrix, this boils down to the diagonal matrix in which elements along the diagonals are the column-wise sum of the elements in the adjacency matrix. Using the same idea, write a function that takes in the adjacency matrix as an argument and returns the degree matrix.

(d) Using γ = 7.5, compute the adjacency matrix A, degree matrix D and the symmetrically normalized adjacency matrix matrix M,

M = ASymNorm = D−1/2AD−1/2 (11)

Note that another interpretation of the matrix M is that it shows the probability of moving/jumping from one node to another.

(e) Applying SVD decomposition on M, write a function that selects the top 3 vectors (corresponding to the highest singular values) in the matrix U and performs the same KMeans clustering used above on them ; show the plots. What do you observe? Did it work? If so explain why, if not, explain not.

Intuition: By selecting the top 3 vectors of the U matrix, we are selecting a new representation of the data points which could be seen as a construction of a low dimension embedding of the data points as mentioned in problem 3.

(f) Now let’s think of the symmetrically normalized adjacency matrix obtained above as the transition matrix in of a Markov Chain. That is, it represents the probability of jumping from one node to another. In order to fully interpret M in such way, it needs to be a proper stochastic matrix which means that the sum of the elements in each column must add up to 1. Write a function that takes in the matrix M and returns Mstoch, the stochastic version of M; compute the stochastic matrix.

Using SVD decomposition on the newly obtained stochastic matrix Mstoch, use your function in part (e) to select the top 3 vectors of the matrix Ustoch and perform the same KMeans clustering used above on them and show the plots. What do you observe? Did it work?

4. Graph Neural Networks

For an undirected graph with no labels on edges, the function that we compute at each layer of a Graph Neural Network must respect certain properties so that the same function (with weight-sharing) can be used at different nodes in the graph. Let’s focus on a single particular “layer” ℓ. For a given node i in the graph, let sℓi−1 be the self-message (i.e. the state computed at the previous layer for this node) for this node from the preceeding layer, while the preceeding layer messages from the ni neighbors of node i are denoted by mℓi,j−1 where j ranges from 1 to ni. We will use w with subscripts and superscripts to denote learnable scalar weights. If there’s no superscript, the weights are shared across layers. Assume that all dimensions work out.

(a) Tell which of these are valid functions for this node’s computation of the next self-message sℓi.

For any choices that are not valid, briefly point out why.

Note: we are not asking you to judge whether these are useful or will have well behaved gradients. Validity means that they respect the invariances and equivariances that we need to be able to deploy as a GNN on an undirected graph.

where the max acts component-wise on

the vectors. where the max acts component-wise on the vectors.

(b) We are given the following simple graph on which we want to train a GNN. The goal is binary node classification (i.e. classifying the nodes as belonging to type 1 or 0) and we want to hold back nodes 1 and 4 to evaluate performance at the end while using the rest for training. We decide that the surrogate loss to be used for training is the average binary cross-entropy loss.

Figure 3: Simple Undirected Graph

nodes 1 2 3 4 5

yi 0 1 1 1 0

yˆi a b c d e

Table 1: yi is the ground truth label, while yˆi is the predicted probability of node i belonging to class 1 after training.

Table 1 gives you relevant information about the situation.

Compute the training loss at the end of training.

Remember that with n training points, the formula for average binary cross-entropy loss is

!

where the x in the sum ranges over the training points and yˆ(x) is the network’s predicted probability that the label for point x is 1.

(c) Suppose we decide to use the following update rule for the internal state of the nodes at layer ℓ.

s (12)

where the tanh nonlinearity acts element-wise.

For a given node i in the graph, let sℓi−1 be the self-message for this node from the preceeding layer, while the preceeding layer messages from the ni neighbors of node i are denoted by mℓi,j−1 where j ranges from 1 to ni. We will use W with subscripts and superscripts to denote learnable weights in matrix form. If there’s no superscript, the weights are shared across layers.

(i) Which of the following design patterns does this update rule have? □ Residual connection

□ Batch normalization

(ii) If the dimension of the state s is d-dimensional and W2 has k rows, what are the dimensions of the matrix W1?

(iii) If we choose to use the state siℓ−1 itself as the message mℓ−1 going to all of node i’s neighbors, please write out the update rules corresponding to (12) giving sℓi for the graph in Figure 3 for nodes i = 2 and i = 3 in terms of information from earlier layers. Expand out all sums. 5. Zachary’s Karate Club (Coding)

Figure 4: Zachary’s Karate Club Graph

A social network captures 34 members of a karate club, documenting links between pairs of members who interacted outside the club.

We will train a GNN to cluster people in the karate club in such that people who are more likely to associate with either the officer or Mr. Hi will be close together, while the distance beween the 2 classes will be far.

In the original paper titled “Semi-Supervised Classification with Graph Convolutional Networks” that can be found here https://arxiv.org/pdf/1609.02907.pdf, the authors framed this as a node-level classification problem on a graph. We will pretend that we only know the affiliation labels for some of the nodes (which we’ll call our training set) and we’ll predict the affiliation labels for the rest of the nodes (our test set).

Implement all the TODOs in hw5_zkc.ipynb (colab link) and include your notebook with your submission.

(a) Go through q_zkc.ipynb. We want our network to be aware of information about the nodes themselves instead of only the neighborhood, so we add self loops our adjacency matrix. The paper called this A˜. Compute A˜ to add self loops to your adjacency matrix.

(b) Write a function that takes in A˜ as argument and returns the A˜SymNorm adjacency matrix.

(c) The other input to our GNN is the graph node matrix X which contains node features. For simplicity, we set X to be the identity matrix because we don’t have any node features in this example. Generate the feature input matrix X.

(d) We will now implement a single layer GNN. Implement the forward and backward pass functions for GNN_Layer class. Details can be found in the notebook.

(e) Run the forward and backward passes and ensure the checks pass.

(f) We are now ready to setup our classification network! Use the GNN and Softmax layers to setup the network.

(g) Instantiate the GNN model with the correct input and output dimensions.

(h) With the model, data and optimizer ready, fill in the todos in the training loop function and train your model. Plot the clustered data.

(i) Explain why we obtain 100% on accuracy on our test set, yet we see in the plot that 2 samples seem to be misclassified.

6. Homework Process and Study Group

We also want to understand what resources you find helpful and how much time homework is taking, so we can change things in the future if possible.

(a) What sources (if any) did you use as you worked through the homework?

(b) If you worked with someone on this homework, who did you work with?

List names and student ID’s. (In case of homework party, you can also just describe the group.)

(c) Roughly how many total hours did you work on this homework? Write it down here where you’ll need to remember it for the self-grade form.

Contributors:
