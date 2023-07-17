# README

This Jupyter Notebook contains code for dense retrieval using pre-trained BERT embeddings. It demonstrates how to load a dataset, preprocess the data, generate embeddings for the corpus, and perform retrieval based on search queries. Additionally, it includes implementations of several indexing methods for efficient search, such as KDTree, Hierarchical Navigable Small World (HNSW), Annoy, LSH, and Naive Search.

The KMeansTree is the demo implementation for project **Enhancing Dense Retrieval Efficiency with Hierarchical Clustering**, with choosing K-means as the clustering method.

The benchmark is for KMeansTree's implementation.

See report [here](http://doc.searchso.cn/docs/EDRFHC.html)

## Usage

### init

1. Install the required dependencies.
2. Mount Google Drive by following the instructions provided by the code. (not required)
3. Create "DenseRetrieve" Folder at the root folder of the Google Drive.
4. Load the pre-trained BERT model and tokenizer.
5. Load the dataset for training (cc_news in this case).
6. Define the `load_corpus_data()` function to preprocess and filter the dataset. Adjust the desired length range for descriptions.
7. Generate embeddings for the preprocessed corpus using BERT. **(warning, slow!)**
8. Save the generated embeddings to a JSON file.
9.  Load the embeddings from the JSON file. (not required if embeddings are generated already)
10. Define the search queries using the `load_search_queries()` function.
11. Generate embeddings for the search queries using BERT.
12. Perform retrieval by calculating the similarity between query embeddings and corpus embeddings.
13. Visualize the distribution of the corpus embeddings using t-SNE.

### KMeansTree

1. Implement and benchmark various indexing methods for efficient search, such as KDTree, HNSW, Annoy, LSH, and Naive Search.
   1. The implementation of the HNSW, Annoy, and LSH are mostly based on the existing library.
   2. The naive search is the iteration of the entire vector space.
   3. They are all warped inside a class that processes two methods, construct and search, so that the benchmark can be done more easily.
2. Implementation of KMeansTree, which is the demo case of our project.
   1. The construction of KMeansTree has the following steps. 
      1. All the vector will be put into a singe leaf layer.
      2. The leaf layer will start a KMeans process. The K here can be predefined fix number or dynamic generated based on the cluster inner distance. The calculation of distance is depend on the distance function that one might choose. The typical choices are l2norm, l1norm, dot product, cosine distance, hamming distance and so on.
      3. After the KMeans process, the algorithm will generate K branches that connected to K Leaves. The relationship is one-to-one. In the branch node, algorithm stores the partitioning information.
      4. The algorithm do the KMeans to each of the Leaf Node again until a certain threshold is meet. The threshold can be depth, or the number of vectors in one single leaf. The threshold can also apply to individual node, meaning that the tree can be imbalanced.
   2. A potential algorithm speed up may be from the parallelism of the KMeans methods, when the tree starts to grow.
   3. When running with large dataset, the KMeans method can be very slow, because of the needs of calculating distance over and over again. The usage of numpy may help but there are also overhead for numpy objects been converted into python objects.
3. Run the benchmark.
   1. The benchmark is relatively a simple program, because of the standardized interface of search methods that constructed. The mainly focuses on the timing of the algorithm. The memory is monitored through htop in the console, as python does not track its own memory very well.
   2. The benchmark times the construct and the search. Before the construct of the index of the search algorithm, the benchmark will first split the data into train and test data. The train data will be first indexed, and the test data will be the input query that the benchmark later searches.
   3. The construct will be timed and the time will be noted. 
   4. Then the benchmark will iterate through the test data, and search them one by one. The time it used will also be noted. The resulting total time will be divide by the number of the test data to get the average query time.
   5. A report will be generated from running the benchmark.
