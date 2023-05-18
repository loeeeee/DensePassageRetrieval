# README

This Jupyter Notebook contains code for dense retrieval using pre-trained BERT embeddings. It demonstrates how to load a dataset, preprocess the data, generate embeddings for the corpus, and perform retrieval based on search queries. Additionally, it includes implementations of several indexing methods for efficient search, such as KDTree, Hierarchical Navigable Small World (HNSW), Annoy, LSH, and Naive Search.

The KMeansTree is the demo implementation for project **Enhancing Dense Retrieval Efficiency with Hierarchical Clustering**, with choosing K-means as the clustering method.

The benchmark is for KMeansTree's implementation.

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
   1. 
3. Run the benchmark.
