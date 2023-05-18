# README

This Jupyter Notebook contains code for dense retrieval using pre-trained BERT embeddings. It demonstrates how to load a dataset, preprocess the data, generate embeddings for the corpus, and perform retrieval based on search queries. Additionally, it includes implementations of several indexing methods for efficient search, such as KDTree, Hierarchical Navigable Small World (HNSW), Annoy, LSH, and Naive Search.

The KMeansTree is the demo implementation for project **Enhancing Dense Retrieval Efficiency with Hierarchical Clustering**, with choosing K-means as the clustering method.

The benchmark is for KMeansTree's implementation.

## Usage

### init

1. Install the required dependencies.
2. Mount Google Drive by following the instructions provided by the code. (not required)
3. Load the pre-trained BERT model and tokenizer.
4. Load the dataset for training (cc_news in this case).
5. Define the `load_corpus_data()` function to preprocess and filter the dataset. Adjust the desired length range for descriptions.
6. Generate embeddings for the preprocessed corpus using BERT. **(warning, slow!)**
7. Save the generated embeddings to a JSON file.
8. Load the embeddings from the JSON file. (not required if embeddings are generated already)
9. Define the search queries using the `load_search_queries()` function.
10. Generate embeddings for the search queries using BERT.
11. Perform retrieval by calculating the similarity between query embeddings and corpus embeddings.
12. Visualize the distribution of the corpus embeddings using t-SNE.

### KMeansTree

1. Implement and benchmark various indexing methods for efficient search, such as KDTree, HNSW, Annoy, LSH, and Naive Search.
2. Implementation of KMeansTree, which is the demo case of our project.
3. Run the benchmark.
