from kdtree import KDTree
from read_source import read_source
from naive_search import NaiveSearch
import time

TEST_FILE = "10000"

def main():
    vectors, test = read_source(TEST_FILE)
    kdt = KDTree(vectors)
    ns = NaiveSearch(vectors)
    test_result = {
            "input_parameters": {
                    "source_file": TEST_FILE,
                    "tested_method": "KDTree",
                    "class_para": {}
                },
            "index_time": 0, # Note how much time is used for indexing
            "search_time": 0, # Count how many seconds was saved when do the search after indexing
            }
    time_0 = time.time()
    kdt.construct()
    time_1 = time.time()
    test_result["index_time"] = time_1 - time_0

    for i in test:
        time_0 = time.time()
        kdt.search(i)
        time_1 = time.time()

        test_result["search_time"] += time_1 - time_0

    test_result["search_time"] /= len(test)

    print(test_result)

if __name__ == "__main__":
    main()



