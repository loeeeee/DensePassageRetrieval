
from read_source import read_source
import time

TEST_FILE = "10000"

def benchmark(SA, test):
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
    SA.construct()
    time_1 = time.time()
    test_result["index_time"] = time_1 - time_0

    for i in test:
        time_0 = time.time()
        SA.search(i)
        time_1 = time.time()

        test_result["search_time"] += time_1 - time_0

    test_result["search_time"] /= len(test)

    print(test_result)