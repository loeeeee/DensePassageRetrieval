from annoy_alt import Annoy
from read_source import read_source
from naive_search import NaiveSearch
import time

TEST_FILE = "10000"

def main():
    vectors, test = read_source(TEST_FILE)
    ann = Annoy(vectors)
    ns = NaiveSearch(vectors)
    test_result = {
            "input_parameters": {
                    "source_file": TEST_FILE,
                    "tested_method": "Annoy",
                    "class_para": {}
                },
            "index_time": 0, # Note how much time is used for indexing
            "hit_rate": 0, # Count how similar the vector is the naive search
            "search_time_abs_improvement": 0, # Count how many seconds was saved when do the search after indexing
            "search_time_relative_improvement": 0, # Count what is the speed up ratio for the search after indexing
            }
    time_0 = time.time()
    ann.construct()
    time_1 = time.time()
    test_result["index_time"] = time_1 - time_0

    for i in test:
        time_0 = time.time()
        result = ann.search(i)
        time_1 = time.time()
        ns_result = ns.search(i)
        time_2 = time.time()
        cnt = 0
        for j in result:
            if j in ns_result:
                cnt += 1

        test_result["hit_rate"] += cnt
        test_result["search_time_abs_improvement"] += time_2 - time_1 - time_1 + time_0
        test_result["search_time_relative_improvement"] += (time_2 - time_1) / (time_1 - time_0)

    test_result["hit_rate"] /= len(test)
    test_result["search_time_abs_improvement"] /= len(test)
    test_result["search_time_relative_improvement"] /= len(test)

    print(test_result)

if __name__ == "__main__":
    main()



