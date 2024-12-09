import time

from doc_loader import load_documents
from indexer import Indexer
from searcher import Searcher

def print_summary(number_indexed_docs, number_unique_tokens, total_kbs):
    print(f"Number of Indexed Documents: {number_indexed_docs} \n"
          f"Number of Unique Tokens: {number_unique_tokens} \n"
          f"Total Number KBs of Index: {total_kbs} kbs")

def main():
    # # load documents from data directory
    # data_dir = "DEV" # path to the json files containing html
    # documents = load_documents(data_dir)

    # build the index
    # partial_index_dir = "PARTIAL_INDEXES"
    # indexer = Indexer(partial_index_dir)
    # indexer.build_index(documents)
    # print("Indexing complete.")
    #
    # # retrieve & print metrics:
    # number_indexed_docs = len(documents)
    # number_unique_tokens = len(indexer.unique_tokens)
    # total_kbs = indexer.index_kbs
    # # print_summary(number_indexed_docs, number_unique_tokens, total_kbs)

    # DEBUG
    partial_index_dir = "PARTIAL_INDEXES"
    indexer = Indexer(partial_index_dir)
    indexer.load_doc_lengths()
    indexer.load_doc_id_url_map()
    # indexer.finish_final_index()

    # populate the in-memony ds
    indexer.load_secondary_index()  # TODO: just put them in the same secondary index ds

    # run the search engine
    searcher = Searcher(indexer)
    while(True):
        query = input("Enter search query: ")

        start_time = time.time()
        # urls = searcher.search(query)
        urls = searcher.bool_search(query)
        print(f"II size: { len(indexer.inverted_index) }")
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000  # convert to ms

        print(f"Query processed in {elapsed_time:.2f} ms")

        for url in urls:
            print(url)

if __name__ == "__main__":
    main()
