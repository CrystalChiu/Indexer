from doc_loader import load_documents
from indexer import Indexer

def print_summary(number_indexed_docs, number_unique_tokens, total_kbs):
    print(f"Number of Indexed Documents: {number_indexed_docs} \n"
          f"Number of Unique Tokens: {number_unique_tokens} \n"
          f"Total Number KBs of Index: {total_kbs}")

def main():
    # load documents from data directory
    # data_dir = "DEV"  # path to the json files containing html
    data_dir = "TEST"
    documents = load_documents(data_dir)

    # initialize indexer and build the index
    partial_index_dir = "PARTIAL_INDEXES"
    indexer = Indexer(partial_index_dir)
    indexer.build_index(documents)
    indexer.merge_indexes()

    # retrieve & print metrics:
    number_indexed_docs = len(documents)
    number_unique_tokens = len(indexer.unique_tokens)
    total_kbs = indexer.index_kbs

    print_summary(number_indexed_docs, number_unique_tokens, total_kbs)

    print("Indexing complete. Partial indexes saved to disk.")

if __name__ == "__main__":
    main()
