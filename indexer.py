import os
import json
from collections import defaultdict
from heapq import merge
from bs4 import BeautifulSoup
from nltk.stem.porter import PorterStemmer
from collections import OrderedDict

from tokenizer import tokenize

# CONST GLOBALS
_CHUNK_SIZE = 10000

class Indexer:
    # Create the index dictionary DS
    def __init__(self, partial_index_dir):
        self.inverted_index = defaultdict(list) # inverted index = dict{token, [posting1, ...]}
        self.partial_index_dir = partial_index_dir
        self.doc_count = 0
        self.final_index_file = "final_index"

        # for summary
        self.index_kbs = 0
        self.unique_tokens = set()

    # Extracts the tokens and their freq from the current document to make posting and put each token into index
    def add_document(self, doc_id, content_tokens, url):
        # freq dict of each token in given doc
        term_frequency = defaultdict(int)

        # count occurances of each token in content
        for token in content_tokens:
            term_frequency[token] += 1
            self.unique_tokens.add(token)

        for token, count in term_frequency.items():
            posting = {
                "doc_id": doc_id,
                "term_frequency": count,
                "url": url
            }
            self.inverted_index[token].append(posting)

    # Dumps contents of current index into a new file and clears it
    def save_partial_index(self, partial_index_num):
        partial_index_path = os.path.join(self.partial_index_dir, f"partial_index_{partial_index_num}.json")

        with open(partial_index_path, 'w', encoding='utf-8') as file:
            json.dump(self.inverted_index, file)

        self.inverted_index.clear()

    def build_index(self, documents):
        os.makedirs(self.partial_index_dir, exist_ok=True)
        partial_index_num = 0
        documents_iter = iter(documents.items())

        while True:
            try:
                for _ in range(_CHUNK_SIZE):
                    doc_id, doc_data = next(documents_iter)
                    content = doc_data['content'];
                    url = doc_data['url'];

                    soup = BeautifulSoup(content, "html.parser")
                    text_content = soup.get_text(separator=" ").strip()

                    content_tokens = tokenize(text_content)
                    self.add_document(doc_id, content_tokens, url)
                    self.doc_count += 1

                self.save_partial_index(partial_index_num)
                partial_index_num += 1
            except StopIteration:
                break

        if self.inverted_index:
            self.save_partial_index(partial_index_num)

    # Reads from all partial indexes simultaneously to merge into final complete index
    def multi_way_merge(self):
        partial_files = [
            os.path.join(self.partial_index_dir, f)
            for f in os.listdir(self.partial_index_dir) if f.startswith("partial_index_")
        ]

        # open all partial index files and load iterators for each
        partial_file_iters = []
        for file in partial_files:
            with open(file, 'r', encoding='utf-8') as f:
                partial_index = json.load(f)
                partial_file_iters.append(iter(sorted(partial_index.items())))

        # merging iterators on the token (key) to avoid loading all data into memory
        merged_index = defaultdict(list)
        for token, postings in merge(*partial_file_iters, key=lambda x: x[0]):
            merged_index[token].extend(postings)

        # save final merged index to file
        with open(self.final_index_file, 'w', encoding='utf-8') as file:
            json.dump(merged_index, file)

        # calculate final index size in KB
        self.index_kbs = os.path.getsize(self.final_index_file) / 1024

    # Handles AND boolean queries, returns a list of the top 5 URLs returned
    def search(self, query):
        # process in order of increasing frequency -> start smallest and continue cutting (doc freq)
        # ex: brutus AND calpurnia AND caesar ---> (calpurnia AND brutus) AND ceasar
        # get and of each term in posting
        stemmer = PorterStemmer()
        terms = query.split(' ')

        # apply porter stemming to match how tokens represented in inverted index
        stemmed_terms = [stemmer.stem(term) for term in terms]

        # retrieve posting lists for each term
        with open(self.final_index_file, 'r', encoding='utf-8') as file:
            final_index = json.load(file)

        # retrieve posting lists for each stemmed term from final merged index
        posting_lists = []
        for term in stemmed_terms:
            if term in final_index:
                posting_lists.append(final_index[term])
            else:
                print(f"No documents found for term: {term}")
                return []

        # sort the posting lists by length (smallest first for optimization)
        posting_lists.sort(key=len) #ASC

        # perform intersection of the posting lists
        result_set = posting_lists[0]  # Start with the smallest list
        for posting in posting_lists[1:]:
            result_set = [doc for doc in result_set if doc in posting]  # Perform intersection
            if not result_set:
                print("No documents match the query.")
                return []

        result_docs = sorted(result_set, key=lambda posting: posting['term_frequency'], reverse=True)

        # get the top 5 results or fewer
        # top_5_docs = result_docs[:5]
        urls = OrderedDict()
        for posting in result_docs:
            urls[posting['url']] = None

        return list(urls.keys())


