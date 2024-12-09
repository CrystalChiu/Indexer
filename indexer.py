import math
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
    def __init__(
        self,
        partial_index_dir,
        final_index_file="final_index",
        doc_id_url_file="doc_id_url_map.json",
        secondary_index_file="secondary_index.json",
        doc_lengths_file="doc_len_file.json"
    ):
        self.inverted_index = defaultdict(list)  # inverted index = dict{token, [posting1, ...]}
        self.partial_index_dir = partial_index_dir
        self.final_index_file = final_index_file
        self.doc_id_url_map = {}
        self.doc_id_url_file = doc_id_url_file
        self.secondary_index = {}
        self.secondary_index_file = secondary_index_file
        self.doc_lengths = {}
        self.doc_lengths_file = doc_lengths_file

        # for summary
        self.index_kbs = 0
        self.unique_tokens = set()

    #-------------DEBUG HELPER FUNCTIONS------------
    # save doc_id to URL mapping
    def save_doc_id_url_map(self):
        with open(self.doc_id_url_file, 'w', encoding='utf-8') as file:
            json.dump(self.doc_id_url_map, file)

    # load doc_id to URL mapping from file
    def load_doc_id_url_map(self):
        if os.path.exists(self.doc_id_url_file):
            with open(self.doc_id_url_file, 'r', encoding='utf-8') as file:
                self.doc_id_url_map = json.load(file)
        else:
            print(f"{self.doc_id_url_file} not found. Please ensure the file exists.")

    def load_secondary_index(self):
        if os.path.exists(self.secondary_index_file):
            with open(self.secondary_index_file, 'r', encoding='utf-8') as file:
                self.secondary_index = json.load(file)
        else:
            print(f"{self.secondary_index_file} not found. Please ensure the file exists.")

    def load_doc_lengths(self):
        if os.path.exists(self.doc_lengths_file):
            with open(self.doc_lengths_file, 'r', encoding='utf-8') as file:
                self.doc_lengths = json.load(file)
        else:
            print(f"{self.doc_lengths_file} not found. Please ensure the file exists.")

    #-------------INDEX FUNCTIONS------------

    # Extracts the tokens and their freq from the current document to make posting and put each token into index
    def add_document(self, doc_id, content_tokens, url):
        # freq dict of each token in given doc
        term_frequency = defaultdict(int)
        term_positions = defaultdict(list)

        # count occurances of each token in content
        for position, token in enumerate(content_tokens):
            term_frequency[token] += 1
            term_positions[token].append(position)
            self.unique_tokens.add(token)

        # add mapping to url for current document's docID
        self.doc_id_url_map[doc_id] = url

        for token, count in term_frequency.items():
            posting = {
                "doc_id": doc_id,
                "tf": count, # store raw term freq
                "url": url,
            }
            self.inverted_index[token].append(posting)

    # writes contents of current index to partial index file - each entry delimited by newline
    def save_partial_index(self, partial_index_num):
        partial_index_path = os.path.join(self.partial_index_dir, f"partial_index_{partial_index_num}.jsonl")

        with open(partial_index_path, 'w', encoding='utf-8') as file:
            for token, postings in self.inverted_index.items():
                json.dump({token: postings}, file)
                file.write("\n")

        self.inverted_index.clear()

    def multi_way_merge(self):
        partial_files = [
            os.path.join(self.partial_index_dir, f)
            for f in os.listdir(self.partial_index_dir) if f.startswith("partial_index_")
        ]

        # open a read buffer for every partial index
        partial_file_iters = []
        open_files = []
        try:
            for file in partial_files:
                f = open(file, 'r', encoding='utf-8')
                open_files.append(f)
                partial_file_iters.append((json.loads(line) for line in f))

            # continuously write the next entry from a partial index, delimited by newline
            with open(self.final_index_file, 'w', encoding='utf-8') as final_file:
                for entry in merge(*partial_file_iters, key=lambda x: next(iter(x.keys()))):
                    token = next(iter(entry.keys()))
                    postings = entry[token]

                    json.dump({token: postings}, final_file)
                    final_file.write("\n")

            # calc final index size
            self.index_kbs = os.path.getsize(self.final_index_file) / 1024

        finally:
            # close all open files
            for f in open_files:
                f.close()

    # Creates secondary offset index and finds the vector length of each unique document
    def finish_final_index(self):
        num_docs = len(self.doc_id_url_map)
        with open(self.final_index_file, 'r', encoding='utf-8') as index_file:
            while True:
                offset = index_file.tell()  # get current file position first

                line = index_file.readline()
                if not line:
                    break

                entry = json.loads(line)
                term = next(iter(entry))
                postings = entry[term]

                # find term offset in final index and add to bookkeeping index
                self.secondary_index[term] = offset

                idf = math.log(num_docs / len(postings))

                for posting in postings:
                    doc_id = posting['doc_id']
                    tf = posting['tf']
                    tf_idf = tf * idf

                    # accumulate the squared TF-IDF values for the document (everything under the sqrt for magn calc)
                    if doc_id not in self.doc_lengths:
                        self.doc_lengths[doc_id] = 0
                    self.doc_lengths[doc_id] += tf_idf ** 2

        # iterate through the accumulator table to sqrt everything and find doc length for each doc
        for doc_id in self.doc_lengths:
            self.doc_lengths[doc_id] = math.sqrt(self.doc_lengths[doc_id])

        # save secondary index and doc length map to file
        with open(self.doc_lengths_file, 'w', encoding='utf-8') as file:
            json.dump(self.doc_lengths, file)

        with open(self.secondary_index_file, 'w', encoding='utf-8') as aux_file:
            json.dump(self.secondary_index, aux_file)

    def build_index(self, documents):
        # create a directory to hold partial indexes if not exist
        os.makedirs(self.partial_index_dir, exist_ok=True)
        partial_index_num = 0
        documents_iter = iter(documents.items())

        while True:
            try:
                for _ in range(_CHUNK_SIZE):
                    doc_id, doc_data = next(documents_iter)
                    content = doc_data['content']
                    url = doc_data['url']

                    soup = BeautifulSoup(content, "html.parser")
                    text_content = soup.get_text(separator=" ").strip()

                    content_tokens = tokenize(text_content)
                    self.add_document(doc_id, content_tokens, url)

                self.save_partial_index(partial_index_num)
                partial_index_num += 1
            except StopIteration:
                break
        if self.inverted_index:
            self.save_partial_index(partial_index_num)

        self.save_doc_id_url_map()  # DEBUG
        self.multi_way_merge()  # build final index
        self.finish_final_index()  # build bookkeeping index once final index done
