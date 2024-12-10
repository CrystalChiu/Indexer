import json
import math
import heapq
import nltk
from collections import defaultdict, OrderedDict
from tokenizer import tokenize
from nltk.stem import PorterStemmer

class Searcher:
    def __init__(self, indexer):
        self.indexer = indexer # inverted index
        self.doc_lengths = indexer.doc_lengths # documentID to document length map (for normalization)
        self.secondary_index = indexer.secondary_index  # bookkeeping index
        self.final_index_file = indexer.final_index_file  # path to final inverted index
        self.doc_id_url_map = indexer.doc_id_url_map  # docID to URL map
        self.N = len(self.doc_id_url_map)

        print(f"secondary index len: {len(self.secondary_index)}")
        print(f"doc map len: {len(self.doc_id_url_map)}")
        print(f"N: {self.N}")

    #-----------Ranked Retrieval-----------
    # Retrieves posting list associated with a term via secondary index
    def get_postings(self, term):
        if term not in self.secondary_index:
            print(f"term not found in index: {term}")
            return []

        offset = self.secondary_index[term]
        with open(self.final_index_file, 'r', encoding='utf-8') as index_file:
            index_file.seek(offset)
            line = index_file.readline()
            entry = json.loads(line)

            return entry.get(term, []) # retrieve the given term's posting list or return empty list

    # convert a query into its vector representation weighted by tf-idf
    def process_query(self, query):
        query_tokens = tokenize(query)
        print(f"query tokens: {query_tokens}")
        tfs = defaultdict(int)

        for token in query_tokens:
            tfs[token] += 1

        # calc TF-IDF weights for query
        query_vector = {}
        for term, tf in tfs.items():
            idf = math.log(self.N / len(self.get_postings(term))) if term in self.secondary_index else 0
            query_vector[term] = tf * idf

        return query_vector

    def search(self, query):
        doc_vectors = {} # doc to tf-idf scores
        top_k = 10  # num top documents to retrieve
        heap = []  # min-heap for top-k scores

        query_vector = self.process_query(query)
        query_tf_idf_squared_sum = 0

        # process term-at-a-time, processing terms with higher idf first (maybe stop if doc scores are relatively unchanging?)
        for term in sorted(
            query_vector.keys(),
            key=lambda t: math.log(self.N / (len(self.get_postings(t)) or 1)),
            reverse=True,
        ):
            postings = self.get_postings(term)
            print(f"{len(postings)} found for {term}")
            if not postings: continue

            query_tf_idf = query_vector[term]
            query_tf_idf_squared_sum += query_tf_idf ** 2

            for posting in postings:
                doc_id = posting['doc_id']
                tf = posting['tf']
                idf = math.log(self.N / len(postings))
                doc_weight = tf * idf

                if doc_id not in doc_vectors:
                    doc_vectors[doc_id] = 0

                # update score for accumulator
                # (normalized) doc_tf-idf * query_tf-idf
                doc_vectors[doc_id] += doc_weight * query_tf_idf

        # calculate cosine similarity btwn query vector and each document with at least one query
        for doc_id, score in doc_vectors.items():
            doc_magn = self.indexer.doc_lengths[doc_id] #TODO: rename to doc_vector_lengths
            query_magn = math.sqrt(query_tf_idf_squared_sum)
            normalized_score = score / (doc_magn * query_magn)

            # heap maintenance
            if len(heap) < top_k:
                heapq.heappush(heap, (normalized_score, doc_id))
            else:
                # replace smallest score in the heap if the new score is larger
                if normalized_score > heap[0][0]:
                    heapq.heapreplace(heap, (normalized_score, doc_id))

        # retrieve the results from the heap with k nodes and sort in desc order
        ranked_results = sorted(heap, key=lambda x: x[0], reverse=True)

        # DEBUG
        # return [(self.doc_id_url_map[doc_id], score) for score, doc_id in ranked_results]

        return [self.doc_id_url_map[doc_id] for score, doc_id in ranked_results]

    #-----------Boolean Search-----------
    # Handles AND boolean queries, returns a list of the top 5 URLs returned
    def bool_search(self, query):
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
        posting_lists.sort(key=len)
    
        # perform intersection of the posting lists
        result_set = set(posting['doc_id'] for posting in posting_lists[0])  # Start with the smallest list
        for posting_list in posting_lists[1:]:
            # Use set intersection to retain only the `doc_id`s present in both sets
            result_set &= set(posting['doc_id'] for posting in posting_list)
            if not result_set:
                print("No documents match the query.")
                return []
    
        intersected_postings = []
        for posting_list in posting_lists:
            for posting in posting_list:
                if posting['doc_id'] in result_set:
                    intersected_postings.append(posting)
    
        intersected_postings.sort(key=lambda p: p['tf'], reverse=True)
    
        return list(intersected_postings)