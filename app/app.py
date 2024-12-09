import time
import sys
import os
from flask import Flask, request, jsonify, render_template

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
app = Flask(__name__)

from searcher import Searcher
from indexer import Indexer

@app.route('/')
def home():
    return render_template('search.html')

@app.route('/search', methods=['GET'])
def handle_search():
    start_time = time.time()
    query = request.args.get('query', '')

    try:
        results = searcher.search(query)
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000  # convert to ms

        return jsonify(results, elapsed_time)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # initialize index and all in-memory data structures
    partial_index_dir = "../PARTIAL_INDEXES"
    final_index_file = "../final_index"
    doc_id_url_file = "../doc_id_url_map.json"
    secondary_index_file = "../secondary_index.json"
    doc_lengths_file = "../doc_len_file.json"

    indexer = Indexer(partial_index_dir, final_index_file, doc_id_url_file, secondary_index_file, doc_lengths_file)
    indexer.load_doc_lengths()
    indexer.load_doc_id_url_map()
    indexer.load_secondary_index()

    # initialize searcher
    searcher = Searcher(indexer)

    # start app
    app.run(debug=True)
