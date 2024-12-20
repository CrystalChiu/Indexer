import os
import json

def load_documents(data_dir):
    documents = {}
    for domain_folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, domain_folder)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                try:
                    # load doc metadata without specifying encoding
                    with open(file_path, 'r', encoding='utf-8') as file:
                        doc = json.load(file)

                    # get encoding or default to utf 8
                    encoding = doc.get('encoding', 'utf-8')

                    # read file content
                    with open(file_path, 'r', encoding=encoding, errors='replace') as file:
                        doc = json.load(file)
                        if 'content' in doc:
                            documents[file_name] = doc['content']
                except Exception as e:
                    print(f"Error occurred while reading {file_path}: {e}")

    return documents
