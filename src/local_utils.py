import requests
import json
import os
import re
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

def fetch_data(data_path):
    # To hit our API, you'll be making requests to:
    base_url = "https://ckan0.cf.opendata.inter.prod-toronto.ca"
    
    # Datasets are called "packages". Each package can contain many "resources"
    # To retrieve the metadata for this package and its resources, use the package name in this page's URL:
    url = base_url + "/api/3/action/package_show"
    params = { "id": "pcard-expenditures"}
    package = requests.get(url, params = params).json()

    
    # To get resource data:
    for idx, resource in enumerate(package["result"]["resources"]):
    
        # To get metadata for non datastore_active resources:
        if not resource["datastore_active"]:
            url = base_url + "/api/3/action/resource_show?id=" + resource["id"]
            resource_metadata = requests.get(url).json()
            if resource_metadata:

                file_url = resource_metadata['result']['url']
                file_name = file_url.split("/")[-1]
                response = requests.get(file_url)

                output_file_path = os.path.join(data_path, "raw", file_name)

                if os.path.exists(output_file_path):
                    continue

                if response.status_code == 200:
                    with open(output_file_path, "wb") as file:
                        file.write(response.content)
                    print(f"File downloaded and saved as {output_file_path}")
                else:
                    print(f"Failed to download the file. Status code: {response.status_code}")

            else: 
                print(f"Fail to download resource: {params['id']}")



def choose_longer_string(strings):
    longest_string = ""
    strings = sorted(strings)
    for s in strings:
        if len(s) > len(longest_string):
            longest_string = s

    return longest_string

def semantic_similarity(sentence1, sentence2, model_name='bert-base-uncased'):
    # Load the pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Tokenize the sentences and get embeddings
    tokens = tokenizer([sentence1, sentence2], return_tensors='pt', padding=True, truncation=True)
    embeddings = model(**tokens).last_hidden_state.mean(dim=1)

    # Calculate cosine similarity
    similarity_score = cosine_similarity(embeddings[0].reshape(1, -1).detach().numpy(), embeddings[1].reshape(1, -1).detach().numpy())[0][0]

    return similarity_score


def remove_stop_words(input_string):
    # Load the list of stop words
    stop_words = set(stopwords.words('english') + ["toronto"])

    # Tokenize the input string into words
    words = input_string.split()

    # Remove stop words and rejoin the words
    filtered_words = [word for word in words if word.lower() not in stop_words]
    filtered_string = ' '.join(filtered_words)

    return filtered_string


def preprocess_text(text):
    # Remove non-alphanumeric characters and split into tokens
    return remove_stop_words(re.sub(r'\s+', ' ',re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()).strip())

