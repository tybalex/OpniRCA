from elasticsearch import Elasticsearch
import os
from os.path import join as join
import pickle
import logging

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__file__)
logger.setLevel("DEBUG")

DATA_DIR = os.getenv("DATA_DIR", "./data")
OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", "https://demo.tybalex.us:9200")

es = Elasticsearch(
    [OPENSEARCH_URL],
    port=9200,
    http_compress=True,
    http_auth=("admin", "admin"),
    verify_certs=False,
    use_ssl=False,
)

def read_pkl_file(filename: str):
    try:
        with open(join(DATA_DIR, filename), 'rb') as fin:
            df = pickle.load(fin)
        return df
    except Exception as e:
        logger.error(e)
        logger.error("read file error.")
        return None

def write_pkl_file(filename: str, data):
    try:
        with open(join(DATA_DIR, filename), 'wb') as f:
            pickle.dump(data, f) 
        return True
    except:
        logger.error("write file error.")
        return False