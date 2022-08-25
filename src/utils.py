# Standard Library
import logging
import os
import pickle
from collections import defaultdict
from os.path import join as join

# Third Party
from elasticsearch import Elasticsearch

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__file__)
logger.setLevel("DEBUG")

DATA_DIR = os.getenv("DATA_DIR", "./data")
OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", "https://dev.tybalex.us:9200")

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
        with open(join(DATA_DIR, filename), "rb") as fin:
            df = pickle.load(fin)
        return df
    except Exception as e:
        logger.error(e)
        logger.error("read file error.")
        return None


def write_pkl_file(filename: str, data):
    try:
        with open(join(DATA_DIR, filename), "wb") as f:
            pickle.dump(data, f)
        return True
    except:
        logger.error("write file error.")
        return False


def list_clusters_and_service():
    query = {
        "query": {"bool": {"must": [{"range": {"endTime": {"gte": "now-10m"}}}]}},
        "fields": ["cluster_id", "serviceName"],
        "_source": False,
    }
    cluster_service_dict = defaultdict(set)
    res = es.search(index="otel-v1-apm-span", body=query, scroll="1m", size=10000)
    counter = 0
    counter += len(res)
    [
        cluster_service_dict[h["fields"]["cluster_id"][0]].add(
            h["fields"]["serviceName"][0]
        )
        for h in res["hits"]["hits"]
    ]
    total_size = res["hits"]["total"]["value"]
    print(f"total size : {total_size}")
    total_size -= 10000
    # return l

    sid = res["_scroll_id"]
    # scroll_size = page['hits']['total']

    # # Start scrolling
    while total_size > 0:
        print("Scrolling...")
        page = es.scroll(scroll_id=sid, scroll="1m")
        # Update the scroll ID
        sid = page["_scroll_id"]
        # Get the number of results that we returned in the last scroll
        scroll_size = len(page["hits"]["hits"])
        print("scroll size: " + str(scroll_size))
        [
            cluster_service_dict[h["fields"]["cluster_id"][0]].add(
                h["fields"]["serviceName"][0]
            )
            for h in page["hits"]["hits"]
        ]
        counter += scroll_size
        total_size -= scroll_size
    logger.info(f"fetched size: {counter}")
    return cluster_service_dict
