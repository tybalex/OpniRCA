# Standard Library
import logging
import os
import time
from typing import List

# Local
import parse_traces as pt
from anomaly_detection import train_history_model
from trace_encoding import span_encoding, trace_encoding
from utils import DATA_DIR, es, list_clusters_and_service

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__file__)
logger.setLevel("DEBUG")


def model_trainer(cluster_id: str, service_list: List[str]):
    logger.info(f"working on cluster : {cluster_id}")
    TRAINING_INTERVAL = "2h"  # 2h
    if cluster_id + "history.pkl" in os.listdir(DATA_DIR):
        logger.info("normal interval model is ready.")
    else:
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"match": {"cluster_id": cluster_id}},
                        {"range": {"endTime": {"gte": "now-" + TRAINING_INTERVAL}}},
                    ]
                }
            }
        }
        res = es.count(index="otel-v1-apm-span", body=query)["count"]
        logger.info(f"available data in last 2 hours : {res}")
        if res > 300000:
            logger.info("======prepare model data==========")
            history_parse = pt.get_parsed_history_data(cluster_id)
            history_trace_dict = trace_encoding(history_parse, cluster_id)
            history_span_df = span_encoding(history_parse, cluster_id, True)
            history_normal_model = train_history_model(
                history_trace_dict, history_span_df, cluster_id
            )
        else:
            logger.info("Not Enough Data Yet....")

    service_list = []
    service_list_filename = cluster_id + "service_list.conf"
    if service_list_filename in os.listdir(DATA_DIR):
        logger.info("=======service_list is ready=========")
        with open(DATA_DIR + "/" + service_list_filename) as fin:
            for line in fin:
                service_list.append(line.strip())
    new_list = service_list
    logger.info(new_list)
    if len(new_list) != len(service_list):  # if they are different
        logger.info(f"===========new service list========== \n {new_list}")
        service_list = new_list
        with open(DATA_DIR + "/" + service_list_filename, "w") as fout:
            for line in service_list:
                fout.write(line + "\n")
    else:
        logger.info("=====service list remains the same=====")


def main():
    while True:
        clusters = list_clusters_and_service()
        logger.info(f"active clusters and services : {clusters}")
        for c_id in clusters:
            model_trainer(c_id, clusters[c_id])
        time.sleep(300)


if __name__ == "__main__":
    main()
