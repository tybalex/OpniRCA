from dataclasses import dataclass
from elasticsearch import Elasticsearch
from collections import defaultdict
import time
from datetime import datetime
import pickle
import pandas as pd
import logging
import os
import parse_traces as pt
from trace_encoding import span_encoding, trace_encoding
from anomaly_detection import train_history_model

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__file__)
logger.setLevel("DEBUG")

es = Elasticsearch(
    ["https://demo.tybalex.us:9200"],
    port=9200,
    http_compress=True,
    http_auth=("admin", "admin"),
    verify_certs=False,
    use_ssl=False,
)


def list_latest_service():   
    query = {
      "query": {
        "bool": {
          "must": [
            {
              "range": {
              "endTime": {
              "gte": "now-10m"
                }
              }
            }

          ]
        }
      },
      "fields": [
        "serviceName"
      ],
      "_source": False
    }
    res = es.search( index="otel-v1-apm-span", body=query, scroll = '1m', size=10000)
    l = [h["fields"]["serviceName"][0] for h in res["hits"]["hits"]]
    total_size = res['hits']['total']["value"]
    print(f"total size : {total_size}")
    total_size -= 10000
    # return l

    sid = res['_scroll_id']
    print(sid)
    # scroll_size = page['hits']['total']
      
    # # Start scrolling
    while (total_size > 0):
          print("Scrolling...")
          page = es.scroll(scroll_id = sid, scroll = '1m')
          # Update the scroll ID
          sid = page['_scroll_id']
          # Get the number of results that we returned in the last scroll
          scroll_size = len(page['hits']['hits'])
          print("scroll size: " + str(scroll_size))
          l.extend([h["fields"]["serviceName"][0] for h in page["hits"]["hits"]])
          total_size -= scroll_size
    print(f"fetched size: {len(l)}")
    return set(l)



def main_trainer():
    while True:
        if "history.pkl" in os.listdir("./"):
            logger.info("normal interval model is ready.")
        else:
            query = {
                  "query": {
                    "bool": {
                      "must": [
                        {
                          "range": {
                          "endTime": {
                          "gte": "now-2h"
                            }
                          }
                        }

                      ]
                    }
                  }
                }
            res = es.count(index='otel-v1-apm-span', body=query)["count"]
            logger.info(f"available data in last 2 hours : {res}")
            if res > 300000:
                logger.info("======prepare model data==========")
                history_parse = pt.get_parsed_history_data()
                history_trace_dict = trace_encoding(history_parse)
                history_span_df = span_encoding(history_parse, True)
                history_normal_model = train_history_model(history_trace_dict, history_span_df)

        service_list = []
        if "service_list.conf" in os.listdir("./"):
            logger.info("=======service_list is ready=========")
            with open("service_list.conf", 'r') as fin:
                for line in fin:
                    service_list.append(line.strip())
        new_list = list_latest_service()
        logger.info(new_list)
        if len(new_list) != len(service_list): # if they are different
            logger.info(f"===========new service list========== \n {new_list}")
            service_list = new_list
            with open("service_list.conf", 'w') as fout:
                for line in service_list:
                    fout.write(line + "\n")
        else:
            logger.info("=====service list remains the same=====")
        time.sleep(300)




if __name__ == '__main__':
    main_trainer()




