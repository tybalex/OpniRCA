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


def service_list():   
    query = {
      "query": {
        "bool": {
          "must": [
            {
              "range": {
              "endTime": {
              "gte": "now-5m"
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
        if "history.pkl" in os.list_dir("./"):
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
                pt.get_parsed_history_data()

        service_list = []
        if "service_list.txt" in os.list_dir("./"):
            logger.info("service_list is ready.")
            with open("service_list.txt") as fin:
                for line in fin:
                    service_list.append(line.strip())
        new_list = service_list()
        logger.info(new_list)
        if len(new_list) != service_list: # if they are different
            logger.info("new service list!")
            service_list = new_list
            with open("service_list.txt") as fout:
                for line in service_list:
                    fout.write(line + "\n")
        time.sleep(300)




if __name__ == '__main__':
    main_trainer()




