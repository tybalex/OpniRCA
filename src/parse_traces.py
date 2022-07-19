


from dataclasses import dataclass
from elasticsearch import Elasticsearch
from collections import defaultdict
import time
from datetime import datetime
import pickle
import pandas as pd
import logging

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


def datetime_to_timestamp(date):
    st = 0
    if date.endswith("Z"):
        if "." not in date:
            print(date)
            date = date[:-1] + ".000000"
        elif len(second_part:=date.split(".")[1]) == 10:
            date = date[:-4]
            st = 1
        elif len(second_part) == 7:
            date = date[:-1]
            st = 2
        elif len(second_part) == 4:
            date = date[:-1] + "000"
            st = 3
        else:
            print(date)
    d = datetime.strptime(date, "%Y-%m-%dT%H:%M:%S.%f")
    return datetime.timestamp(d) * 1000000


def parse_traces():
    res = get_from_os()
    parse_results(res)


def get_from_os():   
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
      }
    }
    res = es.search( index="otel-v1-apm-span", body=query, scroll = '1m', size=10000)
    l = res["hits"]["hits"]
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
          l.extend(page['hits']['hits'])
          total_size -= scroll_size
    print(f"fetched size: {len(l)}")
    return l



def parse_results(l):
    http_status_set = set()
    pair_service_set = set()
    traces_dict = defaultdict() 
    span_dict = defaultdict(dict)


    for _x in l:
        x = _x["_source"]
        traceid = x["traceId"]
        kind = x["kind"]
        parent_spanid = x["parentSpanId"]
        this_spanid = x["spanId"]
        latency = datetime_to_timestamp(x["endTime"]) - datetime_to_timestamp(x["startTime"])
        if parent_spanid == "": ## top level span
            if kind == "SPAN_KIND_SERVER":
                
                traces_dict[traceid] = {
                "trace_start_timestamp" : datetime_to_timestamp(x["startTime"]),
                "trace_end_timestamp" : datetime_to_timestamp(x["traceGroupFields.endTime"]),
                "http_status" : int(x["span.attributes.http@status_code"]) #// 100
                }
                http_status_set.add(int(x["span.attributes.http@status_code"]))
            elif kind == "SPAN_KIND_CLIENT": 
                custom_http = 210 if int(x["traceGroupFields.statusCode"]) == 0 else 510
                traces_dict[traceid] = {
                "trace_start_timestamp" : datetime_to_timestamp(x["startTime"]),
                "trace_end_timestamp" : datetime_to_timestamp(x["traceGroupFields.endTime"]),
                "http_status" : custom_http
                }
            # span_dict[this_spanid] = {
            #         "trace_id" :traceid,
            #         "span_id" : this_spanid,
            #         "start_timestamp" : datetime_to_timestamp(x["startTime"]),
            #         "end_timestamp" : datetime_to_timestamp(x["endTime"]),
            #         "latency" : latency, #x["durationInNanos"],
            #         }

        else:
            if kind == "SPAN_KIND_CLIENT":
                span_dict[this_spanid] = {
                    "trace_id" :traceid,
                    "span_id" : this_spanid,
                    "start_timestamp" : datetime_to_timestamp(x["startTime"]),
                    "end_timestamp" : datetime_to_timestamp(x["endTime"]),
                    "latency" : latency, #x["durationInNanos"],
                    }
    for _x in l:
        x = _x["_source"]
        traceid = x["traceId"]
        kind = x["kind"]
        parent_spanid = x["parentSpanId"]
        this_spanid = x["spanId"]
        
        if parent_spanid == "": ## top level span
            if kind == "SPAN_KIND_CLIENT":
                span_dict[this_spanid]["source"] = x["serviceName"]
                span_dict[this_spanid]["trace_start_timestamp"] = traces_dict[traceid]["trace_start_timestamp"]
                span_dict[this_spanid]["trace_end_timestamp"] = traces_dict[traceid]["trace_end_timestamp"]
                span_dict[this_spanid]["http_status"] = traces_dict[traceid]["http_status"]
            #assert kind == "SPAN_KIND_SERVER"
            # span_dict[this_spanid]["target"] = x["serviceName"]
            # span_dict[this_spanid]["source"] = "gateway"
            # span_dict[this_spanid]["trace_start_timestamp"] = traces_dict[traceid]["trace_start_timestamp"]
            # span_dict[this_spanid]["trace_end_timestamp"] = traces_dict[traceid]["trace_end_timestamp"]
            # span_dict[this_spanid]["http_status"] = traces_dict[traceid]["http_status"]
            
            # span_dict[this_spanid] = {
            #     "trace_id" :traceid,
            #     "span_id" : this_spanid,
            #     "start_timestamp" : x["startTime"],
            #     "end_timestamp" : x["endTime"],
            #     "latency" : latency, #x["durationInNanos"],
            #     "trace_start_timestamp" : datetime_to_timestamp(x["startTime"]),
            #     "trace_end_timestamp" : datetime_to_timestamp(x["traceGroupFields.endTime"]),
            #     "http_status_code" : x["span.attributes.http@status_code"],
            #     "source" : x["serviceName"],
            #     "target" : x["serviceName"]
            # }
        else: ## child spans
            if traceid not in traces_dict:
                print(f"lost main trace span, id: {traceid}")
                continue
            if kind == "SPAN_KIND_INTERNAL":
                continue
            elif kind == "SPAN_KIND_SERVER":
                span_dict[parent_spanid]["target"] = x["serviceName"]
            elif kind == "SPAN_KIND_CLIENT":
                span_dict[this_spanid]["source"] = x["serviceName"]
                span_dict[this_spanid]["trace_start_timestamp"] = traces_dict[traceid]["trace_start_timestamp"]
                span_dict[this_spanid]["trace_end_timestamp"] = traces_dict[traceid]["trace_end_timestamp"]
                span_dict[this_spanid]["http_status"] = traces_dict[traceid]["http_status"]
                
                
            

    print(f"trace number : {len(traces_dict)}")
    microserviser_pairs = defaultdict(int)
    print(f"http status codes : {http_status_set}")
    features = ['source', 'target', 'start_timestamp', 'end_timestamp',
            'trace_id',
            'latency', 'http_status',
            'trace_start_timestamp', 'trace_end_timestamp']
    data = {
            'source': [], 'target': [], 'start_timestamp': [], 'end_timestamp': [], 'trace_label': [],
            'trace_id': [],
            'latency': [], 'http_status': [],
            'trace_start_timestamp': [], 'trace_end_timestamp': [],
        }
    # for si in span_dict:
    #     span = span_dict[si]
    #     if "trace_id" not in span or span["trace_id"] not in traces_dict or "source" not in span or "target" not in span:
    #         continue
    #     else:
    #         for key in features:
    #             data[key].append(span[key])
    #         data["trace_label"].append(0)
    #         microserviser_pairs[span["source"] + "-" + span["target"]] += 1
 
    # df = pd.DataFrame.from_dict(
    #     data, orient='columns',
    # )
    def new_dict(trace_id, http_status):
        h = http_status // 100
        label = 1 if (h == 4 or h == 5 ) else 0
        raw_features = {
                's_t': [], 'timestamp': [], 'endtime': [], 'label': label,
                'trace_id': trace_id,
                'latency': [], 'http_status': []
            }
        return raw_features

    df = {}
    svcs = set()
    for si in span_dict:
        span = span_dict[si]
        if "trace_id" not in span or span["trace_id"] not in traces_dict or "source" not in span or "target" not in span:
            continue
        else:
            this_trace_id=span["trace_id"]
            http_status = span["http_status"]
            # http_status = 500
            if this_trace_id not in df:
                df[this_trace_id] = new_dict(this_trace_id, http_status)
            df[this_trace_id]["timestamp"].append(span["start_timestamp"])
            df[this_trace_id]["endtime"].append(span["end_timestamp"])
            # if span["target"] == "adservice":
            #     df[this_trace_id]["latency"].append(span["latency"] * 2)
            # else:
                # df[this_trace_id]["latency"].append(span["latency"])
            df[this_trace_id]["latency"].append(span["latency"])
            
            df[this_trace_id]["http_status"].append(span["http_status"])
            df[this_trace_id]["s_t"].append((span["source"], span["target"]))
            svcs.add(span["source"])
            svcs.add(span["target"])

    
    res = [df[d] for d in df]
    df = res

    print(f"valid record : {len(df)}")
    with open(output_file:="realtime.pkl", 'wb') as f:
        logger.info(f"output file : {output_file}")
        pickle.dump(df, f)
    print(microserviser_pairs)
    print(svcs)
        






