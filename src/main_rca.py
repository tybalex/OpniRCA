

import parse_traces as pt
from trace_encoding import span_encoding, trace_encoding
from feature_selection import selecting_feature_main
from anomaly_detection import train_history_model, span_anomaly_detection_main
from localization import rc_localization
import time

import logging

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__file__)
logger.setLevel("DEBUG")

def main_rca():
	
    # parse data and preprocess
    start_time = time.time()
    s1 = start_time
    realtime_parse = pt.load_parsed_realtime_data()

    history_parse = pt.load_parsed_history_data()
 
    logger.info(f"parse time taken : {time.time() - start_time }" )
    start_time = time.time()
    # reorganize
    history_trace_dict = trace_encoding(history_parse)
    history_span_df = span_encoding(history_parse, True)
    realtime_span_df = span_encoding(realtime_parse)
    logger.info(f"encoding time taken : {time.time() - start_time }" )
    start_time = time.time()
    # feature selection
    features = selecting_feature_main(realtime_span_df, history_span_df)
    logger.info(f"feature time taken : {time.time() - start_time }" )
    start_time = time.time()
    # normal model
    history_normal_model = train_history_model(history_trace_dict, history_span_df)
    logger.info(f"normal model time taken : {time.time() - start_time }" )
    start_time = time.time()
    # span TAD
    trace_anomaly = span_anomaly_detection_main(realtime_span_df, features, history_normal_model)
    logger.info(f"TAD time taken : {time.time() - start_time }" )
    start_time = time.time()
    # localization
    respond = rc_localization(trace_anomaly)
    logger.info(f"localization time taken : {time.time() - start_time }" )
    logger.info(f"Total time taken : {time.time() - s1 }" )
    return respond


if __name__ == '__main__':
    main_rca()