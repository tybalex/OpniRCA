# Standard Library
import logging
import time

# Local
import parse_traces as pt
from anomaly_detection import span_anomaly_detection_main, train_history_model
from feature_selection import selecting_feature_main
from localization import rc_localization
from trace_encoding import span_encoding, trace_encoding

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__file__)
logger.setLevel("DEBUG")


def main_rca(cluster_id: str):

    # parse data and preprocess
    start_time = time.time()
    s1 = start_time

    history_parse = pt.load_parsed_history_data(cluster_id)
    if history_parse is None:
        return "model is NOT ready."

    # realtime_parse = pt.load_parsed_realtime_data()
    realtime_parse = pt.get_parsed_realtime_data(cluster_id)

    logger.info(f"parse time taken : {time.time() - start_time }")
    start_time = time.time()
    # reorganize
    history_trace_dict = trace_encoding(history_parse, cluster_id)
    history_span_df = span_encoding(history_parse, cluster_id, True)
    realtime_span_df = span_encoding(realtime_parse, cluster_id)
    logger.info(f"encoding time taken : {time.time() - start_time }")
    start_time = time.time()
    # feature selection
    features = selecting_feature_main(realtime_span_df, history_span_df, cluster_id)
    logger.info(f"feature time taken : {time.time() - start_time }")
    start_time = time.time()
    # normal model
    history_normal_model = train_history_model(
        history_trace_dict, history_span_df, cluster_id
    )
    logger.info(f"normal model time taken : {time.time() - start_time }")
    start_time = time.time()
    # span TAD
    trace_anomaly = span_anomaly_detection_main(
        realtime_span_df, features, history_normal_model, cluster_id
    )
    logger.info(f"TAD time taken : {time.time() - start_time }")
    start_time = time.time()
    # localization
    respond = rc_localization(trace_anomaly, cluster_id)
    logger.info(f"localization time taken : {time.time() - start_time }")
    logger.info(f"Total time taken : {time.time() - s1 }")
    return respond


if __name__ == "__main__":
    main_rca()
