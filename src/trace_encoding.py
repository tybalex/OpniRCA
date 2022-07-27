import sys
from typing import Dict, Tuple,List
import numpy as np

import pickle
import pandas as pd
from pathlib import Path
import logging

from rca_config import *

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__file__)
logger.setLevel("DEBUG")
"""
Encode train-ticket pickle data into data frame of invocations:
    source, target, start time, end time, trace_id, features
    ...
"""

### python run_invo_encoding.py -i data/test/basic_abort_1011.pkl -o dataframe/basic_abort_1011.pkl
### python run_invo_encoding.py -i A/uninjection/3.pkl -o dataframe/uninjection/3.pkl


def simple_name(full_name):
    if 'istio-ingressgateway' in full_name:
        return 'gateway'
    full_name = full_name.split('.')[0]
    ret_list = []
    ok = False
    for part in full_name.split("-"):
        if part == "ts":
            ok = True
        elif "service" in part:
            ok = False
            break
        elif ok:
            ret_list.append(part)
    if len(ret_list) <= 0:
        ret = full_name
    else:
        ret = "-".join(ret_list)
    assert ret in INVOLVED_SERVICES, f"full={full_name}, ret_list={ret_list}"
    return ret


def span_encoding(input_data, is_history=False) -> pd.DataFrame:
    # input_file = Path(input_file)
    # output_file = Path(output_file)
    # output_file.parent.mkdir(exist_ok=True)
    # # logger.debug(f"input file: {input_file}, output_file: {output_file}")
    # with open(str(input_file.resolve()), 'rb') as f:
    #     input_data = pickle.load(f)
    if is_history:
        try:
            with open("encoded_history_span.pkl", 'rb') as fin:
                df = pickle.load(fin)
            logger.info("using encoded span history data.")
            return df
        except:
            logger.info("encoded history span doesn't exist, will generate one.")

    if ENABLE_ALL_FEATURES:
        data = {
            'source': [], 'target': [], 'start_timestamp': [], 'end_timestamp': [], 'trace_label': [],
            'trace_id': [],
            'latency': [], 'cpu_use': [], 'mem_use_percent': [], 'mem_use_amount': [],
            'file_write_rate': [], 'file_read_rate': [],
            'net_send_rate': [], 'net_receive_rate': [], 'http_status': [],
            'trace_start_timestamp': [], 'trace_end_timestamp': [],
        }
    else:
        data = {
            'source': [], 'target': [], 'start_timestamp': [], 'end_timestamp': [], 'trace_label': [],
            'trace_id': [],
            'latency': [], 'http_status': [],
            'trace_start_timestamp': [], 'trace_end_timestamp': [],
        }

    for trace in input_data:
        indices = np.asarray([idx for idx, (source, target) in enumerate(trace['s_t']) if source != target])
        if len(indices) <= 0:
            continue
        for key, item in trace.items():
            if isinstance(item, list) and key != 'root_cause' and key != 'fault_type':
                try:
                    trace[key] = np.asarray(item)[indices]
                except IndexError:
                    raise RuntimeError(f"{key} {item} {indices}")
        data['source'].extend(list(simple_name(_[0]) for _ in trace['s_t']))
        data['target'].extend(list(simple_name(_[1]) for _ in trace['s_t']))

        if ENABLE_ALL_FEATURES:
            data['start_timestamp'].extend(_ / 1e6 for _ in trace['timestamp'])
            data['end_timestamp'].extend(_ / 1e6 for _ in trace['endtime'])
            data['trace_start_timestamp'].extend(min(trace['timestamp']) / 1e6 for _ in trace['timestamp'])
            data['trace_end_timestamp'].extend(max(trace['endtime']) / 1e6 for _ in trace['endtime'])
            data['trace_label'].extend(trace['label'] for _ in trace['s_t'])
            data['trace_id'].extend(trace['trace_id'] for _ in trace['s_t'])
            data['latency'].extend(_ / 1e6 for _ in trace['latency'])
            data['cpu_use'].extend(_ * 1e-2 for _ in trace['cpu_use'])
            data['mem_use_percent'].extend(_ / 1e2 for _ in trace['mem_use_percent'])  #
            data['mem_use_amount'].extend(_ / 1e12 for _ in trace['mem_use_amount'])  # 1000MB disabled
            data['file_write_rate'].extend(_ / 1e12 for _ in trace['file_write_rate'])  # 100MB
            data['file_read_rate'].extend(_ / 1e12 for _ in trace['file_read_rate'])  # 100MB
            data['net_send_rate'].extend(_ / 1e12 for _ in trace['net_send_rate'])
            data['net_receive_rate'].extend(_ / 1e12 for _ in trace['net_receive_rate'])
            data['http_status'].extend(int(_) // 100 if _ != 0 else 9 for _ in trace['http_status'])
        else:
            data['start_timestamp'].extend(_ for _ in trace['timestamp'])
            data['end_timestamp'].extend(_ for _ in trace['endtime'])
            data['trace_start_timestamp'].extend(min(trace['timestamp']) for _ in trace['timestamp'])
            data['trace_end_timestamp'].extend(max(trace['endtime']) for _ in trace['endtime'])
            data['trace_label'].extend(trace['label'] for _ in trace['s_t'])
            data['trace_id'].extend(trace['trace_id'] for _ in trace['s_t'])
            data['latency'].extend(_ for _ in trace['latency'])
            data['http_status'].extend(int(_) // 100 if _ != 0 else 9 for _ in trace['http_status'])

    df = pd.DataFrame.from_dict(
        data, orient='columns',
    )
    for feature_name in FEATURE_NAMES:
        assert feature_name in df.columns
    for service in np.unique(df.source):
        assert service in INVOLVED_SERVICES, f'{service} {df[df.source == service]}'
    for service in np.unique(df.target):
        assert service in INVOLVED_SERVICES, f'{service} {df[df.source == service]}'

    if is_history:
        with open("encoded_history_span.pkl", 'wb') as fout:
            pickle.dump(df, fout)
    return df
    # with open(output_file, 'wb+') as f:
    #     logger.info(f"output file : {output_file}")
    #     logger.info(df)
    #     pickle.dump(df, f)

def encoding_data(source_data: List, drop_service=(), drop_fault_type=()):
    def pair2index(s_t):
        return SERVICE2IDX.get(simple_name(s_t[1]))

    if ENABLE_ALL_FEATURES:
        _data = np.ones((len(source_data), len(INVOLVED_SERVICES), 9), dtype=np.float32) * -1
    else:
        _data = np.ones((len(source_data), len(INVOLVED_SERVICES), 2), dtype=np.float32) * -1

    _labels = np.zeros((len(source_data),), dtype=np.bool)
    _trace_ids = [""] * len(source_data)
    _service_mask = np.zeros((len(source_data), len(INVOLVED_SERVICES)), dtype=np.bool)
    _root_causes = np.zeros((len(source_data), len(INVOLVED_SERVICES)), dtype=np.bool)
    for trace_idx, trace in enumerate(source_data):
        if 'fault_type' in trace and trace['fault_type'] in drop_fault_type:
            continue
        if 'root_cause' in trace and any(_ in drop_service for _ in trace['root_cause']):
            continue
        indices = np.asarray([idx for idx, (source, target) in enumerate(trace['s_t']) if source != target])
        if len(indices) <= 0:
            continue
        for key, item in trace.items():
            if isinstance(item, list) and key != 'root_cause' and key != 'fault_type':
                trace[key] = np.asarray(item)[indices]
        service_idx = np.asarray(list(map(pair2index, (trace['s_t']))))
        _service_mask[trace_idx, service_idx] = True
        # assert all(np.diff(trace['endtime']) <= 0), f'end time is not sorted: {trace["endtime"]}'

        if ENABLE_ALL_FEATURES:
            _data[trace_idx, service_idx, 0] = np.asarray(trace['latency']) / 1e6
            _data[trace_idx, service_idx, 1] = np.asarray(trace['cpu_use']) / 100
            _data[trace_idx, service_idx, 2] = np.asarray([round(_, 2) for _ in trace['mem_use_percent']])
            _data[trace_idx, service_idx, 3] = np.asarray(trace['mem_use_amount']) / 1e9  # 1000M
            _data[trace_idx, service_idx, 4] = np.asarray(trace['file_write_rate']) / 1e8
            _data[trace_idx, service_idx, 5] = np.asarray(trace['file_read_rate']) / 1e8
            _data[trace_idx, service_idx, 6] = np.asarray(trace['net_send_rate']) / 1e8
            _data[trace_idx, service_idx, 7] = np.asarray(trace['net_receive_rate']) / 1e8
            _data[trace_idx, service_idx, 8] = list(map(lambda x: x // 100 if x != 0 else 9, (trace['http_status'])))
        else:
            _data[trace_idx, service_idx, 0] = np.asarray(trace['latency'])
            _data[trace_idx, service_idx, 1] = list(map(lambda x: int(x) // 100 if x != 0 else 9, (trace['http_status'])))

        _labels[trace_idx] = trace['label']
        _trace_ids[trace_idx] = trace['trace_id']
        _trace_root_causes = trace['root_cause'] if 'root_cause' in trace else []
        for _root_cause in _trace_root_causes:
            _root_causes[trace_idx, SERVICE2IDX[_root_cause]] = True
    _mask = np.tile(_service_mask[:, :, np.newaxis], (1, 1, 9))
    return _data, _labels, _mask, _trace_ids, _root_causes



def trace_encoding(input_data: pd.DataFrame, drop_service=0, drop_fault_type=0):
    try:
        with open("encoded_history_trace.pkl", 'rb') as fin:
            result = pickle.load(fin)
        logger.info("using encoded trace history data.")
        return result
    except:
        logger.info("encoded history trace doesn't exist, will generate one.")
    drop_service = list(INVOLVED_SERVICES)[:drop_service]
    drop_fault_type = list(FAULT_TYPES)[:drop_fault_type]
    data, labels, masks, trace_ids, root_causes = encoding_data(input_data, drop_service, drop_fault_type)
    # np.savez(
    #     output_file,
    #     data=data.reshape((len(data), -1)),
    #     labels=labels,
    #     masks=masks.reshape((len(data), -1)),
    #     trace_ids=trace_ids,
    #     root_causes=root_causes,
    # )
    res = {"data" : data.reshape((len(data), -1)), "labels" : labels,"masks" : masks.reshape((len(data), -1)) , "trace_ids" : trace_ids, "root_causes" : root_causes}
    with open("encoded_history_trace.pkl", 'wb') as fout:
        pickle.dump(res, fout)
    return res

