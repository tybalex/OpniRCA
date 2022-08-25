# Standard Library
import logging
import time

# Third Party
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Local
from rca_config import FEATURE_NAMES
from utils import read_pkl_file, write_pkl_file

DEBUG = True

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__file__)
logger.setLevel("DEBUG")
### python run_anomaly_detection_prepare_model.py -i dataframe/uninjection/3.pkl -t dataframe/uninjection-trace/3.pkl.npz -o model/history/3.pkl


def _extract_data(x):
    return x["data"], x["labels"], x["masks"], x["trace_ids"]


def train_history_model(trace_history_data, invo_history_data, cluster_id: str):
    history_model = read_pkl_file(cluster_id + "history_model.pkl")
    if history_model is not None:
        logger.info("using trained model.")
        return history_model
    else:
        logger.info("normal model doesn't exist, will train one.")

    his_data, his_labels, his_masks, his_trace_ids = _extract_data(trace_history_data)
    # rus = RandomUnderSampler(random_state=0)
    # X_resampled, y_resampled = rus.fit_resample(his_data, his_labels)

    result = {}
    for algorithm in ["RF-Trace", "MLP-Trace"]:
        if algorithm == "RF-Trace":
            model = RandomForestClassifier(n_estimators=100, n_jobs=10, verbose=0)
        elif algorithm == "MLP-Trace":
            model = MLPClassifier(
                batch_size=256,
                early_stopping=True,
                verbose=0,
                learning_rate_init=1e-4,
                max_iter=100,
                hidden_layer_sizes=(100, 100),
            )
        elif algorithm == "KNN-Trace":
            model = KNeighborsClassifier()
        else:
            raise RuntimeError()
        model.fit(his_data, his_labels)
        result[algorithm] = model

    invo_history = invo_history_data.set_index(
        keys=["source", "target"], drop=False
    ).sort_index()
    indices = np.unique(invo_history.index.values)
    for source, target in indices:
        reference = invo_history.loc[(source, target), FEATURE_NAMES].values
        token = f"IF-{source}-{target}"
        model = IsolationForest(contamination=0.01, n_jobs=10)
        model.fit(reference)
        result[token] = model

    for source, target in indices:
        for feature in FEATURE_NAMES:
            reference = invo_history.loc[(source, target), feature].values
            token = f"reference-{source}-{target}-{feature}-mean-variance"
            result[token] = {
                "mean": np.mean(reference[:]),
                "std": np.maximum(np.std(reference[:]), 0.1),
            }

    write_pkl_file(cluster_id + "history_model.pkl", result)
    return result


def _anomaly_detection_isolation_forest(df, result_column, cache):
    indices = np.unique(df.index.values)
    for source, target in indices:
        empirical = df.loc[(source, target), FEATURE_NAMES].values
        # reference = history.loc[(source, target), FEATURE_NAMES].values
        token = f"IF-{source}-{target}"
        if token not in cache:
            df.loc[(source, target), result_column] = 0
            continue
        model = cache[token]
        predict = model.predict(empirical)
        df.loc[(source, target), result_column] = predict
    return df


def _anomaly_detection_3sigma_without_useful_features(
    df, result_column, cache, threshold
):
    indices = np.unique(df.index.values)
    useful_feature = {key: FEATURE_NAMES for key in indices}
    return _anomaly_detection_3sigma(
        df, result_column, useful_feature, cache, threshold
    )


def _anomaly_detection_3sigma(df, result_column, useful_feature, cache, threshold):
    indices = np.unique(df.index.values)
    for source, target in indices:
        if (source, target) not in useful_feature:  # all features are not useful
            df.loc[(source, target), result_column] = 0
            continue
        features = useful_feature[(source, target)]
        empirical = df.loc[(source, target), features].values
        mean, std = [], []
        for idx, feature in enumerate(features):
            token = f"reference-{source}-{target}-{feature}-mean-variance"
            if token in cache:
                mean.append(cache[token]["mean"])
                std.append(cache[token]["std"])
            else:
                mean.append(np.mean(empirical, axis=0)[idx])
                std.append(np.maximum(np.std(empirical, axis=0)[idx], 0.1))
        mean = np.asarray(mean)
        std = np.asarray(std)
        predict = np.zeros(empirical.shape)
        for idx, feature in enumerate(features):
            predict[:, idx] = (
                np.abs(empirical[:, idx] - mean[idx]) > threshold * std[idx]
            )
        predict = np.max(predict, axis=1)

        df.loc[(source, target), result_column] = predict
    return df


def span_anomaly_detection_main(
    input_data, useful_feature: dict, history_model, cluster_id: str, main_threshold=1
):
    threshold = main_threshold

    cache = history_model
    df = input_data.set_index(keys=["source", "target"], drop=False).sort_index()
    tic = time.time()
    df = _anomaly_detection_3sigma(df, "Ours-predict", useful_feature, cache, threshold)
    toc = time.time()
    print("algo:", "ours", "time:", toc - tic, "invos:", len(df))

    df = _anomaly_detection_3sigma_without_useful_features(
        df, "NoSelection-predict", cache, threshold
    )

    # tic = time.time()
    df = _anomaly_detection_isolation_forest(df, "IF-predict", cache=cache)
    # toc = time.time()
    # print("algo:", "IF", "time:", toc - tic, 'invos:', len(df))

    df["predict"] = df["Ours-predict"]
    logger.info(len(df[df["predict"] == df["trace_label"]]) / len(df))
    return df
