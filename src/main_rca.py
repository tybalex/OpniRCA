

from parse_traces import parse_traces

import pickle


def main():
	# get anomaly signal to trigger the rca 
    # with open("../data/test/admin-order_abort_1011.pkl", "rb") as fin:
    #     input_data = pickle.load(fin)
    #     for i in input_data:
    #         assert len(set(i["http_status"])) == 1
    #         print(i["http_status"])
    # with open("../A/uninjection/3.pkl", "rb") as fin:
    # 	input_data = pickle.load(fin)
    # 	for i in input_data:
    #         print(i["http_status"])
    parse_traces()


if __name__ == '__main__':
    main()