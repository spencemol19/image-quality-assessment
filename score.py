import subprocess
import statistics
import json
import time
import sys
import os


def check_within_range(s, prev, threshold_val):
    return prev < s <= threshold_val


def get_predict_cli_str(weights_file='{pwd}/models/MobileNet/weights_mobilenet_technical_0.11.hdf5'):
    if '{pwd}' in weights_file:
        pwd = os.path.abspath(os.path.curdir)
        weights_file = weights_file.format(pwd=pwd)

    return ['./predict', '--docker-image', 'nima-cpu', '--base-model-name', 'MobileNet', '--weights-file', weights_file,
            '--image-source', '']


def main(args):
    threshold_vals = []

    # check for threshold(s)
    if '-t' in args:
        threshold_ind = args.index('-t')
        threshold_vals = sorted([float(v) for v in args[threshold_ind + 1:]])
        args = args[:threshold_ind]

    if args[0].lower().replace('-', '') == 'f':
        cli_exec_str = get_predict_cli_str()

        for dir in args[1:]:
            if os.path.isdir(dir):
                start_time = time.time()

                cli_exec_str[-1] = dir

                raw_output = subprocess.check_output(cli_exec_str).decode('utf-8', errors='ignore')
                data_segment = raw_output.split('step')[1]
                output = json.loads(data_segment)

                aggr_scores = [s['mean_score_prediction'] for s in output]

                end_time = time.time()

                print('\n%s\n\nprocessed %d in %f s for "%s" data' % (
                    '-' * 60, len(aggr_scores), end_time - start_time, os.path.basename(dir)))

                total_valid = []

                for idx, threshold_val in enumerate(threshold_vals):
                    prev = 0 if idx == 0 else threshold_vals[idx - 1]
                    valid = [s for s in aggr_scores if check_within_range(s, prev, threshold_val)]
                    total_valid.extend(valid)
                    print('\nFound %d/%d in [%f, %f]' % (len(valid), len(aggr_scores), prev, threshold_val))

                neg_outliers = [s for s in aggr_scores if s < 0]

                pos_outliers = [s for s in aggr_scores if s > threshold_vals[-1]]

                print('\nFound %d/%d negative outliers' % (
                    len(neg_outliers), len(aggr_scores)))

                print('\nFound %d/%d positive outliers' % (
                    len(pos_outliers), len(aggr_scores)))

                print('\nMean: %f' % (sum(aggr_scores) / len(aggr_scores)))
                print('\nStandard deviation: %f\n' % statistics.stdev(aggr_scores))


if __name__ == '__main__':
    main(sys.argv[1:])
