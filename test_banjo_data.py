import subprocess
import json
import time
import sys
import os

DATA_PATH = './banjo_data'

THRESHOLDS = [3, 5, 7, 10]

MAX_DEPTH_DATA_SET_DIRS = 3

DATA_DIRS_CMD = 'find ./banjo_data/* -mindepth 0 -maxdepth {max_depth} -type d'.format(
    max_depth=MAX_DEPTH_DATA_SET_DIRS)


def run_model_test_batch(args, data_dirs=None):
    exec_base = ['python', 'score.py', '-f']

    if not data_dirs:
        data_dirs = subprocess.check_output(DATA_DIRS_CMD, shell=True).decode('utf-8', errors='ignore').splitlines()

    dir_names = [os.path.abspath(_dir) for _dir in data_dirs]

    exec_base.extend(dir_names)

    lower_args = [a.lower() for a in args]

    is_silent = False

    # if aesthetic iqa flag is passed in, forward it
    if '-a' in lower_args:
        exec_base.append('-a')

    # if silent flag
    if '-s' in lower_args:
        is_silent = True
        exec_base.append('-s')

    if '-c' in lower_args:
        base_idx = lower_args.index('-c')
        class_val = lower_args[base_idx + 1]
        if '-' not in class_val:
            exec_base.append('-c')
            exec_base.append(class_val)

    exec_base.append('-t')

    exec_base.extend([str(t) for t in THRESHOLDS])

    print('\nExecution basis: %s\n' % exec_base)

    if is_silent:
        subprocess.call(exec_base)
    else:
        out_file = './banjo-test/banjo_data_model_performance.%d.txt' % int(time.time())

        testing_output = subprocess.check_output(exec_base).decode('utf-8', errors='ignore').strip()

        with open(out_file, 'w') as out_fi:
            out_fi.write(testing_output)


def main(args):
    if '--config' in args:
        config_idx = args.index('--config')
        config_file = args[config_idx + 1]
        if os.path.isfile(config_file):
            with open(config_file) as config_src:
                raw_classes = json.load(config_src)
                for idx, _class in enumerate(raw_classes):
                    idx += 1
                    data_dirs = [os.path.join(DATA_PATH, d) for d in _class]
                    data_dirs = [d for d in data_dirs if os.path.isdir(d)]
                    c_args = args[:config_idx] + ['-s', '-c', str(idx)]
                    run_model_test_batch(c_args, data_dirs)

    else:
        run_model_test_batch(args)


if __name__ == '__main__':
    main(sys.argv[1:])
