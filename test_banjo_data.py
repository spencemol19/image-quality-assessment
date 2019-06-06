import subprocess
import time
import sys
import os

DATA_PATH = './banjo_data'

THRESHOLDS = [3, 5, 7, 10]

MAX_DEPTH_DATA_SET_DIRS = 3

DATA_DIRS_CMD = 'find ./banjo_data/* -mindepth 0 -maxdepth {max_depth} -type d'.format(
    max_depth=MAX_DEPTH_DATA_SET_DIRS)


def main(args):
    exec_base = ['python', 'score.py', '-f']

    data_dirs = subprocess.check_output(DATA_DIRS_CMD, shell=True).decode('utf-8', errors='ignore').splitlines()

    dir_names = [os.path.abspath(_dir) for _dir in data_dirs]

    exec_base.extend(dir_names)

    # if gpu spec flag is passed in, forward it
    if '-g' in [a.lower() for a in args]:
        exec_base.append('-g')

    # if aesthetic iqa flag is passed in, forward it
    if '-a' in [a.lower() for a in args]:
        exec_base.append('-a')

    exec_base.append('-t')

    exec_base.extend([str(t) for t in THRESHOLDS])

    out_file = './banjo-test/banjo_data_model_performance.%d.txt' % int(time.time())

    print(exec_base)

    testing_output = subprocess.check_output(exec_base).decode('utf-8', errors='ignore').strip()

    with open(out_file, 'w') as out_fi:
        out_fi.write(testing_output)


if __name__ == '__main__':
    main(sys.argv[1:])
