import subprocess
import time
import os

DATA_PATH = './banjo_data'

THRESHOLDS = [3, 5, 7]


def main():
    exec_base = ['python', 'score.py', '-f']

    dir_names = [os.path.join(os.path.abspath(DATA_PATH), _dir) for _dir in os.listdir(DATA_PATH)]
    exec_base.extend(dir_names)

    exec_base.append('-t')

    exec_base.extend([str(t) for t in THRESHOLDS])

    exec_base.extend(['>', 'banjo_data_model_performance.%d.txt' % int(time.time())])

    subprocess.call(exec_base)


if __name__ == '__main__':
    main()
