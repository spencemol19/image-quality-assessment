import subprocess
import time
import os

DATA_PATH = './banjo_data'

THRESHOLDS = [3, 5, 7, 10]


def main():
    exec_base = ['python', 'score.py', '-f']

    dir_names = [os.path.join(os.path.abspath(DATA_PATH), _dir) for _dir in os.listdir(DATA_PATH)]
    exec_base.extend(dir_names)

    exec_base.append('-t')

    exec_base.extend([str(t) for t in THRESHOLDS])

    out_file = './banjo-test/banjo_data_model_performance.%d.txt' % int(time.time())

    testing_output = subprocess.check_output(exec_base).decode('utf-8', errors='ignore').strip()

    with open(out_file, 'w') as out_fi:
        out_fi.write(testing_output)


if __name__ == '__main__':
    main()
