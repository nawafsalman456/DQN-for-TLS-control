import subprocess
import os
import sys

tests_dir = f"{os.environ.get('PROJECT_ROOT')}\\verif\\tests"
sys.path.append(tests_dir)

TESTS_LIST = [
    "static_try_sim_test.py",
    "random_try_sim_test.py"
]

# TODO:
# make it in parallel.
# change to sumo (without gui) here anyway.
# print results of the best algorithm.
# suppress stdout ? allow only stderr ?
for test in TESTS_LIST:
    subprocess.run(['python', f"{tests_dir}\{test}"])

