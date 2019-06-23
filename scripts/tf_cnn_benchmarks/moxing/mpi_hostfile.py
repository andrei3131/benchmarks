#!/usr/bin/env python

from __future__ import print_function
import os
import argparse
import time

# Args
parser = argparse.ArgumentParser(prog='mpi_hostfile', description='MPI hostfile configuration')
parser.add_argument('-s', '--slots', type=int, default=1, help='Each host slots cnt')
parser.add_argument('-q', '--quiet', action='store_true', help='Quiet mode')
parser.add_argument('-f', '--file', type=str, help='hostfile output path')
args, unparsed = parser.parse_known_args()

# Batch inject env
DLS_TASK_INDEX = os.env['DLS_TASK_INDEX']
BATCH_WORKER_HOSTS = "BATCH_CUSTOM%s_HOSTS" % DLS_TASK_INDEX

# hostfile path
# The hostfile should be placed with the rsh-agent.sh used by MPI
HOSTFILE_PATH = args.file

# Log
def log(text):
    if not args.quiet:
        print('INFO: %s' % text)


def err(text):
    print('ERROR: %s' % text)


def get_running_worker_pod_list():
    """find out all running worker pods
    """
    batch_job_id = os.env['BATCH_JOB_ID']
    get_pod_cmd = 'kubectl get po --no-headers -o custom-columns=:metadata.name -l batch-job-id=%s' \
                  '--field-selector=status.phase==Running' \
                  % batch_job_id

    # pass os env for kubectl access kube-apiserver with correct ip:port
    kubectl_rel = os.popen(get_pod_cmd)
    return_stdout = kubectl_rel.read().strip()
    return_code = kubectl_rel.close()
    if return_code is not None or return_stdout == "":
        log('Label %s: worker pods not found' % worker_label)
        return []

    return return_stdout.splitlines()


def gen_hostfile(host_list, slots_cnt):
    """Write mpi hostfile as
    worker-0 slots=1
    worker-1 slots=1
    ..."""
    with open(HOSTFILE_PATH, 'w') as out:
        line_pattern = '%s slots=%s\n'
        for host in host_list:
            out.write(line_pattern % (host, slots_cnt))


if __name__ == '__main__':
    """mpi_conf waits for workers become running and generates the hostfile for OpenMPI
    """

    if args.slots <= 0 or args.slots > 8:
        err('invalid OpenMPI slots')
        exit(1)

    slots = args.slots

    batch_worker_hosts = os.environ[BATCH_WORKER_HOSTS]
    if batch_worker_hosts == "":
        err('batch worker hosts not found')
        exit(1)

    expected_worker_num = len(batch_worker_hosts.split(','))

    hosts = []

    cur_times = 0
    try_times = 100000000000
    while cur_times < try_times:
        time.sleep(5)

        hosts = get_running_worker_pod_list()
        hosts_num = len(hosts)
        log('get running workers num: %d/%d' % (hosts_num, expected_worker_num))
        if len(hosts) == expected_worker_num:
            break
        else:
            cur_times += 1

    if not (cur_times < try_times):
        err('timeout for waiting the expected number of workers')
        exit(1)

    gen_hostfile(hosts, slots)
