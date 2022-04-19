import argparse
from typing import List
import pandas as pd
import numpy as np
import math
import os
from datetime import datetime
from statistics import mean


from TPAgentUtil import TPAgentUtil
from PairWiseEnv import CIPairWiseEnv
from TPPairWiseDQNAgent import TPPairWiseDQNAgent
from ci_cycle import CICycleLog
from Config import Config
from TestcaseExecutionDataLoader import TestCaseExecutionDataLoader
from CustomCallback import  CustomCallback
from stable_baselines.bench import Monitor
from pathlib import Path
from CIListWiseEnvMultiAction import CIListWiseEnvMultiAction
from CIListWiseEnv import CIListWiseEnv
from PointWiseEnv import CIPointWiseEnv
import sys

def millis_interval(start, end):
    """start and end are datetime instances"""
    diff = end - start
    millis = diff.days * 24 * 60 * 60 * 1000
    millis += diff.seconds * 1000
    millis += diff.microseconds / 1000
    return millis


## find the cycle with maximum number of test cases
def get_max_test_cases_count(cycle_logs:list):
    max_test_cases_count = 0
    for cycle_log in cycle_logs:
        if cycle_log.get_test_cases_count()>max_test_cases_count:
            max_test_cases_count = cycle_log.get_test_cases_count()
    return max_test_cases_count


def experiment(mode, algo, test_case_data, start_cycle, end_cycle, episodes, model_path, dataset_name, conf,verbos=False):
    log_dir = os.path.dirname(conf.log_file)
#    -- fix end cycle issue
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if start_cycle <= 0:
        start_cycle = 0

    if end_cycle >= len(test_case_data)-1:
        end_cycle = len(test_case_data)
    ## check for max cycle and end_cycle and set end_cycle to max if it is larger than max
    log_file = open(conf.log_file, "a")
    # Create sorted test case log file
    log_file_test_cases = open(log_dir+"/sorted_test_case.csv", "a")
    log_file.write("timestamp,mode,algo,model_name,episodes,steps,cycle_id,training_time,testing_time,winsize,test_cases,failed_test_cases, apfd, nrpa, random_apfd, optimal_apfd" + os.linesep)

    first_round: bool = True

    if start_cycle > 0:
        first_round = False
        #previous model exists
        previous_model_path = model_path + "/" + mode + "_" + algo + dataset_name + "_" + str(0) + "_" + str(start_cycle-1)

    model_save_path = None
    apfds=[]
    nrpas=[]
    # end_cycle is the length of the test case data or lower
    # every test case data contains set of test cases 
    for i in range(start_cycle, end_cycle - 1):
        # If test cases are less than 6
        # If dataset type is "simple" and failed test cases are 0
        # Then just return to new cycle without performing any actions
        if (test_case_data[i].get_test_cases_count() < 6) or \
                ( (conf.dataset_type == "simple") and
                  (test_case_data[i].get_failed_test_cases_count() < 1)):
            continue
        # If chosen mode is pairwise then
        # N => total cases in the current test_case_data
        # Toal steps depend upon number of episodes and test case count
        elif mode.upper() == 'POINTWISE':
            N = test_case_data[i].get_test_cases_count()
            steps = int(episodes * (N * (math.log(N,2)+1)))
            env = CIPointWiseEnv(test_case_data[i], conf)
        print("Training agent with replaying of cycle " + str(i) + " with steps " + str(steps))

        # Save path of current model to previous_model_path to be used for next cycle
        if model_save_path:
            previous_model_path = model_save_path

        # Create the save path for current model
        model_save_path = model_path + "/" + mode + "_" + algo + dataset_name + "_" + str(
            start_cycle) + "_" + str(i)
        
        # Save the model here
        env = Monitor(env, model_save_path +"_monitor.csv")
        callback_class = CustomCallback(svae_path=model_save_path,
                                        check_freq=int(steps/episodes), log_dir=log_dir, verbose=verbos)


        # If it is first round of training then
        if first_round:
            # Create the new TP Agent
            tp_agent = TPAgentUtil.create_model(algo, env)
            training_start_time = datetime.now()
            # Learn the TP Agent
            tp_agent.learn(total_timesteps=steps, reset_num_timesteps=True, callback=callback_class)
            training_end_time = datetime.now()
            first_round = False
        else:
            # If it is not first round then load the model of the previous cycle
            tp_agent = TPAgentUtil.load_model(algo=algo, env=env, path=previous_model_path+".zip")
            training_start_time = datetime.now()
            # Make the previous model learn
            tp_agent.learn(total_timesteps=steps, reset_num_timesteps=True, callback=callback_class)
            training_end_time = datetime.now()
        print("Training agent with replaying of cycle " + str(i) + " is finished")

        # Initialize j to have current_cycle_count + 1
        # j then will increment until it reaches end_cycle
        j = i+1 ## test trained agent on next cycles

        # Repeat until
        # 1. Number of test cases is less than 6 , or,
        # 2. Dataset_type  is "simple" and failed test cases are 0 and j<end_cycle
        while (((test_case_data[j].get_test_cases_count() < 6)
               or ((conf.dataset_type == "simple") and (test_case_data[j].get_failed_test_cases_count() == 0) ))
               and (j < end_cycle)):
            #or test_case_data[j].get_failed_test_cases_count() == 0) \
            j = j+1
        
        # If j exceeds end _cycle then quit the loop
        if j >= end_cycle-1:
            break
    
        if mode.upper() == 'POINTWISE':
            env_test = CIPointWiseEnv(test_case_data[j], conf)

        test_time_start = datetime.now()
        # Initialize and train test agent
        test_case_vector = TPAgentUtil.test_agent(env=env_test, algo=algo, model_path=model_save_path+".zip", mode=mode)
        test_time_end = datetime.now()
        test_case_id_vector = []

        # Save the information in test_case_id_vector
        for test_case in test_case_vector:
            test_case_id_vector.append(str(test_case['test_id']))
            cycle_id_text = test_case['cycle_id']

        # If not test cases failed then make the apfd data for current test cases
        if test_case_data[j].get_failed_test_cases_count() != 0:
            apfd = test_case_data[j].calc_APFD_ordered_vector(test_case_vector)
            apfd_optimal = test_case_data[j].calc_optimal_APFD()
            apfd_random = test_case_data[j].calc_random_APFD()
            # Only stores ordered APFD vector
            apfds.append(apfd)
        # Else assign 0 to all apfd
        else:
            apfd =0
            apfd_optimal =0
            apfd_random =0

        # Calculate NRPA vector
        nrpa = test_case_data[j].calc_NRPA_vector(test_case_vector)
        nrpas.append(nrpa)


        test_time = millis_interval(test_time_start,test_time_end)
        training_time = millis_interval(training_start_time,training_end_time)
        print("Testing agent  on cycle " + str(j) +
              " resulted in APFD: " + str(apfd) +
              " , NRPA: " + str(nrpa) +
              " , optimal APFD: " + str(apfd_optimal) +
              " , random APFD: " + str(apfd_random) +
              " , # failed test cases: " + str(test_case_data[j].get_failed_test_cases_count()) +
              " , # test cases: " + str(test_case_data[j].get_test_cases_count()), flush=True)
        log_file.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + "," + mode + "," + algo + ","
                       + Path(model_save_path).stem + "," +
                       str(episodes) + "," + str(steps) + "," + str(cycle_id_text) + "," + str(training_time) +
                       "," + str(test_time) + "," + str(conf.win_size) + "," +
                       str(test_case_data[j].get_test_cases_count()) + "," +
                       str(test_case_data[j].get_failed_test_cases_count()) + "," + str(apfd) + "," +
                       str(nrpa) + "," + str(apfd_random) + "," + str(apfd_optimal) + os.linesep)
        log_file_test_cases.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + "," + mode + "," + algo + ","
                       + Path(model_save_path).stem + "," +
                       str(episodes) + "," + str(steps) + "," + str(cycle_id_text) + "," + str(training_time) +
                       "," + str(test_time) + "," + str(conf.win_size) + "," +
                                  ('|'.join(test_case_id_vector)) + os.linesep)
        
        # IF there apfd list is not null
        if (len(apfds)):
            print(f"avrage apfd so far is {mean(apfds)}")
        print(f"avrage nrpas so far is {mean(nrpas)}")

        log_file.flush()
        log_file_test_cases.flush()
    log_file.close()
    log_file_test_cases.close()

def reportDatasetInfo(test_case_data:list):
    cycle_cnt = 0
    failed_test_case_cnt = 0
    test_case_cnt = 0
    failed_cycle = 0

    # For every set of test cases in test_case_data
    for cycle in test_case_data:
        # If test cases are greater than 5 then 
        # Increase the possible cycle count
        # count total test cases obtained so far 
        # Count total failed test cases
        # If there failed test cases in current cycle then 
        #       increase failed cycle count by 1
        if cycle.get_test_cases_count() > 5:
            cycle_cnt = cycle_cnt+1
            test_case_cnt = test_case_cnt + cycle.get_test_cases_count()
            failed_test_case_cnt = failed_test_case_cnt+cycle.get_failed_test_cases_count()
            if cycle.get_failed_test_cases_count() > 0:
                failed_cycle = failed_cycle + 1

    # Report the analysed data of dataset
    print(f"# of cycle: {cycle_cnt}, # of test case: {test_case_cnt}, # of failed test case: {failed_test_case_cnt}, "
          f" failure rate:{failed_test_case_cnt/test_case_cnt}, # failed test cycle: {failed_cycle}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DNN debugger')
    old_limit = sys.getrecursionlimit()
    print("Recursion limit:" + str(old_limit))
    sys.setrecursionlimit(1000000)
    # parser.add_argument('--traningData',help='tranind data folder',required=False)
    parser.add_argument('-m', '--mode', help='[pairwise,pointwise,listwise] ', required=True)
    parser.add_argument('-a', '--algo', help='[a2c,dqn,..]', required=True)
    parser.add_argument('-d', '--dataset_type', help='simple, enriched', required=False, default="simple")
    parser.add_argument('-e', '--episodes', help='Training episodes ', required=True)
    parser.add_argument('-w', '--win_size', help='Windows size of the history', required=False)
    parser.add_argument('-t', '--train_data', help='Train set folder', required=True)
    parser.add_argument('-f', '--first_cycle', help='first cycle used for training', required=False)
    parser.add_argument('-c', '--cycle_count', help='Number of cycle used for training', required=False)
    parser.add_argument('-l', '--list_size', help='Maximum number of test case per cycle', required=False)
    parser.add_argument('-o', '--output_path', help='Output path of the agent model', required=False)


    # parser.add_argument('-f','--flags',help='Input csv file containing testing result',required=False)
    supported_formalization = ['POINTWISE']
    supported_algo = ['DQN']
    args = parser.parse_args()

    # Set up the configuration with dataset info
    conf = Config()
    conf.train_data = args.train_data
    conf.dataset_name = Path(args.train_data).stem
    if not args.win_size:
        conf.win_size = 10
    else:
        conf.win_size = int(args.win_size)
    if not args.first_cycle:
        conf.first_cycle = 0
    else:
        conf.first_cycle = int(args.first_cycle)
    if not args.cycle_count:
        conf.cycle_count = 9999999

    # Decide output path if not specified
    #"POINTWISE" = simple
    #args.algo = DQN
    #conf.dataset_name = paintcontrol_additional-features.csv
    #conf.win_size = 10
    
    if not args.output_path:
        conf.output_path = '../experiments/' + "POINTWISE" + "/" + args.algo + "/" + conf.dataset_name + "_" \
                           + str(conf.win_size) + "/"
        conf.log_file = conf.output_path + "POINTWISE" + "_" + args.algo + "_" + \
                        conf.dataset_name + "_" + args.episodes + "_" + str(conf.win_size) + "_log.txt"
    else:
        conf.output_path = args. output_path + "/" + "POINTWISE" + "/" + args.algo + "/" + conf.dataset_name + "_" \
                           + str(conf.win_size) + "/"
        conf.log_file = conf.output_path + "POINTWISE" + "_" + args.algo + "_" + \
                        conf.dataset_name + "_" + args.episodes + "_" + str(conf.win_size) + "_log.txt"

# Test case data
test_data_loader = TestCaseExecutionDataLoader(conf.train_data, args.dataset_type)
test_data = test_data_loader.load_data()
ci_cycle_logs = test_data_loader.pre_process()

### load data
reportDatasetInfo(test_case_data=ci_cycle_logs)

#training using n cycle staring from first cycle
conf.dataset_type = args.dataset_type 
experiment(mode="POINTWISE", algo=args.algo.upper(), test_case_data=ci_cycle_logs, episodes=int(args.episodes),
           start_cycle=conf.first_cycle, verbos=False,
           end_cycle=conf.first_cycle + conf.cycle_count - 1, model_path=conf.output_path, dataset_name="", conf=conf)
