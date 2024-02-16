import copy
import logging
import numpy as np
import os
import pandas as pd
import sys
import torch

from utils.data_manager import DataManager
from utils.toolkit import count_parameters
from RanPAC import Learner

def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])
    ave_accs=[]
    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        ave_acc=_train(args)
        ave_accs.append(ave_acc)
    return ave_accs


def _train(args):

    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"],args["dataset"], init_cls, args['increment'])
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)
    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        ' ',
        args["seed"],
        args["convnet_type"],
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.info('Starting new run')
    _set_random()
    _set_device(args)
    print_args(args)

    model = Learner(args)
    model.dil_init=False
    if args['dataset']=='core50':
        ds='core50_s1'
        dil_tasks=['s1','s2','s4','s5','s6','s8','s9','s11']
        num_tasks=len(dil_tasks)
        model.is_dil=True
    elif args['dataset']=='cddb':
        ds='cddb_gaugan'
        dil_tasks=['gaugan','biggan','wild','whichfaceisreal','san']
        num_tasks=len(dil_tasks)
        model.topk=2
        model.is_dil=True
    elif args['dataset']=='domainnet':
        ds='domainnet_real'
        dil_tasks=['real','quickdraw','painting','sketch','infograph','clipart']
        num_tasks=len(dil_tasks)
        model.is_dil=True
    else:
        #cil datasets
        model.is_dil=False
        data_manager = DataManager(
            args['dataset'],
            args["shuffle"],
            args["seed"],
            args["init_cls"],
            args["increment"],
            use_input_norm=args["use_input_norm"]
        )
        num_tasks=data_manager.nb_tasks

    acc_curve = {"top1_total": [],"ave_acc": []}
    classes_df=None
    logging.info("Pre-trained network parameters: {}".format(count_parameters(model._network)))
    cnn_matrix=[]
    for task in range(num_tasks):
        if model.is_dil:
            #reset the data manager to the next domain
            data_manager = DataManager(
                args["dataset"]+'_'+dil_tasks[task],
                args["shuffle"],
                args["seed"],
                args["init_cls"],
                args["increment"],
                use_input_norm=args["use_input_norm"]
            )
            model._cur_task=-1
            model._known_classes = 0
            model._classes_seen_so_far = 0
        if classes_df is None:
            classes_df=pd.DataFrame()
            classes_df['init']=-1*np.ones(data_manager._test_data.shape[0])
        model.incremental_train(data_manager)
        acc_total,acc_grouped,predicted_classes,true_classes = model.eval_task()
        col1='pred_task_'+str(task)
        col2='true_task_'+str(task)
        if args['do_not_save']==False:
            classes_df[col1]=np.pad(predicted_classes,(0,data_manager._test_data.shape[0]-len(predicted_classes)),'constant',constant_values=(-1,-1))
            classes_df[col2]=np.pad(true_classes,(0,data_manager._test_data.shape[0]-len(predicted_classes)),'constant',constant_values=(-1,-1))
        model.after_task()
        
        acc_curve["top1_total"].append(acc_total)
        acc_curve["ave_acc"].append(np.round(np.mean(list(acc_grouped.values())),2))
        if args['do_not_save']==False:
            save_results(args,acc_curve["top1_total"],acc_curve["ave_acc"],model,classes_df)

        logging.info("Group Accuracies after this task: {}".format(acc_grouped))
        logging.info("Ave Acc curve: {}".format(acc_curve["ave_acc"]))
        logging.info("Top1 curve: {}".format(acc_curve["top1_total"]))
        
    logging.info('Finishing run')
    logging.info('')
    return acc_curve["ave_acc"]

def save_results(args,top1_total,ave_acc,model,classes_df):
    if not os.path.exists('./results/'):
        os.makedirs('./results/')
    output_df=pd.DataFrame()
    output_df['top1_total']=top1_total
    output_df['ave_acc']=ave_acc
    output_df.to_csv('./results/'+args['dataset']+'_publish_'+str(args['ID'])+'.csv')

    if not os.path.exists('./results/class_preds/'):
        os.makedirs('./results/class_preds/')
    classes_df.to_csv('./results/class_preds/'+args['dataset']+'_class_preds_publish_'+str(args['ID'])+'.csv')

def _set_device(args):
    device_type = args["device"]
    gpus = []
    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))
        gpus.append(device)
    args["device"] = gpus

def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
