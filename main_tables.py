import pandas as pd
import argparse
import time
from trainer import train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str)
    a=parser.parse_args()

    datasets=['cifar224','imagenetr','imageneta','cub','omnibenchmark','vtab','cars','core50','cddb','domainnet']
    if a.d not in datasets:
        print('Dataset '+a.d+' not supported')
        return

    #data in Table 1 and Table 2
    if a.d in ['core50','cddb','domainnet']:
        IDS=[7,6,4,3,5,2]
        ID_names=['Algorithm 1','No RPs','No Phase 1','No RPs or Phase 1','NCM with Phase 1','NCM only','runtime']
    else:
        IDS=[0,7,6,4,3,5,2,1]
        ID_names=['Joint linear probe','Algorithm 1','No RPs','No Phase 1','No RPs or Phase 1','NCM with Phase 1','NCM only','Joint full fine-tuning','runtime']
    results=pd.DataFrame(columns=[a.d],index=ID_names)
    for d in [a.d]:
        t0=time.time()
        for idx,i in enumerate(IDS):
            exps=pd.read_csv('./args/'+d+'_publish.csv')
            args=exps[exps['ID']==i].to_dict('records')[0]
            args['seed']=[args['seed']]
            args['device']=[args['device']]
            args['do_not_save']=True
            ave_accs=train(args)
            results.at[ID_names[idx],d]=ave_accs[0][-1]
            results.at['runtime',d]=time.time()-t0
            results.to_csv('paper_tables/Table_data_main_'+d+'.csv')

    #data in Tables A6, A7 and A8
    IDS=[10,9,8,13,12,11,16,15,14]
    ID_names=['ResNet50, Phase 2','ResNet50, No RPs','ResNet50, NCM only']
    ID_names+=['ResNet152, Phase 2','ResNet152, No RPs','ResNet152, NCM only']
    ID_names+=['CLIP, Phase 2','CLIP, No RPs','CLIP, NCM only']
    results=pd.DataFrame(columns=[a.d],index=ID_names)
    for d in [a.d]:
        t0=time.time()
        for idx,i in enumerate(IDS):
            exps=pd.read_csv('./args/'+d+'_publish.csv')
            args=exps[exps['ID']==i].to_dict('records')[0]
            args['seed']=[args['seed']]
            args['device']=[args['device']]
            args['do_not_save']=True
            ave_accs=train(args)
            results.at[ID_names[idx],d]=ave_accs[0][-1]
            results.at['runtime',d]=time.time()-t0
            results.to_csv('paper_tables/Table_data_si_'+d+'.csv')

if __name__ == '__main__':
    main()