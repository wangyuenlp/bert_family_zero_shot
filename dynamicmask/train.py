import argparse
import glob
import os
import pickle
import random
import re
import json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertConfig,BertForMaskedLM,BertTokenizer,RobertaConfig,RobertaForMaskedLM,RobertaTokenizer,AlbertConfig,AlbertForMaskedLM,AlbertTokenizer
import math
import argparse
from torch.nn import CrossEntropyLoss
from datasets import load_dataset
import pandas as pd

class ArgsParser(object):
     
     def __init__(self):

          parser = argparse.ArgumentParser()
          
          parser.add_argument("--mask_token_id", default=None, type=int, required=False,
                              help="to select the token id to fill the answer slot[Y], which can not be conbined with randommasknumber")          
          parser.add_argument("--ensemble_method", default='average', type=str, required=False,
                              help="the method to ensemble the logits of multiple tokens")
          parser.add_argument("--prediction_position", default=0, type=int, required=False,
                              help="choose the position of token to predict only required when using single ensemble_method")
          parser.add_argument("--similarwords", default=1, type=int, required=False,
                              help="the number of similar words found by cosine similarity")
          parser.add_argument("--start", default=0, type=int, required=False,
                              help="the number of mask token in the start of the text")
          parser.add_argument("--end", default=0, type=int, required=False,
                              help="the number of mask token in the end of the text")
          parser.add_argument("--maskratio", default=0, type=float, required=False,
                              help="the number of mask token in the end of the text")          
          parser.add_argument("--task", default=None, type=str, required=False,
                              help="selected in : [imdb,ag_news,amazon_polarity,dbpedia_14,yelp_review_full,emotion,yahoo_answers_topics]")
          parser.add_argument("--model_type", default=None, type=str, required=True,
                              help="Model type selected in the list: [roberta] ")
          parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                              help="Path to pre-trained model or shortcut name selected in the list:")
          parser.add_argument("--config_name", default=None, type=str,
                              help="Pretrained config name or path if not the same as model_name")
          parser.add_argument("--tokenizer_name", default=None, type=str,
                              help="Pretrained tokenizer name or path if not the same as model_name")
          parser.add_argument("--cache_dir", default=None, type=str,
                              help="Where do you want to store the pre-trained models downloaded from s3")
          parser.add_argument("--max_len", default=512, type=int,
                              help="The maximum total encoder sequence length."
                                   "Longer than this will be truncated, and sequences shorter than this will be padded.")
          parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                              help="Batch size per GPU/CPU for training.")
          parser.add_argument("--no_cuda", action='store_true',
                              help="Whether not to use CUDA when available")
          parser.add_argument('--seed', type=int, default=42,
                              help="random seed for initialization")




          self.parser = parser

     def parse(self):
          args = self.parser.parse_args()
          return args




MODEL_CLASSES = {
    "bert": (BertConfig,BertForMaskedLM,BertTokenizer),
    "roberta": (RobertaConfig,RobertaForMaskedLM,RobertaTokenizer),
    "albert": (AlbertConfig,AlbertForMaskedLM,AlbertTokenizer),
}   


def prepare(args):






    if args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  
        device = torch.device("cuda")
        args.n_gpu = 1
    args.device = device



    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    print("Training/evaluation parameters {}".format(args))







def get_model_tokenizer(args):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    if args.config_name:
        config = config_class.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        config = config_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        config = config_class()

    if args.tokenizer_name:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new {} tokenizer. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name".format(tokenizer_class.__name__)
        )

    if args.model_name_or_path:
        model = model_class.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                cache_dir=args.cache_dir
            )
    else:
        print("Training new model from scratch")
        model = model_class(config=config)



    model.to(args.device)
    

    return model, tokenizer, model_class, tokenizer_class, args



def test(datasets,args, model, tokenizer, Verbalizer,prefix=""):



    # Eval!
    print("***** Running evaluation {} *****".format(prefix))
    print("  Num examples = {}".format(datasets.num_rows))
    print("  Batch size = {}".format(args.per_gpu_eval_batch_size))
    model.eval()
    eval_acc=0

    task2key={
            'dbpedia_14':'content',
            'amazon_polarity':'content',
            'ag_news':'text',
            'imdb':'text',
        }

    for i in trange(datasets.num_rows):
        with torch.no_grad():                        
            
            input_ids=tokenizer.convert_tokens_to_ids(tokenizer.tokenize(datasets[i][task2key[args.task]]))[:args.max_len-2]

            mask_number=max(int(len(input_ids)*args.maskratio),1)
            input_ids=input_ids[:args.max_len-2-mask_number]

            if args.start and args.end:
                input_ids=[tokenizer.mask_token_id]*(mask_number//2)+input_ids+[tokenizer.mask_token_id]*(mask_number-mask_number//2)
            elif args.start:
                input_ids=[tokenizer.mask_token_id]*mask_number+input_ids
            else:
                input_ids=input_ids+[tokenizer.mask_token_id]*mask_number

            input_ids=[tokenizer.bos_token_id]+input_ids+[tokenizer.eos_token_id]

            outputs=model(torch.tensor([input_ids]).to(args.device))

            if args.start and args.end:
                predict_score=torch.cat([outputs.logits[:,1:mask_number//2+1],outputs.logits[:,mask_number//2-mask_number-1:-1]],dim=1)
            elif args.start:
                predict_score=outputs.logits[:,1:mask_number+1]
            else:
                predict_score=outputs.logits[:,-mask_number-1:-1]

            logits=torch.zeros(mask_number,len(Verbalizer)).type_as(outputs.logits)            

            for j in range(mask_number):                
                for k in range(len(Verbalizer)):
                
                    logits[j,k]=predict_score[:,j,Verbalizer[k]].sum(dim=1)/len(Verbalizer[k])

            logits=logits.sum(dim=0)
            predict_label=logits.argmax(0)
            
            eval_acc += (predict_label == datasets[i]['label']).sum().item()




    eval_acc/=datasets.num_rows

    result= eval_acc
    print("***** Test results *****")
    print('start:',args.start)
    print('end:',args.end)
    print('similar words:',args.similarwords)
    print('maskratio:',args.maskratio)    
    print("  {} = {}".format("test_acc", result))


    return result


def main():
    args = ArgsParser().parse()
    prepare(args)

    # Load pretrained model and tokenizer

    
    model, tokenizer, model_class, tokenizer_class, args = get_model_tokenizer(args)



    print("Training/evaluation parameters {}".format(args))
    datasets = load_dataset(args.task)['test']







    if args.task=='imdb' or args.task=='amazon_polarity':
        id2labels=['negative','positive']
    elif args.task=='ag_news':
        id2labels=['politics','sports','business','technology']
    elif args.task=='dbpedia_14':
        id2labels=['company','school','artist','athlete','politics','transportation','building','river','village','animal','plant','album','film','book']

    
    embeddings=model.get_input_embeddings()
    labelname_ids=[]
    for labelname in id2labels:
        if len(tokenizer.tokenize(' ' + labelname)) == 1:
                labelname_ids.append(tokenizer._convert_token_to_id(tokenizer.tokenize(' ' + labelname)[0]))
        else:
            raise ValueError("please check whether the label word is splited to subword.")
    
    
    
    Verbalizer=[]
    for labelnameid in labelname_ids:
        cos=torch.cosine_similarity(embeddings.weight[labelnameid].clone().detach().repeat(embeddings.weight.size(0),1),embeddings.weight,dim=1)
        indices=torch.topk(cos,args.similarwords).indices
        similarword_id=[]
        for item in indices:
            similarword_id.append(item)
        Verbalizer.append(similarword_id)


    results={'task':[],'start':[],'end':[],'maskratio':[],'accuracy':[]}

    for i in range(6):
        args.maskratio=0.05*(i+1)
        result=test(datasets,args, model, tokenizer,Verbalizer)    
        results['task'].append(args.task)
        results['start'].append(args.start)
        results['end'].append(args.end)
        results['maskratio'].append(args.maskratio)
        results['accuracy'].append(result)
    
    results=pd.DataFrame(results)
    results.to_csv('results.csv',mode='a',header=True,index=None)


if __name__ == "__main__":
    main()
