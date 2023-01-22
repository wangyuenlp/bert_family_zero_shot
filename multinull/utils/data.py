import torch
from torch.utils.data import Dataset
import json
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler,WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import random
import copy
from datasets import load_dataset
   

class textclassificationDataset(Dataset):

    def __init__(self,max_len,tokenizer,train,args,datasets):
        self.datasets=datasets
        self.max_len=max_len
        self.tokenizer = tokenizer
        self.data =[]
        self.label=[] 
        self.mask_position=[]
        self.load_data(train,args)

    def __getitem__(self, idx):

        text = self.data[idx]
        label= self.label[idx]
        

        mask_position= self.mask_position[idx]


        return {"src_ids": text["input_ids"],
                            "src_mask":text["attention_mask"],
                            "label":label,
                            "mask_position":mask_position}


    def __len__(self):
        return len(self.data)
    
    def load_data(self,train,args):
        
        task2key={
            'dbpedia_14':'content',
            'amazon_polarity':'content',
            'ag_news':'text',
            'imdb':'text',
            'yelp_review_full':'text',
            'emotion':'text',
            'yahoo_answers_topics':'question_content'
        }


        dataset = self.datasets["test"]
        data_index_list=[i for i in range(dataset.num_rows)]
        if args.model_type=='bert':
            eos_token_id=self.tokenizer.sep_token_id
        else:
            eos_token_id=self.tokenizer.eos_token_id
        if args.mask_token_id:
            mask_token_id=args.mask_token_id
        else:
            mask_token_id=self.tokenizer.mask_token_id

        for i in data_index_list:
            if args.randommasknumber>0:
                
                text=self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(dataset[i][task2key[args.task]]))
                text_len=len(text)
                if len(text)+2+args.randommasknumber>args.max_len:
                    text=text[:args.max_len-args.randommasknumber-2]
                index=[j for j in range(len(text)-1)]
                random.shuffle(index)
                index=index[:args.randommasknumber]
                index.sort()
                new_text=[]
                last_index=0
                for item in index:
                    new_text+=text[last_index:item+1]
                    new_text+=[mask_token_id]
                    last_index=item+1
                new_text+=text[last_index:]
                new_text+=[mask_token_id]*(args.randommasknumber-text_len+1)
                new_text=[self.tokenizer.bos_token_id]+new_text+[eos_token_id]
                text={"input_ids":[],"attention_mask":[]}
                text["input_ids"]=new_text+[self.tokenizer.pad_token_id]*(self.max_len-len(new_text))
                text["attention_mask"]=[1]*(len(new_text))+[0]*(self.max_len-len(new_text))




            else:
                text=self.tokenizer(dataset[i][task2key[args.task]], pad_to_max_length=True, truncation=True,max_length=self.max_len)
                eos_position=text["input_ids"].index(eos_token_id)
                if args.startmasknumber>0:
                    if eos_position==args.max_len-1:
                        text["input_ids"][args.startmasknumber+1:eos_position]=text["input_ids"][1:eos_position-args.startmasknumber]

                        text["input_ids"][1:args.startmasknumber+1]=[mask_token_id for i in range(args.startmasknumber)]

                    elif eos_position<=args.max_len-1-args.startmasknumber:
                        text["input_ids"][args.startmasknumber+1:eos_position+args.startmasknumber+1]=text["input_ids"][1:eos_position+1]

                        text["input_ids"][1:args.startmasknumber+1]=[mask_token_id for i in range(args.startmasknumber)]

                        text["attention_mask"][eos_position+1:eos_position+args.startmasknumber+1]=[1 for i in range(args.startmasknumber)]

                    else:
                        text["input_ids"][args.startmasknumber+1:args.max_len-1]=text["input_ids"][1:args.max_len-args.startmasknumber-1]
                        text["input_ids"][args.max_len-1]=eos_token_id

                        text["input_ids"][1:args.startmasknumber+1]=[mask_token_id for i in range(args.startmasknumber)]
                      
                        text["attention_mask"]=[1 for i in range(args.max_len)]
                eos_position=text["input_ids"].index(eos_token_id)

                if args.endmasknumber>0:
                    if eos_position==args.max_len-1:
                        text["input_ids"][eos_position-args.endmasknumber:eos_position]=[mask_token_id for i in range(args.endmasknumber)]

                    elif eos_position<=args.max_len-1-args.endmasknumber:
                        text["input_ids"][eos_position:eos_position+args.endmasknumber]=[mask_token_id for i in range(args.endmasknumber)]

                        text["input_ids"][eos_position+args.endmasknumber]=eos_token_id
                        text["attention_mask"][eos_position+1:eos_position+args.endmasknumber+1]=[1 for i in range(args.endmasknumber)]
                    else:
                        text["input_ids"][args.max_len-1-args.endmasknumber:args.max_len-1]=[mask_token_id for i in range(args.endmasknumber)]

                        text["input_ids"][args.max_len-1]=eos_token_id
                        text["attention_mask"]=[1 for i in range(args.max_len)]

            eos_position=text["input_ids"].index(eos_token_id)
                
            if args.task!='yahoo_answers_topics':
                labels=dataset[i]['label']
            else:
                labels=dataset[i]['topic']
            self.data.append(text)
            self.label.append(labels)
            mask=[]
            if args.mask_token_id:
                for i in range(args.startmasknumber):
                    mask.append(i+1)
                for i in range(args.endmasknumber):
                    mask.append(eos_position-args.endmasknumber+i)
            else:
                for i in range(args.max_len):
                    if text["input_ids"][i]==mask_token_id:
                        mask.append(i)


            self.mask_position.append(mask)






def get_dataloader(dataset, tokenizer, args, split='train'):


    
    def MASKSentimentcollate_fn(batch):
        """
        Modify target_id as label
        """

        src_ids = torch.tensor([example['src_ids'] for example in batch], dtype=torch.long)
        src_mask = torch.tensor([example['src_mask'] for example in batch], dtype=torch.long)
        label = torch.tensor([example['label'] for example in batch], dtype=torch.long)
        mask_position=torch.tensor([example['mask_position'] for example in batch], dtype=torch.long)

        
        return {"src_ids": src_ids,
                "src_mask":src_mask,
                "label":label,
                "mask_position":mask_position}


    if split == 'train':
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        batch_size = args.train_batch_size
        sampler = RandomSampler(dataset if args.local_rank == -1 else DistributedSampler(dataset)) # SequentialSampler(dataset)
    elif split == 'valid':
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)    
        batch_size = args.eval_batch_size
        sampler = SequentialSampler(dataset)



    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, collate_fn=MASKSentimentcollate_fn)



    return dataloader, args