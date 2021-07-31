import transformers
import json
import sys
import os
import datasets
squad_dataset = datasets.load_dataset('squad')
from datasets import concatenate_datasets, load_dataset
from unittest.mock import patch
import run_qa
import run_qa_alt
covid_file = '../data/COVID-QA.json'
bio_file = '../bioASQ/bioASQ.json'
config_file_location = '../data/config.json'

def make_and_save_full_dataset(train, valid, test, path):
    full_data = datasets.dataset_dict.DatasetDict({'train':train, 'validation':valid, 'test': test})
    full_data.save_to_disk(path)

def get_dataset(filename):
    return datasets.load_dataset('custom_squad.py', data_files= {'train':filename})['train']

def update_config(checkpoint, file_location = '../data/config.json'):
    config_file = open(file_location, 'r')
    config = json.load(config_file)
    print(config)
    config["_name_or_path"] = checkpoint
    config_file = open(file_location, 'w')
    json.dump(config, config_file, indent= 2)
    config_file.close()


def train_adapter(lr):
    data_files = {}
    data_files["train"] = covid_file

    covid_qa = get_dataset(covid_file)
    bio_qa = get_dataset(bio_file)

    squad_qa = concatenate_datasets([squad_dataset['train'], squad_dataset['validation']])
    covid_and_squad_dataset_path = "../data/full_squad_covidQA"

    k_fold = 5
    for i in range(k_fold):
        covid_fold = covid_qa.shard(k_fold, i)

        covid_test = covid_fold.shard(2, 0)
        covid_val = covid_fold.shard(2, 1)
        covid_train = concatenate_datasets([covid_qa.shard(k_fold, j) for j in range(k_fold) if j != i])

        #make_and_save_full_dataset(covid_train, squad_qa, covid_val, covid_test, covid_and_squad_dataset_path)

        checkpoint = 'roberta-base'
        cur_dir = '../models/adapter_baseline_lr'+str(lr)+'/split_' + str(i)

        #log_file = open(cur_dir + "/log_file.txt","w+")
        #sys.stdout = log_file

        squad_qa.shuffle()
        bio_qa.shuffle()
        full_dataset = datasets.concatenate_datasets([bio_qa, covid_train])
        make_and_save_full_dataset(full_dataset, covid_val, covid_test, covid_and_squad_dataset_path)
        run_gradual_ft(cur_dir, checkpoint, covid_val, lr)


def train_adapter_base(lr):
    data_files = {}
    data_files["train"] = covid_file

    covid_qa = get_dataset(covid_file)
    bio_qa = get_dataset(bio_file)

    squad_qa = concatenate_datasets([squad_dataset['train'], squad_dataset['validation']])
    covid_and_squad_dataset_path = "../data/full_squad_covidQA"


    k_fold = 5
    for i in range(k_fold):
        covid_fold = covid_qa.shard(k_fold, i)

        covid_test = covid_fold.shard(2, 0)
        covid_val = covid_fold.shard(2, 1)
        covid_train = concatenate_datasets([covid_qa.shard(k_fold, j) for j in range(k_fold) if j != i])

        #make_and_save_full_dataset(covid_train, squad_qa, covid_val, covid_test, covid_and_squad_dataset_path)

        checkpoint = 'roberta-base'
        cur_dir = '../models/adapter_COVID_baseline_'+str(lr)+'/split_' + str(i)

        #log_file = open(cur_dir + "/log_file.txt","w+")
        #sys.stdout = log_file

        squad_qa.shuffle()
        bio_qa.shuffle()
        make_and_save_full_dataset(covid_train, covid_val, covid_test, covid_and_squad_dataset_path)
        run_gradual_ft(cur_dir, checkpoint, covid_val, lr)


def run_gradual_ft(output_dir, checkpoint, covid_val, lr, dataset_name,adapter_output):
    args = f"""
    run_qa_alt.py 
    --model_name_or_path={checkpoint}
    --dataset_name={dataset_name}
    --do_train
    --per_device_train_batch_size=40
    --per_device_eval_batch_size=40
    --evaluation_strategy=no
    --save_strategy=no
    --logging_strategy=epoch
    --learning_rate={lr}
    --num_train_epochs=1
    --max_seq_length=384
    --doc_stride=128
    --output_dir={output_dir}
    --overwrite_output_dir
    --train_adapter
    --save_adapter_model={adapter_output}
    --adapter_config=houlsby
    --load_adapter=@ukp/roberta-base_qa_squad1_houlsby
    """.split()

    with patch.object(sys, "argv", args):
        run_qa_alt.main()
	

def COVID_adapt():
    train_adapter_base(1e-5)
    train_adapter_base(2e-5)
    train_adapter_base(3e-5)
    train_adapter_base(5e-5)

def BIO_adapt():
    train_adapter(1e-5)
    train_adapter(2e-5)
    train_adapter(3e-5)
    train_adapter(5e-5)

def main():
    #data_files = {}
    #data_files["train"] = covid_file
	
    #covid_qa = get_dataset(covid_file)
    bio_qa = get_dataset(bio_file)

    #squad_qa = concatenate_datasets([squad_dataset['train'], squad_dataset['validation']])
    #covid_and_squad_dataset_path = "../data/full_squad_covidQA"

    bio_path = "../data/bioASQ"
    bio_data = datasets.dataset_dict.DatasetDict({'train':bio_qa})
    bio_data.save_to_disk(bio_path)
    run_gradual_ft("bio_adapter", "roberta-base", None, 3e-5, bio_path,"bio_adapter_pretrained")

if __name__ == "__main__":
    main()
