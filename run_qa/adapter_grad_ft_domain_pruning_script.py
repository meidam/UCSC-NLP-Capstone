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
    K = 6

    for i in range(k_fold):
        covid_fold = covid_qa.shard(k_fold, i)

        covid_test = covid_fold.shard(2, 0)
        covid_val = covid_fold.shard(2, 1)
        covid_train = concatenate_datasets([covid_qa.shard(k_fold, j) for j in range(k_fold) if j != i])

        # make_and_save_full_dataset(covid_train, squad_qa, covid_val, covid_test, covid_and_squad_dataset_path)

        checkpoint = 'roberta-base'
        adapter_load = '@ukp/roberta-base_qa_squad1_houlsby'
        cur_dir = '../adapter_models/gradual_ft_baseline_lr1e-5/split_' + str(i)

        # log_file = open(cur_dir + '/log_file.txt',"w+")

        # sys.stdout = log_file

        squad_qa.shuffle()
        bio_qa.shuffle()

        num_of_shards = int(K / 2)
        squad_qa_shards = [squad_qa.shard(num_of_shards, i) for i in range(num_of_shards)]
        bio_qa_shards = [bio_qa.shard(num_of_shards, i) for i in range(num_of_shards)]

        squad_qa_cur = concatenate_datasets(squad_qa_shards)
        bio_qa_cur = concatenate_datasets(bio_qa_shards)

        squad_rows_to_remove = len(squad_qa_shards[0])
        bio_rows_to_remove = len(bio_qa_shards[0])
        L = squad_qa_shards + bio_qa_shards + [covid_train]
        print(f'Gradual finetuning with K={K} ({K // 2} per SQuAD and bioASQ)')
        for n in range(K + 1):
            output_dir = cur_dir + '/checkpoint_' + str(n)

            full_dataset = concatenate_datasets(L)
            make_and_save_full_dataset(full_dataset, covid_val, covid_test, covid_and_squad_dataset_path)

            print(f'n={n} =============')
            print('Total: ', full_dataset.num_rows)

            run_gradual_ft(output_dir, checkpoint, covid_val, covid_and_squad_dataset_path, adapter_load)
            print('=================\n')

            checkpoint = output_dir
            adapter_load = checkpoint + '/full_squad_covidQA'

            if n < K // 2:
                print(f'Removing {squad_rows_to_remove} from SQuAD')
                _ = L.pop(0)

            elif n < K:
                print(f'Removing {bio_rows_to_remove} from bioASQ')
                _ = L.pop(0)

            else:
                assert full_dataset.num_rows == covid_train.num_rows
                full_dataset = covid_train




def run_gradual_ft(output_dir, checkpoint, lr, dataset_name, adapter_checkpoint):
    args = f"""
    run_qa_alt.py 
    --model_name_or_path={checkpoint}
    --dataset_name={dataset_name}
    --do_train
    --do_eval
    --do_predict
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
    --adapter_config=houlsby
    --load_adapter={adapter_checkpoint}
    """.split()

    with patch.object(sys, "argv", args):
        run_qa_alt.main()


def main():
    train_adapter(5e-5)

if __name__ == "__main__":
    main()
