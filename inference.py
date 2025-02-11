
from data import get_test_dataset
from datasets import concatenate_datasets
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
import os
import pandas as pd
from langchain.prompts import ChatPromptTemplate
import re
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score
from rouge_score import rouge_scorer
from model_handler import local_load_peft_model, run_inout_pipe


load_dotenv()

dataset_path = "glaiveai/glaive-function-calling-v2"
seed = 42
temp_dir = 'dataset_dir'
max_seq_length = 512

def get_test_ds(dataset_path, temp_dir, seed=42):
    ds_test = get_test_dataset(dataset_name=dataset_path, temp_dir=temp_dir)

    ds_test_missed = ds_test.filter(lambda x: 'functioncall' in x['messages'][-1]['content'])
    ds_test_found = ds_test.filter(lambda x: not ('functioncall' in x['messages'][-1]['content']))

    ds_test = concatenate_datasets([
        ds_test_missed.select(range(50)), ds_test_found.select(range(100))
    ]).shuffle(seed=seed)
    return ds_test

def prep_dataset_for_foundation_model(sample):
    instruction, tools = sample[0]['content'].split('Use them if required')
    sample[0]['content'] = instruction + \
        'The answer should be the json of the function call when possible to interpret as a function calling. If possible to create the json, then answer with "<functioncall> {"name": ..., "arguments": ..., ...}". Here are the function call descriptions:' + \
        tools
    return sample

def get_last_checkpoint_from_finetune(run_name, root_dir='results'):
    checkpoints = os.listdir(os.path.join(root_dir, run_name))
    last_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1]
    return os.path.join(root_dir, run_name, last_checkpoint)

def run_models_against_dataset(ds_test, foundation_model, peft_model_small, tokenizer_small, peft_model_big, tokenizer_big):

    results = []
    for sample in ds_test.to_list():
        role_swap = {'system': 'system', 'assistant': 'ai', 'user': 'human'}
        chat_interaction = sample['messages'][:-1]
        langchain_chat_interaction = ChatPromptTemplate.from_messages([
            (role_swap[k['role']], k['content'].replace("{", "{{").replace("}", "}}")) \
                for k in prep_dataset_for_foundation_model(chat_interaction)
        ]).invoke({})
        groundtruth_response = sample['messages'][-1]['content']
        
        response_foundation = foundation_model.invoke(langchain_chat_interaction).content
        response_small = run_inout_pipe(chat_interaction, tokenizer_small, peft_model_small)
        response_big = run_inout_pipe(chat_interaction, tokenizer_big, peft_model_big)

        results.append({
            'chat_interaction': chat_interaction,
            'groundtruth_response': groundtruth_response,
            'response_foundation': response_foundation,
            'response_small': response_small,
            'response_big': response_big,
        })
    results_df = pd.DataFrame(results)
    return results_df

# Helper function to clean json function calls
def extract_fncall(resp):
    resp_funcs_iters = list(re.finditer('functioncall>', resp))[:2]
    if len(resp_funcs_iters): 
        resp = resp[resp_funcs_iters[0].end():].strip()
        if len(resp_funcs_iters)>1:
            resp_funcs_iter = next(re.finditer('functioncall>', resp))
            resp = resp[:resp_funcs_iter.start()-1].strip()
            if resp.endswith('/'):
                resp = resp[:-1]
        try:
            resp = json.dumps(json.loads(resp))
        except json.JSONDecodeError:
            pass
    return resp

def measure_rouge(x, col, groudcol):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(x[groudcol], x[col])
    return {
        'precision': scores['rougeL'].precision*100,
        'recall': scores['rougeL'].recall*100
    }
def process_results_and_generate_metrics(results_df):

    # Binary columns for function call presence
    results_df['groundtruth_has_functioncall'] = results_df.groundtruth_response.apply(lambda x: 'functioncall' in x)
    results_df['foundation_has_functioncall'] = results_df.response_foundation.apply(lambda x: 'functioncall' in x)
    results_df['small_has_functioncall'] = results_df.response_small.apply(lambda x: 'functioncall' in x)
    results_df['big_has_functioncall'] = results_df.response_big.apply(lambda x: 'functioncall' in x)

    # Cleaned function calls
    results_df['groundtruth_cleaned'] = results_df.groundtruth_response.apply(extract_fncall)
    results_df['foundation_cleaned'] = results_df.response_foundation.apply(extract_fncall)
    results_df['small_cleaned'] = results_df.response_small.apply(extract_fncall)
    results_df['big_cleaned'] = results_df.response_big.apply(extract_fncall)

    results_df['foundation_rouge'] = results_df.apply(lambda x: measure_rouge(x, col='foundation_cleaned', groudcol='groundtruth_cleaned'), axis=1)
    results_df['small_rouge'] = results_df.apply(lambda x: measure_rouge(x, col='small_cleaned', groudcol='groundtruth_cleaned'), axis=1)
    results_df['big_rouge'] = results_df.apply(lambda x: measure_rouge(x, col='big_cleaned', groudcol='groundtruth_cleaned'), axis=1)

    results_df['foundation_rouge_precision'] = results_df['foundation_rouge'].apply(lambda x: x['precision'])
    results_df['foundation_rouge_recall'] = results_df['foundation_rouge'].apply(lambda x: x['recall'])

    results_df['small_rouge_precision'] = results_df['small_rouge'].apply(lambda x: x['precision'])
    results_df['small_rouge_recall'] = results_df['small_rouge'].apply(lambda x: x['recall'])

    results_df['big_rouge_precision'] = results_df['big_rouge'].apply(lambda x: x['precision'])
    results_df['big_rouge_recall'] = results_df['big_rouge'].apply(lambda x: x['recall'])

    del results_df['foundation_rouge']
    del results_df['small_rouge']
    del results_df['big_rouge']


    foundation_accuracy_score = accuracy_score(results_df.groundtruth_has_functioncall, results_df.foundation_has_functioncall)*100
    foundation_precision_score = precision_score(results_df.groundtruth_has_functioncall, results_df.foundation_has_functioncall)*100
    foundation_recall_score = recall_score(results_df.groundtruth_has_functioncall, results_df.foundation_has_functioncall)*100
    big_accuracy_score = accuracy_score(results_df.groundtruth_has_functioncall, results_df.big_has_functioncall)*100
    big_precision_score = precision_score(results_df.groundtruth_has_functioncall, results_df.big_has_functioncall)*100
    big_recall_score = recall_score(results_df.groundtruth_has_functioncall, results_df.big_has_functioncall)*100
    small_accuracy_score = accuracy_score(results_df.groundtruth_has_functioncall, results_df.small_has_functioncall)*100
    small_precision_score = precision_score(results_df.groundtruth_has_functioncall, results_df.small_has_functioncall)*100
    small_recall_score = recall_score(results_df.groundtruth_has_functioncall, results_df.small_has_functioncall)*100

    foundation_rouge_precision_score = results_df['foundation_rouge_precision'].mean()
    big_rouge_precision_score = results_df['big_rouge_precision'].mean()
    small_rouge_precision_score = results_df['small_rouge_precision'].mean()
    foundation_rouge_recall_score = results_df['foundation_rouge_recall'].mean()
    big_rouge_recall_score = results_df['big_rouge_recall'].mean()
    small_rouge_recall_score = results_df['small_rouge_recall'].mean()

    return pd.DataFrame({
        'models': ['foundation', 'big', 'small'],
        'Accuracy': [foundation_accuracy_score, big_accuracy_score, small_accuracy_score],
        'Precision': [foundation_precision_score, big_precision_score, small_precision_score],
        'Recall': [foundation_recall_score, big_recall_score, small_recall_score],
        'ROUGE Precision': [foundation_rouge_precision_score, big_rouge_precision_score, small_rouge_precision_score],
        'ROUGE Recall': [foundation_rouge_recall_score, big_rouge_recall_score, small_rouge_recall_score],
    }).round(2).set_index('models').T.map(lambda x: '{:.2f} %'.format(x))

if __name__ == '__main__':

    ds_test = get_test_ds(dataset_path, temp_dir, seed=42)
    print(ds_test)

    foundation_model = AzureChatOpenAI(azure_deployment=os.environ['GPTO_DEPLOYMENT_NAME'])


    model_name = 'HuggingFaceTB/SmolLM2-135M-Instruct'
    run_name = 'SmolLM2-135M-Instruct-fncall_peft-ds_medium-lora_r_4-use_qlora_False-lr_5e-04'
    use_qlora=False

    peft_model_small, tokenizer_small = local_load_peft_model(
        model_name, get_last_checkpoint_from_finetune(run_name+'_new'),
        max_seq_length=max_seq_length,
        use_qlora=use_qlora,
        use_flash_attention=True
    )

    model_name = 'HuggingFaceTB/SmolLM2-360M-Instruct'
    run_name = 'SmolLM2-360M-Instruct-fncall_peft-ds_small-lora_r_4-use_qlora_True-lr_5e-04'
    use_qlora = True

    peft_model_big, tokenizer_big = local_load_peft_model(
        model_name, get_last_checkpoint_from_finetune(run_name+'_new'),
        max_seq_length=max_seq_length,
        use_qlora=use_qlora,
        use_flash_attention=True
    )

    results_df = run_models_against_dataset(ds_test, foundation_model, peft_model_small, tokenizer_small, peft_model_big, tokenizer_big)

    metrics_df = process_results_and_generate_metrics(results_df)
    print(metrics_df)