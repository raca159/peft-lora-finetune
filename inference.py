from datasets import concatenate_datasets
import os
import json
import pandas as pd
import re
from peft import get_peft_model, PeftModel
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import multiprocessing
from datasets import load_dataset

peft_model = PeftModel.from_pretrained(model, checkpoint_lora_path)

def run_inout_pipe(chat_interaction, tokenizer, model):
    prompt = tokenizer.apply_chat_template(chat_interaction, tokenize=False, add_generation_prompt=False)
    # inputs = tokenizer.encode(text=prompt, return_tensors="pt").to(model.device)
    # outputs = model.generate(inputs, max_new_tokens=768)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs)
    outputs = outputs[:, inputs['input_ids'].shape[-1]:]
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

results = []
for sample in ds_test.shuffle(seed=42).select(range(5)).to_list():
    sample_test = sample['messages']

    instruction, tools = sample_test[0]['content'].split('Use them if required')
    sample_test[0]['content'] = instruction + \
        'The answer should be the json of the function call when possible to interpret as a function calling. If possible to create the json, then answer with only "<functioncall> {"name": ..., "arguments": ..., ...}". Here are the function call descriptions:' + \
        tools

    answer = run_inout_pipe(sample_test[:-1], tokenizer, model)
    peft_answer = run_inout_pipe(sample_test[:-1], tokenizer, peft_model)

    results.append({
        'input': sample_test,
        'base_answer': answer,
        'peft_answer': peft_answer,
        'expected': sample_test[-1]['content']
    })

results_df = pd.DataFrame(results)

def extract_fn_call(x):
    try:
        if '<functioncall>' in x:
            return json.loads(x[next(re.finditer('functioncall', x)).start()-1:].replace('<functioncall>', '').replace('</functioncall>', '').replace("'", '').strip())
        else:
            return 'False'
    except:
        # parse error
        return 'None'

results_df['base_answer_process'] = results_df['base_answer'].apply(extract_fn_call)
results_df['peft_answer_process'] = results_df['peft_answer'].apply(extract_fn_call)
results_df['expected_process'] = results_df['expected'].apply(extract_fn_call)

results_df['base_answer_process_str'] = results_df['base_answer_process'].apply(lambda x: json.dumps(x))
results_df['peft_answer_process_str'] = results_df['peft_answer_process'].apply(lambda x: json.dumps(x))
results_df['expected_process_str'] = results_df['expected_process'].apply(lambda x: json.dumps(x))


def calculate_accuracy(x, col):
    return (x.expected_process =='False' and x[col] == 'False') or x[col] == 'None'

def measure_rouge(x, col):
    if x.expected_process == 'False' or x[col] == 'False' or x[col] == 'None':
        return {'precision': 0, 'recall': 0, 'fmeasure': 0}
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(x.expected_process_str, x[col])
    return {
        'precision': scores['rougeL'].precision,
        'recall': scores['rougeL'].recall,
        'fmeasure': scores['rougeL'].fmeasure
    }

def measure_blue(x, col):

    if x.expected_process == 'False' or x[col] == 'False' or x[col] == 'None':
        return 0
    
    return sentence_bleu(x.expected_process_str, x[col], smoothing_function=SmoothingFunction().method1)

results_df['base_classification'] = results_df.apply(lambda x: calculate_accuracy(x, col='base_answer_process'), axis=1)
results_df['peft_classification'] = results_df.apply(lambda x: calculate_accuracy(x, col='peft_answer_process'), axis=1)

results_df['base_rouge'] = results_df.apply(lambda x: measure_rouge(x, col='base_answer_process_str'), axis=1)
results_df['peft_rouge'] = results_df.apply(lambda x: measure_rouge(x, col='peft_answer_process_str'), axis=1)

results_df['base_rouge_precision'] = results_df['base_rouge'].apply(lambda x: x['precision'])
results_df['base_rouge_recall'] = results_df['base_rouge'].apply(lambda x: x['recall'])
results_df['base_rouge_fmeasure'] = results_df['base_rouge'].apply(lambda x: x['fmeasure'])
results_df['peft_rouge_precision'] = results_df['peft_rouge'].apply(lambda x: x['precision'])
results_df['peft_rouge_recall'] = results_df['peft_rouge'].apply(lambda x: x['recall'])
results_df['peft_rouge_fmeasure'] = results_df['peft_rouge'].apply(lambda x: x['fmeasure'])

results_df['base_blue'] = results_df.apply(lambda x: measure_blue(x, col='base_answer_process_str'), axis=1)
results_df['peft_blue'] = results_df.apply(lambda x: measure_blue(x, col='peft_answer_process_str'), axis=1)

base_results_df = results_df[['base_classification', 'base_rouge_precision', 'base_rouge_fmeasure', 'base_rouge_recall', 'base_blue']].rename(columns={
    'base_classification': 'classification',
    'base_rouge_precision': 'rouge_precision',
    'base_rouge_fmeasure': 'rouge_fmeasure',
    'base_rouge_recall': 'rouge_recall',
    'base_blue': 'blue'
})
peft_results_df = results_df[['peft_classification', 'peft_rouge_precision', 'peft_rouge_fmeasure', 'peft_rouge_recall', 'peft_blue']].rename(columns={
    'peft_classification': 'classification',
    'peft_rouge_precision': 'rouge_precision',
    'peft_rouge_fmeasure': 'rouge_fmeasure',
    'peft_rouge_recall': 'rouge_recall',
    'peft_blue': 'blue'
})

base_results_df.describe()
peft_results_df.describe()
