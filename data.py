import multiprocessing
import os
import re
from functools import partial
from datasets import load_from_disk, load_dataset, concatenate_datasets

def chat_str_to_chat_list(chat_str):
    try:
        chat_until_function_call = chat_str[
            : next(re.finditer(r"FUNCTION\sRESPONSE\:", chat_str)).start()
        ].strip()
    except StopIteration:
        chat_until_function_call = chat_str.strip()
    matches = re.findall(
        r"(USER|ASSISTANT):\s(.*?)(?=\n\n|$)", chat_until_function_call, re.DOTALL
    )
    chat_interaction = [
        (matchh[0], matchh[1].replace(" <|endoftext|>", "").strip())
        for matchh in matches
    ]
    return chat_interaction

def transform_dataset_to_chatdict_format(
    tokenize_model, transform_to_text, data_from_sample
):
    texts = []
    system_prompts = list(
        map(lambda x: re.split("SYSTEM\:\s", x)[1].strip(), data_from_sample["system"])
    )
    chats = list(map(chat_str_to_chat_list, data_from_sample["chat"]))
    for systemprompt, chatnow in zip(system_prompts, chats):
        messages = [{"role": "system", "content": systemprompt}] + [
            {"role": role.lower(), "content": msg} for role, msg in chatnow
        ]
        if transform_to_text:
            messages = tokenize_model.apply_chat_template(
                messages, add_generation_prompt=False, tokenize=False
            )
        texts.append(messages)
    return {"messages": texts}

def tokenize_text(tokenize_model, max_seq_length, data_from_sample):
    batch = tokenize_model(
        text=data_from_sample["messages"],
        truncation=True,
        padding="max_length",
        max_length=max_seq_length,
        return_tensors="pt",
    )
    return batch

def get_dataset_size(dataset_size):
    if dataset_size == "small":
        missed_amount = 200
        found_amount = 600
    elif dataset_size == "medium":
        missed_amount = 350
        found_amount = 750
    elif dataset_size == "large":
        missed_amount = 375
        found_amount = 825
    return missed_amount, found_amount

def get_dataset(dataset_path, seed=42):
    fn_calling_dataset = load_dataset(
        dataset_path,
        num_proc=multiprocessing.cpu_count(),
    )
    fn_calling_dataset = fn_calling_dataset["train"]

    # selecting samples only containing interactions of nature: function call or can't perform function call
    dataset_reduced = fn_calling_dataset.filter(
        lambda x: "I'm sorry" in x["chat"] or "functioncall" in x["chat"]
    ).shuffle(seed=seed)
    return dataset_reduced

def get_test_dataset(dataset_name, temp_dir, train_test_split=0.01, seed=42):
    if os.path.exists(os.path.join(temp_dir, "dataset_inference_test")):
        dataset_inference_test = load_from_disk(
            os.path.join(temp_dir, "dataset_inference_test")
        )
        return dataset_inference_test

    dataset_reduced = get_dataset(dataset_name, seed)

    # split data for test
    test_amount = max(int(train_test_split * dataset_reduced.num_rows), 25)
    dataset_reduced_test = dataset_reduced.select(
        range(dataset_reduced.num_rows - test_amount, dataset_reduced.num_rows)
    )
    dataset_inference_test = dataset_reduced_test.map(
        partial(transform_dataset_to_chatdict_format, None, False),
        batched=True,
        remove_columns=dataset_reduced_test.column_names,
    )
    dataset_inference_test.save_to_disk(
        os.path.join(temp_dir, "dataset_inference_test")
    )
    return dataset_inference_test

def get_train_dataset(dataset_name, temp_dir, tokenizer, max_seq_length, dataset_size, chat_text_only=False, train_eval_split=0.1, train_test_split=0.01, seed=42):

    sufix = ''
    if chat_text_only:
        sufix = '_text'

    if os.path.exists(os.path.join(temp_dir, f"dataset_train_eval{sufix}_{dataset_size}")):
        dataset_train_eval = load_from_disk(
            os.path.join(temp_dir, f"dataset_train_eval{sufix}_{dataset_size}")
        )
        return dataset_train_eval

    dataset_reduced = get_dataset(dataset_name, seed)
    missed_amount, found_amount = get_dataset_size(dataset_size)

    # split data for test
    test_amount = max(int(train_test_split * dataset_reduced.num_rows), 25)
    dataset_reduced_train = dataset_reduced.select(
        range(dataset_reduced.num_rows - test_amount)
    )


    # balance and limit dataset with found and not found functioncall examples
    dataset_train_missed = dataset_reduced_train.filter(
        lambda x: "I'm sorry" in x["chat"] and not ("functioncall" in x["chat"])
    )
    dataset_train_found = dataset_reduced_train.filter(
        lambda x: not ("I'm sorry" in x["chat"]) and "functioncall" in x["chat"]
    )
    dataset_train_missed = dataset_train_missed.select(
        range(min(dataset_train_missed.num_rows, missed_amount))
    )
    dataset_train_found = dataset_train_found.select(
        range(min(dataset_train_found.num_rows, found_amount))
    )
    dataset_final_train = concatenate_datasets(
        [dataset_train_missed, dataset_train_found]
    )
    if not chat_text_only:
        dataset_train = dataset_final_train.map(
            partial(transform_dataset_to_chatdict_format, tokenizer, True),
            batched=True,
            remove_columns=dataset_final_train.column_names,
        )
        dataset_train = dataset_train.map(
            partial(tokenize_text, tokenizer, max_seq_length),
            batched=True,
            remove_columns=dataset_train.column_names,
        )
    else:
        dataset_train = dataset_final_train.map(
            partial(transform_dataset_to_chatdict_format, None, False),
            batched=True,
            remove_columns=dataset_final_train.column_names,
        )
    # split between train and validation
    dataset_train_eval = dataset_train.train_test_split(test_size=train_eval_split)

    # save to disk
    dataset_train_eval.save_to_disk(os.path.join(temp_dir, f"dataset_train_eval{sufix}_{dataset_size}"))
    return dataset_train_eval
