from transformers import AutoTokenizer, AutoModelForCausalLM
import json, os, shutil, re, random, requests, io, sys
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
from datasets import load_dataset

os.environ['TOKENIZERS_PARALLELISM'] = 'true'
# os.environ['CUDA_VISIBLE_DEVICES'] = '5'

#specify the parameters
num_gpu = 1
max_prompt_length = 400
max_new_token_length = 600
save_path = "./ckp/0214_mix"
res_path = "./logs/record_res_v3.json"
data_path = "./data/mix_gsm_8k_math_4k.json"

with open(data_path, 'r', encoding='utf-8') as file:
    dataset = json.load(file)

QAs = [{'Q': item['question'], 'A': item['gt_answer']} for item in dataset]
max_epoch = 2
all_steps = (len(QAs)//num_gpu) * max_epoch
save_steps = all_steps // 8

model_path = "/mnt/remote-data/downloads/models/Qwen/Qwen2.5-7B"
beta = 0.04
num_pre_Q = 8
Q_batch_size = 1


records = {"total_count":0,
            "total_acc_correct":0,
            "total_format_correct":0,
            "total_nums":0,
            "total_length":0,
            
            "steps":[],
            "acc_ratio":[],
            "format_ratio":[],
            "avg_length":[],
            "good_cases":[],
            "batch_length":[]
            }

ds_config = {
    "train_micro_batch_size_per_gpu": Q_batch_size*num_pre_Q,
    "gradient_accumulation_steps": 2,
    "optimizer": {
        "type": "AdamW",
        "params": { "lr": 1e-6 }
    },
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
        "stage3_gather_16bit_weights_on_model_save": True,
        "offload_optimizer": {"device": "cpu"}
    }
}

ref_server = "http://localhost:59878"
from ref_server import tensor_to_bytes, bytes_to_tensor, make_bytes_list, bytes_list_to_list

def get_batch():
    try:
        r = requests.get(f"{ref_server}/get").content
        if r == b'empty': return None
    except: return None
    dd = bytes_list_to_list(r)
    data = json.loads(dd[0]) 
    data['inputs'] = bytes_to_tensor(dd[1])
    data['advantages'] = bytes_to_tensor(dd[2])
    data['refs'] = bytes_to_tensor(dd[3])
    return data

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, 
        torch_dtype=torch.bfloat16, _attn_implementation="sdpa")
gen_model = model

from transformers import GenerationConfig
generation_config = GenerationConfig(
            max_new_tokens=max_new_token_length,
            do_sample=True, temperature=0.9, 
            num_return_sequences=num_pre_Q,
            pad_token_id=tokenizer.pad_token_id,
        )

system_prompt = """You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\
The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."""

def gen_answers(prompts):
    tip_text = []
    for x in prompts:
        tip_text.append(tokenizer.apply_chat_template([
             {"role": "system", "content": system_prompt},
             {"role": "user", "content": x}], tokenize=False, add_generation_prompt=True))
        tip_inputs = tokenizer(tip_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
    prompt_length = tip_inputs["input_ids"].shape[-1]

    if prompt_length > max_prompt_length:
        print("!!!! the input is so long")
        return []
    tip_inputs = tokenizer(tip_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
    tip_inputs = {k: v.to(model.device) for k, v in tip_inputs.items()}
    tip_completion_ids = gen_model.generate(**tip_inputs, generation_config=generation_config)
    prompt_length = tip_inputs["input_ids"].shape[-1]
    completion_ids = tip_completion_ids[:, prompt_length:]
    answers = [tokenizer.decode(x).replace('<|endoftext|>', '') for x in completion_ids]
    return answers

from math_verify import parse, verify, ExprExtractionConfig
def reward_correct(item, answer):
    pattern = r'\d+\.\d+|\d+/\d+|\d+'
    nums = re.findall(pattern, answer) 
    if len(nums) == 0: return -1.0
    lastnum = nums[-1] 
    ans = parse(lastnum, extraction_config=[ExprExtractionConfig()])
    ground_truth = parse(item["A"], extraction_config=[ExprExtractionConfig()])
    return 1.0 if verify(ans, ground_truth) else -1.0

def reward_format(item, answer):
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    return 1.0 if re.match(pattern, answer, re.DOTALL | re.VERBOSE) else -1

def gen_samples(inputs):
    prompts = [x["Q"] for x in inputs]
    prompts_text = [tokenizer.apply_chat_template([
             {"role": "system", "content": system_prompt},
             {"role": "user", "content": x}], tokenize=False, add_generation_prompt=True) for x in prompts]
    prompt_inputs = tokenizer(prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)

    prompts = [x["Q"] for x in inputs]
    answers = gen_answers(prompts)
    if len(answers) == 0:
        return torch.Tensor([]), torch.Tensor([]), torch.Tensor([]), [], -1,-1,[]

    rewards = []
    acc_correct_num=0 
    format_correct_num=0
    output_ids_length=[]
    for i, inp in enumerate(inputs):
        for a in answers[i*num_pre_Q:(i+1)*num_pre_Q]:
            acc_reward = reward_correct(inp, a)
            format_reward = reward_format(inp, a)
            if acc_reward > 0 and format_reward<0:
                cur_reward = 0
            if acc_reward > 0 and format_reward>0:
                cur_reward = 2.0
                records['good_cases'].append({"Q":i, "answer:":a})
                # fout.write(json.dumps({"Q":i, "answer:":a}, indent=4))
                # fout.flush()
            if acc_reward< 0 and format_reward<0:
                cur_reward = -2.0
            if acc_reward < 0 and format_reward>0:
                cur_reward = 0.5

            rewards.append(cur_reward)
            if acc_reward>0:
                acc_correct_num += 1
            if format_reward>0:
                format_correct_num += 1
            # rewards.append(reward_correct(inp, a) + reward_format(inp, a))
    output_ids = tokenizer(answers, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False)
    output_ids_length = [len(tokenizer(a)["input_ids"]) for a in answers]
    
    return prompt_inputs["input_ids"], output_ids["input_ids"], torch.tensor(rewards, dtype=torch.float32), answers, acc_correct_num,format_correct_num,output_ids_length

def generate_mode(num=10, rank=0):
    if rank == 0: print('enter generate mode')
   
    records['total_count'] += 1
    for ii in range(num):
        inputs = random.sample(QAs, Q_batch_size)
        prompt_inputs, output_ids, rewards, answers,acc_correct_num,format_correct_num, output_ids_length = gen_samples(inputs)
        if format_correct_num<0:
            continue
        if rank == 0: 
            print('rewards:', rewards)
        if rank == 0:
            print("="*60)
            print(f"[Question:]{inputs[0]['Q']}\n\n")
            for i in range(len(answers)):
                print("-"*30)
                
                print(f'[answers{i}:]\n{answers[i]}')
        if (rewards.max() - rewards.min()).item() < 0.01: continue
        rep = output_ids.shape[0] // prompt_inputs.shape[0]
        prompt_length = prompt_inputs.shape[1]
        Qrep = prompt_inputs.repeat(1, rep).view(-1, prompt_length)
        merged_ids = torch.cat([Qrep, output_ids], dim=1)
        
        grouped_rewards = rewards.view(-1, num_pre_Q)
        mean_grouped = grouped_rewards.mean(dim=1, keepdim=True)
        std_grouped = grouped_rewards.std(dim=1, keepdim=True).clamp(min=1e-4)
        advantages = (grouped_rewards - mean_grouped) / std_grouped
        advantages = advantages.view(-1)  
        
        xdata = make_bytes_list([json.dumps({"plen": prompt_length}).encode(), tensor_to_bytes(merged_ids), tensor_to_bytes(advantages)])
        requests.post(f"{ref_server}/upload", data=xdata)
    if rank == 0: 
        print('exit generate mode')
        #记录结果
        if len(output_ids_length)>0:
            records['total_acc_correct'] += acc_correct_num
            records['total_format_correct'] += format_correct_num
            records['total_length'] += sum(output_ids_length)
            records['total_nums'] += len(output_ids_length)
            
            records['steps'].append(records['total_count'])
            records['acc_ratio'].append(records['total_acc_correct'] / records['total_nums'] if records['total_nums']>0 else 0 )
            records['format_ratio'].append(records['total_format_correct'] / records['total_nums'] if records['total_nums']>0 else 0)
            records['avg_length'].append(records['total_length'] / records['total_nums'] if records['total_nums']>0 else 0)
            records['batch_length'].append(sum(output_ids_length) / len(output_ids_length) if len(output_ids_length)>0 else 0)
            #将结果写入到文件中
            record = {"steps:":records['steps'],
                    "acc_ratio":records['acc_ratio'],
                    "format_ratio":records['format_ratio'],
                    "avg_length":records['avg_length'],
                    "batch_length": records['batch_length'],
                    "good cases":records['good_cases'],
                    }
            with open(res_path, 'w') as file:
                json.dump(record, file, indent=4)

if 'genonly' in sys.argv:
    model.to('cuda')
    generate_mode(999999)
    sys.exit()

import deepspeed
engine, optimizer, _, _ = deepspeed.initialize(config=ds_config, model=model, 
                                               model_parameters=model.parameters())
gen_model = engine

def GRPO_step(batch):
    prompt_length = batch['plen']
    inputs = batch['inputs'].to(engine.device)
    advantages = batch['advantages'].to(engine.device)

    def get_per_token_logps(logits, input_ids):
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)
    
    per_token_logps = get_per_token_logps(engine(inputs).logits, inputs)
    per_token_logps = per_token_logps[:,prompt_length-1:]
    ref_per_token_logps = batch['refs'].to(per_token_logps.device)

    per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
    completion_mask = (inputs[:, prompt_length:] != tokenizer.pad_token_id).int()

    per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
    per_token_loss = -(per_token_loss - beta * per_token_kl)
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    return loss

generate_mode(rank=torch.distributed.get_rank())

from tqdm import tqdm
progress = range(1, all_steps+1)
if torch.distributed.get_rank() == 0: progress = tqdm(progress)
for step in progress:
    batch = get_batch()
    while batch is None:
        generate_mode(rank=torch.distributed.get_rank())
        batch = get_batch()
    loss = GRPO_step(batch)

    engine.backward(loss)
    engine.step()

    if torch.distributed.get_rank() == 0:
        progress.set_description(f"Loss: {loss.item():.6f}")
    
 
    if step % save_steps == 0:
        dist.barrier()
        if torch.distributed.get_rank() == 0:
            print('saving model')
            save_name = f"{save_path}/step_{step}"
            state_dict = engine.module.state_dict()
            state_dict = type(state_dict)({k: v.cpu() for k, v in state_dict.items()})
            engine.module.save_pretrained(save_name, state_dict=state_dict)
            tokenizer.save_pretrained(save_name)
        dist.barrier()
