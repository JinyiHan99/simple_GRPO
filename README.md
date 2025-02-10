# simple_GRPO
A very simple GRPO implement for reproducing r1-like LLM thinking.
This is a simple open source implementation that utilizes the core loss calculation formula referenced from Hugging Face's trl. 
We make the simplest codebase to support: 
- Save the GPU memory to make a feasible and efficient training. 
- Quickly understand RL processes such as GRPO from a teaching perspective. 
- Quickly try a lot of things, such as improved multi-answer generation, regrouping, penalty on KL, and parameter tuning.
     
## Features
### Core Loss Calculation: 
The loss calculation formula is based on Hugging Face's trl. We extend our gratitude to Hugging Face for their contribution.

### Simplicity
The codebase is simple, with only about 200 lines of code spread across 2 files. It only depends on standard libraries such as deepspeed and torch, without requiring dependencies like ray. It is designed to allow for more complex interventions.

### Splited Reference Model
The reference model part is decoupled, which allows it to be run on different GPUs (even on a different machine with 4090s). This avoids having the reference model and the training model on the same GPU, preventing multiple copies created by torch’s multiprocessing, and enabling training of a 7B model on 80G A800s.

## Proven Results
Without requiring SFT, the implementation has already demonstrated formatting preferences and longer thinking processes. 
More experimental results are being compiled and will be released soon.

## Environment
The runtime environment is depicted below:
```
>> ds_report
torch version .................... 2.3.0+cu121
deepspeed info ................... 0.12.0, unknown, unknown
torch cuda version ............... 12.1
torch hip version ................ None
nvcc version ..................... 12.1
deepspeed wheel compiled w. ...... torch 2.3, cuda 12.1
shared memory (/dev/shm) size .... 1007.76 GB
```
At least two GPUs are needed.

## Usage
Run the following command:
``` bash
CUDA_VISIBLE_DEVICES=7 python ref_model.py
```
This just uses one GPU to collect and run the reference model.
We use http to transport data and logits.
They have so little data that they won't be any bottlenecks, http is the easiest to understand and has the fewest dependencies, and can be easily supported on multiple machines (we are using dozens of outdated 4090s to generate QA pairs for them!).

Then, open another bash:
``` bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 deepspeed grpo_ref_split.py
```
Use all other GPUs for training!

All parameters are in the code. We need to try more possibilities than a fking long argparse.

## TODO
- Showing our exciting experimental results.
- Answer generation may be invalid due to a group containing all wrong answers or all correct answers. We need group reorganization and better answer generation.
- GPU memory is still tight if it generates long cots. We have to split the groups to make the batch smaller.

We have implemented and are testing these features. They will be available soon.

## Authors
This project is developed by Dr. Jiaqing Liang in the KnowledgeWorks Lab at Fudan University.
