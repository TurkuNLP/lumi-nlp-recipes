## Finetuning with TRL


## Structure

[`slurm_scripts`](./slurm_scripts/) For setting up your python virtual enviroment and launching slurm jobs.

[`training`](./training/) scripts for training and utility stuff.

[`configs`](./configs/) Yaml files for the accelerate configurations and training arguments

## Getting started

Go to [`slurm_scripts`](./slurm_scripts/) and modify it according to your own paths.

```
sbatch slurm_scripts/setup_venv
```

## Dataset

Either use your own dataset and convert it to chatml's conversational format. You can get inspiration from [`data`](./data/).

#### Format
```json
{"messages": [{"role": "system", "content": "You are..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"messages": [{"role": "system", "content": "You are..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"messages": [{"role": "system", "content": "You are..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

or use some of the ready-made datasets in [`/scratch/project_462000558/TurkuNLP_workshop/data`](/pfs/lustrep3/scratch/project_462000558/TurkuNLP_workshop/data)

## Training

Go to [`configs`](./configs/) if you want to change training arguments, as you would with [`huggingface TrainingArguments`](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments)

Modify the launch script at [`slurm_scripts`](./slurm_scripts/sft.sh)

```
sbatch slurm_scripts/sft.sh
```
Full model weight training on a 34B model requires minimum of 2 nodes and atleast 3 nodes is required. Also note that as you increase the amount of nodes, the training becomes more unstable and prone to nccl crashes/hangs. 

## Useful links

This work was heavily inspired by: 
* https://github.com/pacman100/DHS-LLM-Workshop/tree/main
* https://www.philschmid.de/fine-tune-llms-in-2024-with-trl
* https://github.com/huggingface/alignment-handbook
* https://wandb.ai/capecape/alpaca_ft/reports/How-to-Fine-tune-an-LLM-Part-3-The-Hugging-Face-Trainer--Vmlldzo1OTEyNjMy

My own fork of the alignment handbook is
https://github.com/Vmjkom/alignment-handbook (wip).
The Alignment handbook implements reinforcement learning techniques, along with the Sft. Furthermore, there is more sophisticated data handling

## Documentation

* [`SftTrainer`](https://huggingface.co/docs/trl/main/en/sft_trainer)
* [`Chat templating`](https://huggingface.co/docs/transformers/main/en/chat_templating)
* [`Single and Multi-node Launchers with SLURM`](https://github.com/stas00/ml-engineering/tree/master/orchestration/slurm/launchers)
* [`PEFT`](https://huggingface.co/docs/peft/en/index) (Parameter Efficient finetuning -- Lora...etc)