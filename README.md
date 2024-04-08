# lumi-nlp-recipes
## Interactive generation examples
[https://github.com/LumiOpen/lumiopen-tools.git](https://github.com/LumiOpen/lumiopen-tools)


## Running training on LUMI with `trl` in a multi-node setup

[`sft_trl`](./sft_trl/)

### Workshop 8.4.2024

### Trl - Transformers Reinforcement Learning
https://huggingface.co/docs/trl/main/en/index

## Running a simple Poro finetuning with FSDP
[https://github.com/TurkuNLP/lumi-nlp-recipes/tree/main/transformers_example_fsdp](https://github.com/TurkuNLP/lumi-nlp-recipes/tree/main/transformers_example_fsdp)



## Handy commands:

Interactive shell on an active run.
`srun --jobid <JOBID> --overlap --pty bash`
monitor gpu-usage 
`rocm-smi`
show update-loop with  `watch`
`watch -n 0.5 rocm-smi`

Load administrator tools
`module load LUMI/23.09 && module load systools/23.09`

use basic linux monitoring tools
`htop`
`tree`
...

```
