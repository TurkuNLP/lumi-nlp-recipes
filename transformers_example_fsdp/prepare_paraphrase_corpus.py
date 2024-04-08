from pathlib import Path
import datasets
import os

output_dir = Path("data")
output_dir.mkdir(parents=True, exist_ok=True)

data = datasets.load_dataset('TurkuNLP/turku_paraphrase_corpus', 'plain')
data = data['train'].filter(lambda x: x['binary_label']=='positive').filter(lambda x, i: i%3==0, with_indices=True)

eos_token = "</s>"
def create_instruction(sample):
    return {"text": f"Alkuperäinen lause: {sample['text1']} Parafrasoitu lause: {sample['text2']}{eos_token}\n"}

data = data.map(create_instruction)

with open(os.path.join(output_dir,"paraphrase_corpus.txt"), 'w') as f:
    for row in data['text']:
        f.write(row + "\n")
