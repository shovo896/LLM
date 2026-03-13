

import torch
import torch.nn as nn
import torch.nn.functional as F
import mmap 
import random
import pickle 
import argparse 
parser = argparse.ArgumentParser(description='Train a GPT-like model on the OpenWebText dataset.')
parser.add_argument('batch_size', type=int, default=8,required=True, help='Batch size for training')
args = parser.parse_args()
batch_size = args.batch_size
block_size=128 
max_iters=200 
learning_rate=3e-4 
eval_iters=100 
n_embd=384 
n_head=1
n_layer=1 
dropout = 0.2
print(device)

chars=""
with open("/content/vocab.txt","r",encoding="utf-8") as f:
    chars=f.read()
chars=sorted(set(chars))
text=f.read()
string_to_int={ch:i for i,ch in enumerate(chars)}
int_to_string={i:ch for i,ch in enumerate(chars)}
encode=lambda s:[string_to_int[c] for c in s]
decode=lambda l:"".join([int_to_string[i] for i in l])
vocab_size=len(chars)
print(f"vocab_size: {vocab_size}")
print("encode('hello world'):", encode("hello world"))
def get_random_chunk(split):
    path="/content/output_train.txt" if split=="train" else "/content/output_val.txt"
    with open(path,"r",encoding="utf-8") as f:
        data=f.read()
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            file_size = mm.size()
            start_pos=random.randint(0,(file_size)-batch_size*block_size)
            mm.seek(start_pos)
            block=mm.read(block_size*batch_size-1).decode("utf-8", errors="ignore")
            decoded_block=block.decode("utf-8", errors="ignore").replace("r"," ")
            data=torch.tensor(encode(decoded_block), dtype=torch.long)
return data

def get_batch(split):
    data=get_random_chunk(split)
    ix=torch.randint(len(data)-block_size, (batch_size,))
    x=torch.stack([data[i:i+block_size] for i in ix])
    y=torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)


@ torch.no_grad()
def estimate_loss():
    out={}
    model.eval()
    for split in ['train','val']:
        losses=torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y=get_batch(split)
            _,loss=model(X,Y)
            losses[k]=loss.item()
        out[split]=losses.mean()
    model.train()
    return out

model=GPTLanguageModel(vocab)
print('load the model')
with open("/content/model.pkl","rb") as f:
    model=pickle.load(f)
model=model.to(device)

while True:
    prompt=input("Enter prompt (or 'exit' to quit): ")
    context=torch.tensor(encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    generated=model.generate(context, max_new_tokens=100, temperature=0.8, top_k=40)
    print(decode(generated[0].tolist()))
