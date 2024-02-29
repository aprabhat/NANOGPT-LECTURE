# read the data file
import torch

with open('internet_archive_scifi_v3.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("length of dataset in characters: ", len(text))

# print first 1000 chars
print(text[:1000])

# all unique chars occur in the dataset
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # take string and return a list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # take a list of integers and return a string

print(encode("hi there"))
print(decode(encode("hi there")))

# encode the entire dataset ad store it into a torch tensor
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])  # how chars looks to GPT

# split the dataset into train and validation sets
n = int(0.9 * len(data))
train_data = data[:n]  # first 90%
val_data = data[n:]  # rest 10%

block_size = 8

# example
x = train_data[:block_size]
y = train_data[1:block_size + 1]
for t in range(block_size):
    context = x[:t + 1]
    target = y[t]
    print(f"when input is {context} the target is : {target}")
