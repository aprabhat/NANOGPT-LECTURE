# read the data file
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


