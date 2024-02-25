# read the data file
with open('internet_archive_scifi_v3.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("length of dataset in characters: ", len(text))
