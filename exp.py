import random
all_seqs = {"cat", "dog", "ant", "bat", "This is a sentence."}

print(sorted(list(all_seqs), key=len, reverse=True))

print(sorted(list(all_seqs), key=lambda s: (-len(s), s)))