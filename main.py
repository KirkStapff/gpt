import tiktoken
import torch

import model

print(torch.cuda.is_available())

enc = tiktoken.encoding_for_model("text-davinci-003")
assert enc.decode(enc.encode('hello world')) == 'hello world'

vocab_size = enc.n_vocab
print(f'vocab_size: {vocab_size}')

if __name__ == '__main__':

    torch.manual_seed(42069)

    raw_dataset = open('tinyshakespeare.txt').read()
    dataset = torch.tensor(enc.encode(raw_dataset), dtype=torch.long)

    split = 0.8
    train_set, val_set = dataset[:int(len(dataset)*split)], dataset[int(len(dataset)*split):]

    block_size = 8
    batch_size = 4

    def get_batch(split_type='train'):
        data = train_set if split_type == 'train' else val_set
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i + block_size] for i in ix])
        y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
        return x, y

    bigram = model.BigramLanguageModel(vocab_size)
    logits, loss = bigram(*get_batch())
    print(f'loss: {loss}')
    print(enc.decode(bigram.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

    optimizer = torch.optim.AdamW(bigram.parameters(), lr=1e-3)

    epochs = 100
    for epoch in range(epochs):
        logits, loss = bigram(*get_batch())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        print(f'loss: {loss}')