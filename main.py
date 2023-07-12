import tiktoken
import torch
import os

import model

print(torch.cuda.is_available())

# enc = tiktoken.encoding_for_model("text-davinci-003")
# assert enc.decode(enc.encode('hello world')) == 'hello world'
#


def encode(text):
    return [ord(a) for a in text]


def decode(data):
    return ''.join(map(chr, data))


if __name__ == '__main__':

    torch.manual_seed(42069)

    raw_dataset = open('tinyshakespeare.txt').read()
    dataset = encode(raw_dataset)
    dataset = torch.tensor(dataset, dtype=torch.long)
    print(dataset)
    vocab_size = 128
    print(f'vocab_size: {vocab_size}')
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

    model = model.LanguageModel(vocab_size=vocab_size, embed_size=384, block_size=block_size, batch_size=batch_size)
    x, y = get_batch()
    logits, loss = model(x, y)
    print(f'loss: {loss}')
    print(decode(model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    epochs = 100000
    for epoch in range(epochs):
        logits, loss = model(*get_batch())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        print(f'loss: {loss}')

    print(decode(model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))