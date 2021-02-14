import os

import torch

from src.bi_encoder import BiEncoder
from src.zeshel_dataset import ZeshelDataset

dir_path = os.path.dirname(os.path.realpath(__file__))


def validate(model, valloader):
    total_loss = 0
    for batch in valloader:
        me, ee, loss = model(**batch)
        total_loss += loss.item()
    print('Validation loss', total_loss)


def main():
    model = BiEncoder()
    trainset = ZeshelDataset(os.path.join(dir_path, 'zeshel_transformed'), split='train')
    valset = ZeshelDataset(os.path.join(dir_path, 'zeshel_transformed'), split='val')
    valset = [valset[i] for i in range(30)]
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=4, shuffle=True, num_workers=2)

    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for i, batch in enumerate(trainloader):

        model.zero_grad()
        me, ee, loss = model(**batch)
        print(loss.item())
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            validate(model, valloader)


if __name__ == '__main__':
    main()