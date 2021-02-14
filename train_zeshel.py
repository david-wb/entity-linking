import os

import torch

from src.bi_encoder import BiEncoder
from src.zeshel_dataset import ZeshelDataset

dir_path = os.path.dirname(os.path.realpath(__file__))


def validate(model, valloader):
    with torch.no_grad():
        total_loss = 0
        for batch in valloader:
            me, ee, loss = model(**batch)
            total_loss += loss.item()
        print('Validation loss', total_loss)


def main():
    # torch.cuda.empty_cache()
    # torch.multiprocessing.set_start_method('spawn')
    print('Cuda is available:', torch.cuda.is_available())
    device = 'cpu'  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = BiEncoder(device=device)
    model.to(device)

    trainset = ZeshelDataset(os.path.join(dir_path, 'zeshel_transformed'), split='train', device=device)
    valset = ZeshelDataset(os.path.join(dir_path, 'zeshel_transformed'), split='val', device=device)
    valset = [valset[i] for i in range(100)]
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=4, shuffle=True, num_workers=2)

    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for i, batch in enumerate(trainloader):
        model.zero_grad()
        me, ee, loss = model(**batch)
        loss.backward()
        optimizer.step()
        print(loss.item())

        if i % 10 == 0:
            validate(model, valloader)


if __name__ == '__main__':
    main()