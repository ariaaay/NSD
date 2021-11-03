import torch
import argparse
import numpy as np

from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from nsd_dataset import NSDBrainOnlyDataset
from util.model_config import roi_name_dict


class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=64
        )
        self.encoder_output_layer = nn.Linear(
            in_features=64, out_features=kwargs["latent_dim"]
        )
        self.decoder_hidden_layer = nn.Linear(kwargs["latent_dim"], out_features=64)
        self.decoder_output_layer = nn.Linear(
            in_features=64, out_features=kwargs["input_shape"]
        )

    def forward(self, x):
        activation = self.encoder_hidden_layer(x)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    parser = argparse.ArgumentParser()
    parser.add_argument("--roi", type=str)
    parser.add_argument("--subj", type=int, default=1)
    parser.add_argument("--roi_num", type=int)
    parser.add_argument(
        "--output_dir", type=str, default="/user_data/yuanw3/project_outputs/NSD/output"
    )

    args = parser.parse_args()
    assert args.roi_num > 0
    roi_label = roi_name_dict[args.roi][args.roi_num]

    n = 10000
    epochs = 200
    latent_dims = np.linspace(5, 100, 6)
    (train_idx, test_idx,) = train_test_split(
        np.arange(n), test_size=0.15, random_state=42
    )

    train_data = NSDBrainOnlyDataset(
        args.output_dir, args.subj, train_idx, args.roi, args.roi_num
    )
    test_data = NSDBrainOnlyDataset(
        args.output_dir, args.subj, test_idx, args.roi, args.roi_num
    )

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=80, shuffle=True)

    for sample in train_loader:
        k = sample.shape[1]
        break

    for d in latent_dims:

        model = AE(input_shape=k, latent_dim=int(d)).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        train_losses = []
        for epoch in range(epochs):
            loss = 0
            model.train()

            for batch in train_loader:
                batch = batch.float().to(device)
                optimizer.zero_grad()
                outputs = model(batch)
                train_loss = criterion(outputs, batch)

                train_loss.backward()
                optimizer.step()

                loss += train_loss.item()

            # compute the epoch training loss
            loss = loss / len(train_loader)
            train_losses.append(loss)

            # display the epoch training loss
            print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))

            # testing
            test_losses = []
            test_loss_epoch = 0
            model.eval()
            for batch in test_loader:
                batch = batch.float().to(device)
                outputs = model(batch)
                test_loss = criterion(outputs, batch)
                test_loss_epoch += test_loss.item()

            test_loss_epoch = test_loss_epoch / len(test_loader)
            test_losses.append(test_loss_epoch)

        np.save(
            "%s/autoencode/train_loss_dim%d_%s_%s.npy"
            % (args.output_dir, d, args.roi, roi_label),
            train_losses,
        )
        np.save(
            "%s/autoencode/test_loss_dim%d_%s_%s.npy"
            % (args.output_dir, d, args.roi, roi_label),
            test_losses,
        )
