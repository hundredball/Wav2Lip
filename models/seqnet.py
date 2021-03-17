import torch
from torch import nn
from torch.nn import functional as F
import math
from torchvision.models import vgg19

from .conv import Conv2dTranspose, Conv2d, nonorm_Conv2d


class SeqNetDisc(nn.Module):
    def __init__(self):
        super(SeqNetDisc, self).__init__()

        self.face_encoder = nn.Sequential(
            Conv2d(15, 32, kernel_size=(7, 7), stride=1, padding=3),

            Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            nn.AdaptiveAvgPool2d(2),
        )

        self.lstm_encoder = nn.LSTM(input_size=4, hidden_size=256, num_layers=1, batch_first=True, dropout=0.2,
                                    bias=True)

        self.binary_pred = nn.Sequential(nn.Linear(256, 1), nn.Sigmoid())
        self.label_noise = .0

    def get_lower_half(self, face_sequences):
        return face_sequences[:, :, face_sequences.size(2) // 2:]

    def to_2d(self, face_sequences):
        B = face_sequences.size(0)
        face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=1)
        return face_sequences

    def perceptual_forward(self, false_face_sequences):
        false_face_sequences = self.to_2d(false_face_sequences)

        x = false_face_sequences

        x = self.face_encoder(x)

        # x -> B,S,2,2
        B, S, _, _ = x.shape
        x = x.view(B, S, -1)

        # take output of hidden state at last sequence of size (1,B,H)
        _, (x, _) = self.lstm_encoder(x)

        x = x.view(B, -1)

        output = self.binary_pred(x)

        false_pred_loss = F.binary_cross_entropy(output,
                                                 torch.ones((len(output), 1)).cuda())

        return false_pred_loss

    def forward(self, face_sequences):
        face_sequences = self.to_2d(face_sequences)

        x = face_sequences

        x = self.face_encoder(x)

        # x -> B,S,2,2
        B, S, _, _ = x.shape
        x = x.view(B, S, -1)

        # take output of hidden state at last sequence of size (1,B,H)
        _, (x, _) = self.lstm_encoder(x)

        x = x.view(B, -1)

        output = self.binary_pred(x)

        return output
