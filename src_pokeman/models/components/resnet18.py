from torch import nn


class Backbone(nn.Module):
    # def __init__(
    #     self,
    #     # input_size: int = 784,
    #     # lin1_size: int = 256,
    #     # lin2_size: int = 256,
    #     # lin3_size: int = 256,
    #     # output_size: int = 10,
    # ):
    #     super().__init__()
    #     self.model = nn.Sequential(
    #         nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
    #         nn.MaxPool2d(kernel_size=2, stride=2),
    #         nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
    #         nn.MaxPool2d(kernel_size=2, stride=2),
    #         nn.Dropout2d(p=0.1),
    #         nn.AdaptiveMaxPool2d((1, 1)),
    #         nn.Flatten(),
    #         nn.Linear(64, 32),
    #         nn.ReLU(),
    #         nn.Linear(32, 5),
    #     )
    #
    # def forward(self, x):
    #     batch_size, channels, width, height = 64,3,224,224
    #
    #     # (batch, 1, width, height) -> (batch, 1*width*height)
    #     # x = x.view(batch_size, -1)
    #     # print("-----------------------------",x)
    #
    #     return self.model(x)
    def __init__(self):
        super().__init__()
        self.l1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.l2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.l3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.l4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.l5 = nn.Dropout2d(p=0.1)
        self.l6 = nn.AdaptiveMaxPool2d((1, 1))
        self.l7 = nn.Flatten()
        self.l8 = nn.Linear(64, 32)
        self.l9 = nn.ReLU()
        self.l10 = nn.Linear(32, 5)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        return x

if __name__ == "__main__":
    _ = Backbone()
