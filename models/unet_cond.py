import torch
import torch.nn as nn
import torch.nn.functional as F

from models.unet import EncoderBlock, DecoderBlock, MidBlock


class UNetCond(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=2,
        base_dim=32,
        dim_mults=[2, 4, 8, 16],
        n_classes=10,
        class_channels=0,
        time_embedding_dim=1,
        cond_channels=0,
    ):
        super().__init__()
        assert base_dim % 2 == 0

        self.in_channels = in_channels
        self.out_channels = out_channels

        channels_down, channels_up = self._cal_channels(
            base_dim, dim_mults
        )

        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_dim),
            nn.SiLU(inplace=True),
        )
        self.time_embedding = nn.Linear(1, time_embedding_dim) 

        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(in_ch, out_ch)
                for (in_ch, out_ch) in channels_down
            ]
        )

        total_cond_dim = time_embedding_dim + class_channels + cond_channels
        self.mid_block = MidBlock(channels_down[-1][1], total_cond_dim)

        self.class_embed = (
            nn.Embedding(n_classes, class_channels) if class_channels > 0 else None
        )

        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(in_ch, out_ch)
                for (in_ch, out_ch) in channels_up
            ]
        )

        self.final_conv = nn.Conv2d(
            channels_up[-1][-1] // 2, out_channels, kernel_size=1
        )

    def forward(self, x, classes, t, condition=None):
        x = self.init_conv(x)

        conds = []
        conds.append(self.time_embedding(t))
        if self.class_embed is not None:
            conds.append(self.class_embed(classes).unsqueeze(-1).unsqueeze(-1))
        if condition is not None:
            conds.append(condition.unsqueeze(-1).unsqueeze(-1))
        cond = torch.cat(conds, dim=1) if conds else None

        encoder_shortcuts = []
        for encoder_block in self.encoder_blocks:
            x, x_skip = encoder_block(x)
            encoder_shortcuts.append(x_skip)

        x = self.mid_block(x, cond)

        encoder_shortcuts.reverse()
        for decoder_block, x_skip in zip(self.decoder_blocks, encoder_shortcuts):
            x = decoder_block(x, x_skip)

        x = self.final_conv(x)
        return x

    def _cal_channels(self, base_dim, dim_mults):
        dims = [base_dim * mult for mult in dim_mults]
        dims.insert(0, base_dim)
        channels_down = []
        for i in range(len(dims) - 1):
            channels_down.append((dims[i], dims[i + 1]))
        channels_up = []
        for i in range(1, len(dims)):
            channels_up.append((dims[-i], dims[-(i + 1)]))
        return channels_down, channels_up


if __name__ == "__main__":
    x = torch.randn(3, 1, 224, 224)
    classes = torch.randint(0, 10, (3,))
    t = torch.rand(x.size(0), 1, 1, 1)

    model = UNetCond(
        in_channels=1,
        out_channels=1,
        base_dim=32,
        dim_mults=[2, 4, 8, 16],
        n_classes=10,
        class_channels=1,
        time_embedding_dim=1,
    )
    y = model(x, classes, t)
    print(y.shape)

    from torchinfo import summary
    summary(model, input_data=(x, classes, t))