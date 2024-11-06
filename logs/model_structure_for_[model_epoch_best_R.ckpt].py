class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, batch_first=True, dim_feedforward=512)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4)
        self.layer1 = nn.Sequential(
            self.transformer_encoder)
        self.layer4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(16))
        self.fc1 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        out = self.layer1(x).transpose(2,1)
        # out = self.layer2(out)
        # out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        rainfall_intensity = self.fc3(out)
        return rainfall_intensity
