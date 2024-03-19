import torch
import torch.nn as nn
import torch.nn.functional as F

'''
class ConvDecoder(nn.Module):
    def __init__(self, b_size=32, inp_dim=64):
        super(ConvDecoder, self).__init__()
        self.dconv1 = nn.ConvTranspose2d(inp_dim,128,3,stride=2, padding=1)
        self.bn1    = nn.BatchNorm2d(128)
        self.dconv2 = nn.ConvTranspose2d(128,64,3,stride=2, padding=1)
        self.bn2    = nn.BatchNorm2d(64)
        self.dconv3 = nn.ConvTranspose2d(64,32,3,stride=2, padding=1)
        self.bn3    = nn.BatchNorm2d(32)
        self.dconv4 = nn.ConvTranspose2d(32,1,3,stride=1, padding=1)

        self.size1  = torch.Size([b_size * 10, 128, 16, 16])
        self.size2  = torch.Size([b_size * 10, 64, 32, 32])
        self.size3  = torch.Size([b_size * 10, 32, 64, 64])

        self.inp_dim = inp_dim

    def forward(self, input):
        h1 = self.bn1(self.dconv1(input, self.size1))
        a1 = F.elu(h1)
        h2 = self.bn2(self.dconv2(a1, self.size2))
        a2 = F.elu(h2)
        h3 = self.bn3(self.dconv3(a2, self.size3))
        a3 = F.elu(h3)
        h4 = self.dconv4(a3)
        return h4
'''

input_size = 64
hidden_size = 64
channel_size = 320
output_size = 64

class ConvEncoder(nn.Module):
    def __init__(self, output_size):
        super(ConvEncoder, self).__init__()

        # layer 1: input layer
        self.fc0 = nn.Linear(input_size, input_size)

        # Initialize the p_mask hyperparameter
        self.p_mask = nn.Parameter(torch.randn(input_size), requires_grad=True)

        # layer 2: p set output
        self.p_output_layer = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(input_size, input_size),
            nn.ReLU(True)
        )
        self.Wp = nn.Parameter(
            torch.tril(torch.randn(channel_size, channel_size)), requires_grad=True
        )
        self.Wp.data.diagonal().fill_(1)
        self.Wp_diag = nn.Parameter(
            torch.ones(channel_size), requires_grad=True
        )

        # layer 3: z set output
        self.z_output_layer = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(input_size, input_size),
            nn.ReLU(True)
        )
        self.Wz = nn.Parameter(
            torch.randn(channel_size, channel_size), requires_grad=True
        )

        # layer 4: fully connected output
        self.fc1 = nn.Sequential(
            nn.Linear(input_size, 4*output_size),
            nn.ReLU(True)
        )

        # MLP layer
        self.fc2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(True)
        )
        self.fc3 = nn.Linear(16, 16)

    def split_input(self, x):
        # Use the p_mask to split the input into p and z sets
        p_mask = self.p_mask.to(x.device) > 0

        p_set = x * p_mask
        
        return p_set
    
    def forward(self, x):
        x = self.fc0(x)
        x = x.view(channel_size, 1, hidden_size, input_size)

        # Split input into p and z sets
        p_set = self.split_input(x)
        p_set = p_set.to(torch.float32)

        with torch.no_grad():
            self.Wp.data = torch.tril(self.Wp.data)
            self.Wp.data.diagonal().clamp_(min=0, max=1)
        Wp = self.Wp + torch.diag(self.Wp_diag)
        p_output = torch.mm(Wp, p_set.view(channel_size, -1))
        p_output = p_output.view(channel_size, 1, hidden_size, input_size)
        p_output = self.p_output_layer(p_output)

        Wz = self.Wz
        z_output = torch.mm(Wz, p_output.view(channel_size, -1))
        z_output = z_output.view(channel_size, 1, hidden_size, input_size)
        z_output = self.z_output_layer(z_output)
        
        combined_output = p_output + z_output
        output = self.fc1(combined_output)

        return output

class ConvDecoder(nn.Module):
    def __init__(self, b_size=32, inp_dim=64):
        super(ConvDecoder, self).__init__()
        self.dconv1 = nn.ConvTranspose2d(inp_dim,64,3,stride=2, padding=1)
        self.bn1    = nn.BatchNorm2d(64)
        self.dconv2 = nn.ConvTranspose2d(64,32,3,stride=1, padding=1)
        self.bn2    = nn.BatchNorm2d(32)
        self.dconv3 = nn.ConvTranspose2d(32,16,3,stride=2, padding=1)
        self.bn3    = nn.BatchNorm2d(16)
        self.dconv4 = nn.ConvTranspose2d(16,1,3,stride=1, padding=1)

        self.size1  = torch.Size([b_size * 10, 64, 32, 32])
        self.size2  = torch.Size([b_size * 10, 16, 64, 64])

        self.inp_dim = inp_dim

    def forward(self, input):
        # print("ConvDecoder",input.shape)
        h1 = self.bn1(self.dconv1(input, self.size1))
        a1 = F.elu(h1)
        h2 = self.bn2(self.dconv2(a1))
        a2 = F.elu(h2)
        h3 = self.bn3(self.dconv3(a2, self.size2))
        a3 = F.elu(h3)
        h4 = self.dconv4(a3)
        # print("ConvDecoder",h4.shape)
        return h4



