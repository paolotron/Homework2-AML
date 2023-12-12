import torch
import torch.nn as nn
from torch.autograd import Variable

# LSTM
class MyLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(MyLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lin_i_xx = nn.Linear(input_size, hidden_size)
        self.lin_i_hh = nn.Linear(hidden_size, hidden_size, bias=False)

        self.lin_f_xx = nn.Linear(input_size, hidden_size)
        self.lin_f_hh = nn.Linear(hidden_size, hidden_size, bias=False)

        self.lin_c_xx = nn.Linear(input_size, hidden_size)
        self.lin_c_hh = nn.Linear(hidden_size, hidden_size, bias=False)

        self.lin_o_xx = nn.Linear(input_size, hidden_size)
        self.lin_o_hh = nn.Linear(hidden_size, hidden_size, bias=False)


    def forward(self, x, state):
        if state is None:
            device = x.device
            state = (Variable(torch.randn(x.size(0), x.size(1)).to(device)),
                     Variable(torch.randn(x.size(0), x.size(1)).to(device)))
        ht_1, ct_1 = state
        it = torch.sigmoid(self.lin_i_xx(x) + self.lin_i_hh(ht_1))
        ft = torch.sigmoid(self.lin_f_xx(x) + self.lin_f_hh(ht_1))
        ct_tilde = torch.tanh(self.lin_c_xx(x) + self.lin_c_hh(ht_1))
        ct = (ct_tilde * it) + (ct_1 * ft)
        ot = torch.sigmoid(self.lin_o_xx(x) + self.lin_o_hh(ht_1))
        ht = ot * torch.tanh(ct)
        return ht, ct


#ConvLSTM
class MyConvLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, kernel_size=3, stride=1, padding=1):
        super(MyConvLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv_i_xx = nn.Conv2d(input_size, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_i_hh = nn.Conv2d(hidden_size, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding,
                                   bias=False)

        self.conv_f_xx = nn.Conv2d(input_size, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_f_hh = nn.Conv2d(hidden_size, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding,
                                   bias=False)

        self.conv_c_xx = nn.Conv2d(input_size, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_c_hh = nn.Conv2d(hidden_size, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding,
                                   bias=False)

        self.conv_o_xx = nn.Conv2d(input_size, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_o_hh = nn.Conv2d(hidden_size, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding,
                                   bias=False)


    def forward(self, x, state):
        if state is None:
            device = x.device
            state = (Variable(torch.randn(x.size(0), x.size(1), x.size(2), x.size(3)).to(device)),
                     Variable(torch.randn(x.size(0), x.size(1), x.size(2), x.size(3)).to(device)))
        ht_1, ct_1 = state
        it = torch.sigmoid(self.conv_i_xx(x) + self.conv_i_hh(ht_1))
        ft = torch.sigmoid(self.conv_f_xx(x) + self.conv_f_hh(ht_1))
        ct_tilde = torch.tanh(self.conv_c_xx(x) + self.conv_c_hh(ht_1))
        ct = (ct_tilde * it) + (ct_1 * ft)
        ot = torch.sigmoid(self.conv_o_xx(x) + self.conv_o_hh(ht_1))
        ht = ot * torch.tanh(ct)
        return ht, ct
