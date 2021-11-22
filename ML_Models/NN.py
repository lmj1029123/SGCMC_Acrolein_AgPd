import torch 

class MultiLayerNet(torch.nn.Module):
    def __init__(self, N_sym, n_nodes, activations, N_element, bias = True, scaling = None):

        super(MultiLayerNet, self).__init__()
        N_layers = len(n_nodes)
        if N_layers == 0:
            self.net = torch.nn.Linear(N_sym, N_element, bias = bias)
        else:
            layers = []
            for n in range(N_layers):
                if n == 0:
                    layers += [torch.nn.Linear(N_sym, n_nodes[n], bias = bias)]
                    layers += [activations[n]]
                else:
                    layers += [torch.nn.Linear(n_nodes[n-1], n_nodes[n], bias = bias)]
                    layers += [activations[n]]
            layers += [torch.nn.Linear(n_nodes[-1], N_element, bias = bias)]
            self.net = torch.nn.Sequential(*layers)
        
        self.scaling = scaling




    def forward(self, x):

        y_pred = self.net(x)
        return y_pred




