import torch
import torch.nn as nn
import torch.nn.functional as F


# Positional Encoding
# borrow this implementation from nerf-pytorch
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

# TODO: same architecture as in NeRFactor. The NeRFactor says it uses 4 layers, but according to the released codes, it's 4 + 1
class ShapeModel(nn.Module):
    def __init__(self, net_depth=4, net_width=128, pos_pe=10, view_dir_pe=4, skips=[2], use_view_dir=False, out_ch=1, act_out=torch.sigmoid):
        ''' Shape network used to cache shape related features such as normals and visibilty
        - net_depth: number of layers in the network
        - net_width: width of the network
        - position_pe: encoding levels of 3D locations
        - view_dir_pe: embedder for view direction encoding levels of view directions
        - skips: list of skip connections
        - use_view_dir: whether to use view direction as the network's input
        - out_ch: number of output channels, 1 for visibility, 3 for normals
        - act_out: activation function for the network's last output layer, Sigmoid for visibility
        '''
        super(ShapeModel, self).__init__()
        self.net_depth = net_depth
        self.net_width = net_width
        self.pos_pe = pos_pe
        self.view_dir_pe = view_dir_pe
        self.skips = skips
        self.use_view_dir = use_view_dir
        self.out_ch = out_ch
        self.act_out = act_out # activation function for the final output layer
        
        # position encoding functions for position and view direction
        self.pos_pe_fn, self.pos_pe_dim = get_embedder(self.pos_pe)
        self.view_dir_pe_fn, self.view_dir_pe_dim = get_embedder(self.view_dir_pe)

        self.in_dim = self.pos_pe_dim + self.view_dir_pe_dim

        self.linear_layers = nn.ModuleList([nn.Linear(self.in_dim, self.net_width)])
        for i in range(0, self.net_depth-1): # TODO: nerfactor paper says the skip is concatenated with the output of the second layer, but in its codes, skip should be connected after third layer 
            if i in self.skips:
                self.linear_layers.append(nn.Linear(self.in_dim + self.net_width, self.net_width))
            else:
                self.linear_layers.append(nn.Linear(self.net_width, self.net_width))

        self.out_layer = nn.Linear(self.net_width, self.out_ch)
        

    def forward(self, pos, view_dir):

        # positional encoding 
        pos_pe = self.pos_pe_fn(pos)
        view_dir_pe = self.view_dir_pe_fn(view_dir)
        net_input = torch.cat((pos_pe, view_dir_pe), dim=-1)
        h = net_input
        for i, l in enumerate(self.linear_layers):
            h = F.relu(self.linear_layers[i](h))
            if i in self.skips:
                h = torch.cat((net_input, h), dim=-1)
        net_output = self.act_out(self.out_layer(h))

        return net_output





# used for test
if __name__ == "__main__":
    visibility_net = ShapeModel()
    pos = torch.randn((10, 3))
    dir = torch.randn_like(pos)

    out = visibility_net(pos, dir)
    print(out.shape)
    print(out)
    print(visibility_net)
