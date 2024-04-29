from torch import nn, einsum
from torch.optim import Adam
from math import log2, floor
import torch
import torch.nn.functional as F
from kornia.filters import filter2d
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange



def set_requires_grad(model, bool):
    for p in model.parameters():
        p.requires_grad = bool

def is_power_of_two(val):
    return log2(val).is_integer()


def default(val, d):
    if val is None:
        return d
    return val


class GlobalContext(nn.Module):
    def __init__(
        self,
        *,
        chan_in,
        chan_out
    ):
        super().__init__()
        self.to_k = nn.Conv2d(chan_in, 1, 1)
        chan_intermediate = max(3, chan_out // 2)

        self.net = nn.Sequential(
            nn.Conv2d(chan_in, chan_intermediate, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(chan_intermediate, chan_out, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        context = self.to_k(x)
        context = context.flatten(2).softmax(dim = -1)
        out = einsum('b i n, b c n -> b c i', context, x.flatten(2))
        out = out.unsqueeze(-1)
        return self.net(out)
    

class PixelShuffleUpsample(nn.Module):
    def __init__(self, dim, dim_out = None):
        super().__init__()
        dim_out = default(dim_out, dim)
        conv = nn.Conv2d(dim, dim_out * 4, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            nn.PixelShuffle(2)
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // 4, i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o 4) ...')

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)


class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)
    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f [None, :, None]
        return filter2d(x, f, normalized=True)
    

def Conv2dSame(dim_in, dim_out, kernel_size, bias = True):
    pad_left = kernel_size // 2
    pad_right = (pad_left - 1) if (kernel_size % 2) == 0 else pad_left

    return nn.Sequential(
        nn.ZeroPad2d((pad_left, pad_right, pad_left, pad_right)),
        nn.Conv2d(dim_in, dim_out, kernel_size, bias = bias)
    )


class Noise(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x, noise = None):
        b, _, h, w, device = *x.shape, x.device

        if noise is None:
            noise = torch.randn(b, 1, h, w, device = device)

        return x + self.weight * noise
    

class ChanNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = ChanNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x
    
class SumBranches(nn.Module):
    def __init__(self, branches):
        super().__init__()
        self.branches = nn.ModuleList(branches)
    def forward(self, x):
        return sum(map(lambda fn: fn(x), self.branches))
    

def SPConvDownsample(dim, dim_out = None):
    # https://arxiv.org/abs/2208.03641 shows this is the most optimal way to downsample
    # named SP-conv in the paper, but basically a pixel unshuffle
    dim_out = default(dim_out, dim)
    return nn.Sequential(
        Rearrange('b c (h s1) (w s2) -> b (c s1 s2) h w', s1 = 2, s2 = 2),
        nn.Conv2d(dim * 4, dim_out, 1)
    )

class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding = 0, stride = 1, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)
    
class LinearAttention(nn.Module):
    def __init__(self, dim, dim_head = 64, heads = 8, kernel_size = 3):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads

        self.kernel_size = kernel_size
        self.nonlin = nn.GELU()

        self.to_lin_q = nn.Conv2d(dim, inner_dim, 1, bias = False)
        self.to_lin_kv = DepthWiseConv2d(dim, inner_dim * 2, 3, padding = 1, bias = False)

        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias = False)
        self.to_kv = nn.Conv2d(dim, inner_dim * 2, 1, bias = False)

        self.to_out = nn.Conv2d(inner_dim * 2, dim, 1)

    def forward(self, fmap):
        h, x, y = self.heads, *fmap.shape[-2:]

        # linear attention

        lin_q, lin_k, lin_v = (self.to_lin_q(fmap), *self.to_lin_kv(fmap).chunk(2, dim = 1))
        lin_q, lin_k, lin_v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = h), (lin_q, lin_k, lin_v))

        lin_q = lin_q.softmax(dim = -1)
        lin_k = lin_k.softmax(dim = -2)

        lin_q = lin_q * self.scale

        context = einsum('b n d, b n e -> b d e', lin_k, lin_v)
        lin_out = einsum('b n d, b d e -> b n e', lin_q, context)
        lin_out = rearrange(lin_out, '(b h) (x y) d -> b (h d) x y', h = h, x = x, y = y)

        # conv-like full attention

        q, k, v = (self.to_q(fmap), *self.to_kv(fmap).chunk(2, dim = 1))
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) c x y', h = h), (q, k, v))

        k = F.unfold(k, kernel_size = self.kernel_size, padding = self.kernel_size // 2)
        v = F.unfold(v, kernel_size = self.kernel_size, padding = self.kernel_size // 2)

        k, v = map(lambda t: rearrange(t, 'b (d j) n -> b n j d', d = self.dim_head), (k, v))

        q = rearrange(q, 'b c ... -> b (...) c') * self.scale

        sim = einsum('b i d, b i j d -> b i j', q, k)
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()

        attn = sim.softmax(dim = -1)

        full_out = einsum('b i j, b i j d -> b i d', attn, v)
        full_out = rearrange(full_out, '(b h) (x y) d -> b (h d) x y', h = h, x = x, y = y)

        # add outputs of linear attention + conv like full attention

        lin_out = self.nonlin(lin_out)
        out = torch.cat((lin_out, full_out), dim = 1)
        return self.to_out(out)
    

class SimpleDecoder(nn.Module):
    def __init__(
        self,
        *,
        chan_in,
        chan_out = 3,
        num_upsamples = 4,
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        final_chan = chan_out
        chans = chan_in

        for ind in range(num_upsamples):
            last_layer = ind == (num_upsamples - 1)
            chan_out = chans if not last_layer else final_chan * 2
            layer = nn.Sequential(
                PixelShuffleUpsample(chans),
                nn.Conv2d(chans, chan_out, 3, padding = 1),
                nn.GLU(dim = 1)
            )
            self.layers.append(layer)
            chans //= 2

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class Generator(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        latent_dim = 256,
        fmap_max = 512,
        fmap_inverse_coef = 12,
        transparent = False,
        greyscale = False,
        attn_res_layers = [],
        freq_chan_attn = False
    ):
        super().__init__()
        resolution = log2(image_size)
        assert is_power_of_two(image_size), 'image size must be a power of 2'

        if transparent:
            init_channel = 4
        elif greyscale:
            init_channel = 1
        else:
            init_channel = 3

        fmap_max = default(fmap_max, latent_dim)

        self.initial_conv = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, latent_dim * 2, 4),
            nn.BatchNorm2d(latent_dim * 2),
            nn.GLU(dim = 1)
        )

        num_layers = int(resolution) - 2
        features = list(map(lambda n: (n,  2 ** (fmap_inverse_coef - n)), range(2, num_layers + 2)))
        features = list(map(lambda n: (n[0], min(n[1], fmap_max)), features))
        features = list(map(lambda n: 3 if n[0] >= 8 else n[1], features))
        features = [latent_dim, *features]

        in_out_features = list(zip(features[:-1], features[1:]))

        self.res_layers = range(2, num_layers + 2)
        self.layers = nn.ModuleList([])
        self.res_to_feature_map = dict(zip(self.res_layers, in_out_features))

        self.sle_map = ((3, 7), (4, 8), (5, 9), (6, 10))
        self.sle_map = list(filter(lambda t: t[0] <= resolution and t[1] <= resolution, self.sle_map))
        self.sle_map = dict(self.sle_map)

        self.num_layers_spatial_res = 1

        for (res, (chan_in, chan_out)) in zip(self.res_layers, in_out_features):
            image_width = 2 ** res

            attn = None
            # if image_width in attn_res_layers:
            #     attn = PreNorm(chan_in, LinearAttention(chan_in))

            sle = None
            if res in self.sle_map:
                residual_layer = self.sle_map[res]
                sle_chan_out = self.res_to_feature_map[residual_layer - 1][-1]
                sle = GlobalContext(
                    chan_in = chan_out,
                    chan_out = sle_chan_out
                )

            layer = nn.ModuleList([
                nn.Sequential(
                    PixelShuffleUpsample(chan_in),
                    Blur(),
                    Conv2dSame(chan_in, chan_out * 2, 4),
                    Noise(),
                    nn.BatchNorm2d(chan_out * 2),
                    nn.GLU(dim = 1)
                ),
                sle,
                attn
            ])
            self.layers.append(layer)

        self.out_conv = nn.Conv2d(features[-1], init_channel, 3, padding = 1)

    def forward(self, x):
        x = rearrange(x, 'b c -> b c () ()')
        x = self.initial_conv(x)
        x = F.normalize(x, dim = 1)

        residuals = dict()

        for (res, (up, sle, attn)) in zip(self.res_layers, self.layers):
            if attn is not None:
                x = attn(x) + x

            x = up(x)

            if sle is not None:
                out_res = self.sle_map[res]
                residual = sle(x)
                residuals[out_res] = residual

            next_res = res + 1
            if next_res in residuals:
                x = x * residuals[next_res]

        return self.out_conv(x)


class Discriminator(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        fmap_max = 512,
        fmap_inverse_coef = 12,
        transparent = False,
        greyscale = False,
        disc_output_size = 5,
        attn_res_layers = []
    ):
        super().__init__()
        resolution = log2(image_size)
        assert is_power_of_two(image_size), 'image size must be a power of 2'
        assert disc_output_size in {1, 5}, 'discriminator output dimensions can only be 5x5 or 1x1'

        resolution = int(resolution)

        if transparent:
            init_channel = 4
        elif greyscale:
            init_channel = 1
        else:
            init_channel = 3

        num_non_residual_layers = max(0, int(resolution) - 8)
        num_residual_layers = 8 - 3

        non_residual_resolutions = range(min(8, resolution), 2, -1)
        features = list(map(lambda n: (n,  2 ** (fmap_inverse_coef - n)), non_residual_resolutions))
        features = list(map(lambda n: (n[0], min(n[1], fmap_max)), features))

        if num_non_residual_layers == 0:
            res, _ = features[0]
            features[0] = (res, init_channel)

        chan_in_out = list(zip(features[:-1], features[1:]))

        self.non_residual_layers = nn.ModuleList([])
        for ind in range(num_non_residual_layers):
            first_layer = ind == 0
            last_layer = ind == (num_non_residual_layers - 1)
            chan_out = features[0][-1] if last_layer else init_channel

            self.non_residual_layers.append(nn.Sequential(
                Blur(),
                nn.Conv2d(init_channel, chan_out, 4, stride = 2, padding = 1),
                nn.LeakyReLU(0.1)
            ))

        self.residual_layers = nn.ModuleList([])

        for (res, ((_, chan_in), (_, chan_out))) in zip(non_residual_resolutions, chan_in_out):
            image_width = 2 ** res

            attn = None

            self.residual_layers.append(nn.ModuleList([
                SumBranches([
                    nn.Sequential(
                        Blur(),
                        SPConvDownsample(chan_in, chan_out),
                        nn.LeakyReLU(0.1),
                        nn.Conv2d(chan_out, chan_out, 3, padding = 1),
                        nn.LeakyReLU(0.1)
                    ),
                    nn.Sequential(
                        Blur(),
                        nn.AvgPool2d(2),
                        nn.Conv2d(chan_in, chan_out, 1),
                        nn.LeakyReLU(0.1),
                    )
                ]),
                attn
            ]))

        last_chan = features[-1][-1]
        if disc_output_size == 5:
            self.to_logits = nn.Sequential(
                nn.Conv2d(last_chan, last_chan, 1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(last_chan, 1, 4)
            )
        elif disc_output_size == 1:
            self.to_logits = nn.Sequential(
                Blur(),
                nn.Conv2d(last_chan, last_chan, 3, stride = 2, padding = 1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(last_chan, 1, 4)
            )

        self.to_shape_disc_out = nn.Sequential(
            nn.Conv2d(init_channel, 64, 3, padding = 1),
            Residual(PreNorm(64, LinearAttention(64))),
            SumBranches([
                nn.Sequential(
                    Blur(),
                    SPConvDownsample(64, 32),
                    nn.LeakyReLU(0.1),
                    nn.Conv2d(32, 32, 3, padding = 1),
                    nn.LeakyReLU(0.1)
                ),
                nn.Sequential(
                    Blur(),
                    nn.AvgPool2d(2),
                    nn.Conv2d(64, 32, 1),
                    nn.LeakyReLU(0.1),
                )
            ]),
            Residual(PreNorm(32, LinearAttention(32))),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Conv2d(32, 1, 4)
        )

        self.decoder1 = SimpleDecoder(chan_in = last_chan, chan_out = init_channel)
        self.decoder2 = SimpleDecoder(chan_in = features[-2][-1], chan_out = init_channel) if resolution >= 9 else None


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new
    
class LightWeightGan(nn.Module):
    def __init__(
        self,
        *,
        latent_dim,
        image_size,
        optimizer = "adam",
        fmap_max = 512,
        fmap_inverse_coef = 12,
        transparent = False,
        greyscale = False,
        disc_output_size = 5,
        attn_res_layers = [],
        freq_chan_attn = False,
        ttur_mult = 1.,
        lr = 2e-4,
        rank = 0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size

        G_kwargs = dict(
            image_size = image_size,
            latent_dim = latent_dim,
            fmap_max = fmap_max,
            fmap_inverse_coef = fmap_inverse_coef,
            transparent = transparent,
            greyscale = greyscale,
            attn_res_layers = attn_res_layers,
            freq_chan_attn = freq_chan_attn
        )

        self.G = Generator(**G_kwargs)

        self.D = Discriminator(
            image_size = image_size,
            fmap_max = fmap_max,
            fmap_inverse_coef = fmap_inverse_coef,
            transparent = transparent,
            greyscale = greyscale,
            attn_res_layers = attn_res_layers,
            disc_output_size = disc_output_size
        )

        self.ema_updater = EMA(0.995)
        self.GE = Generator(**G_kwargs)
        set_requires_grad(self.GE, False)


        if optimizer == "adam":
            self.G_opt = Adam(self.G.parameters(), lr = lr, betas=(0.5, 0.9))
            self.D_opt = Adam(self.D.parameters(), lr = lr * ttur_mult, betas=(0.5, 0.9))
        else:
            assert False, "No valid optimizer is given"

        self.apply(self._init_weights)
        self.reset_parameter_averaging()


    def _init_weights(self, m):
        if type(m) in {nn.Conv2d, nn.Linear}:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def EMA(self):
        def update_moving_average(ma_model, current_model):
            for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
                old_weight, up_weight = ma_params.data, current_params.data
                ma_params.data = self.ema_updater.update_average(old_weight, up_weight)

            for current_buffer, ma_buffer in zip(current_model.buffers(), ma_model.buffers()):
                new_buffer_value = self.ema_updater.update_average(ma_buffer, current_buffer)
                ma_buffer.copy_(new_buffer_value)

        update_moving_average(self.GE, self.G)

    def reset_parameter_averaging(self):
        self.GE.load_state_dict(self.G.state_dict())