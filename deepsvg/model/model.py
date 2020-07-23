from deepsvg.difflib.tensor import SVGTensor
from deepsvg.utils.utils import _pack_group_batch, _unpack_group_batch, _make_seq_first, _make_batch_first

from .layers.transformer import *
from .layers.improved_transformer import *
from .layers.positional_encoding import *
from .basic_blocks import FCN, HierarchFCN, ResNet
from .config import _DefaultConfig
from .utils import (_get_padding_mask, _get_key_padding_mask, _get_group_mask, _get_visibility_mask,
                    _get_key_visibility_mask, _generate_square_subsequent_mask, _sample_categorical, _threshold_sample)

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from scipy.optimize import linear_sum_assignment


class SVGEmbedding(nn.Module):
    def __init__(self, cfg: _DefaultConfig, seq_len, rel_args=False, use_group=True, group_len=None):
        super().__init__()

        self.cfg = cfg

        self.command_embed = nn.Embedding(cfg.n_commands, cfg.d_model)

        args_dim = 2 * cfg.args_dim if rel_args else cfg.args_dim + 1
        self.arg_embed = nn.Embedding(args_dim, 64)
        self.embed_fcn = nn.Linear(64 * cfg.n_args, cfg.d_model)

        self.use_group = use_group
        if use_group:
            if group_len is None:
                group_len = cfg.max_num_groups
            self.group_embed = nn.Embedding(group_len+2, cfg.d_model)

        self.pos_encoding = PositionalEncodingLUT(cfg.d_model, max_len=seq_len+2)

        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.kaiming_normal_(self.command_embed.weight, mode="fan_in")
        nn.init.kaiming_normal_(self.arg_embed.weight, mode="fan_in")
        nn.init.kaiming_normal_(self.embed_fcn.weight, mode="fan_in")

        if self.use_group:
            nn.init.kaiming_normal_(self.group_embed.weight, mode="fan_in")

    def forward(self, commands, args, groups=None):
        S, GN = commands.shape

        src = self.command_embed(commands.long()) + \
              self.embed_fcn(self.arg_embed((args + 1).long()).view(S, GN, -1))  # shift due to -1 PAD_VAL

        if self.use_group:
            src = src + self.group_embed(groups.long())

        src = self.pos_encoding(src)

        return src


class ConstEmbedding(nn.Module):
    def __init__(self, cfg: _DefaultConfig, seq_len):
        super().__init__()

        self.cfg = cfg

        self.seq_len = seq_len

        self.PE = PositionalEncodingLUT(cfg.d_model, max_len=seq_len)

    def forward(self, z):
        N = z.size(1)
        src = self.PE(z.new_zeros(self.seq_len, N, self.cfg.d_model))
        return src


class LabelEmbedding(nn.Module):
    def __init__(self, cfg: _DefaultConfig):
        super().__init__()

        self.label_embedding = nn.Embedding(cfg.n_labels, cfg.dim_label)

        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.kaiming_normal_(self.label_embedding.weight, mode="fan_in")

    def forward(self, label):
        src = self.label_embedding(label)
        return src


class Encoder(nn.Module):
    def __init__(self, cfg: _DefaultConfig):
        super().__init__()

        self.cfg = cfg

        seq_len = cfg.max_seq_len if cfg.encode_stages == 2 else cfg.max_total_len
        self.use_group = cfg.encode_stages == 1
        self.embedding = SVGEmbedding(cfg, seq_len, use_group=self.use_group)

        if cfg.label_condition:
            self.label_embedding = LabelEmbedding(cfg)
        dim_label = cfg.dim_label if cfg.label_condition else None

        if cfg.model_type == "transformer":
            encoder_layer = TransformerEncoderLayerImproved(cfg.d_model, cfg.n_heads, cfg.dim_feedforward, cfg.dropout, d_global2=dim_label)
            encoder_norm = LayerNorm(cfg.d_model)
            self.encoder = TransformerEncoder(encoder_layer, cfg.n_layers, encoder_norm)
        else:  # "lstm"
            self.encoder = nn.LSTM(cfg.d_model, cfg.d_model // 2, dropout=cfg.dropout, bidirectional=True)

        if cfg.encode_stages == 2:
            if not cfg.self_match:
                self.hierarchical_PE = PositionalEncodingLUT(cfg.d_model, max_len=cfg.max_num_groups)

            hierarchical_encoder_layer = TransformerEncoderLayerImproved(cfg.d_model, cfg.n_heads, cfg.dim_feedforward, cfg.dropout, d_global2=dim_label)
            hierarchical_encoder_norm = LayerNorm(cfg.d_model)
            self.hierarchical_encoder = TransformerEncoder(hierarchical_encoder_layer, cfg.n_layers, hierarchical_encoder_norm)

    def forward(self, commands, args, label=None):
        S, G, N = commands.shape
        l = self.label_embedding(label).unsqueeze(0).unsqueeze(0).repeat(1, commands.size(1), 1, 1) if self.cfg.label_condition else None

        if self.cfg.encode_stages == 2:
            visibility_mask, key_visibility_mask = _get_visibility_mask(commands, seq_dim=0), _get_key_visibility_mask(commands, seq_dim=0)

        commands, args, l = _pack_group_batch(commands, args, l)
        padding_mask, key_padding_mask = _get_padding_mask(commands, seq_dim=0), _get_key_padding_mask(commands, seq_dim=0)
        group_mask = _get_group_mask(commands, seq_dim=0) if self.use_group else None

        src = self.embedding(commands, args, group_mask)

        if self.cfg.model_type == "transformer":
            memory = self.encoder(src, mask=None, src_key_padding_mask=key_padding_mask, memory2=l)

            z = (memory * padding_mask).sum(dim=0, keepdim=True) / padding_mask.sum(dim=0, keepdim=True)
        else:  # "lstm"
            hidden_cell = (src.new_zeros(2, N, self.cfg.d_model // 2),
                           src.new_zeros(2, N, self.cfg.d_model // 2))
            sequence_lengths = padding_mask.sum(dim=0).squeeze(-1)
            x = pack_padded_sequence(src, sequence_lengths, enforce_sorted=False)

            packed_output, _ = self.encoder(x, hidden_cell)

            memory, _ = pad_packed_sequence(packed_output)
            idx = (sequence_lengths - 1).long().view(1, -1, 1).repeat(1, 1, self.cfg.d_model)
            z = memory.gather(dim=0, index=idx)

        z = _unpack_group_batch(N, z)

        if self.cfg.encode_stages == 2:
            src = z.transpose(0, 1)
            src = _pack_group_batch(src)
            l = self.label_embedding(label).unsqueeze(0) if self.cfg.label_condition else None

            if not self.cfg.self_match:
                src = self.hierarchical_PE(src)

            memory = self.hierarchical_encoder(src, mask=None, src_key_padding_mask=key_visibility_mask, memory2=l)
            z = (memory * visibility_mask).sum(dim=0, keepdim=True) / visibility_mask.sum(dim=0, keepdim=True)
            z = _unpack_group_batch(N, z)

        return z


class VAE(nn.Module):
    def __init__(self, cfg: _DefaultConfig):
        super(VAE, self).__init__()

        self.enc_mu_fcn = nn.Linear(cfg.d_model, cfg.dim_z)
        self.enc_sigma_fcn = nn.Linear(cfg.d_model, cfg.dim_z)

        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.normal_(self.enc_mu_fcn.weight, std=0.001)
        nn.init.constant_(self.enc_mu_fcn.bias, 0)
        nn.init.normal_(self.enc_sigma_fcn.weight, std=0.001)
        nn.init.constant_(self.enc_sigma_fcn.bias, 0)

    def forward(self, z):
        mu, logsigma = self.enc_mu_fcn(z), self.enc_sigma_fcn(z)
        sigma = torch.exp(logsigma / 2.)
        z = mu + sigma * torch.randn_like(sigma)

        return z, mu, logsigma


class Bottleneck(nn.Module):
    def __init__(self, cfg: _DefaultConfig):
        super(Bottleneck, self).__init__()

        self.bottleneck = nn.Linear(cfg.d_model, cfg.dim_z)

    def forward(self, z):
        return self.bottleneck(z)


class Decoder(nn.Module):
    def __init__(self, cfg: _DefaultConfig):
        super(Decoder, self).__init__()

        self.cfg = cfg

        if cfg.label_condition:
            self.label_embedding = LabelEmbedding(cfg)
        dim_label = cfg.dim_label if cfg.label_condition else None

        if cfg.decode_stages == 2:
            self.hierarchical_embedding = ConstEmbedding(cfg, cfg.num_groups_proposal)

            hierarchical_decoder_layer = TransformerDecoderLayerGlobalImproved(cfg.d_model, cfg.dim_z, cfg.n_heads, cfg.dim_feedforward, cfg.dropout, d_global2=dim_label)
            hierarchical_decoder_norm = LayerNorm(cfg.d_model)
            self.hierarchical_decoder = TransformerDecoder(hierarchical_decoder_layer, cfg.n_layers_decode, hierarchical_decoder_norm)
            self.hierarchical_fcn = HierarchFCN(cfg.d_model, cfg.dim_z)

        if cfg.pred_mode == "autoregressive":
            self.embedding = SVGEmbedding(cfg, cfg.max_total_len, rel_args=cfg.rel_targets, use_group=True, group_len=cfg.max_total_len)

            square_subsequent_mask = _generate_square_subsequent_mask(self.cfg.max_total_len+1)
            self.register_buffer("square_subsequent_mask", square_subsequent_mask)
        else:  # "one_shot"
            seq_len = cfg.max_seq_len+1 if cfg.decode_stages == 2 else cfg.max_total_len+1
            self.embedding = ConstEmbedding(cfg, seq_len)

        if cfg.model_type == "transformer":
            decoder_layer = TransformerDecoderLayerGlobalImproved(cfg.d_model, cfg.dim_z, cfg.n_heads, cfg.dim_feedforward, cfg.dropout, d_global2=dim_label)
            decoder_norm = LayerNorm(cfg.d_model)
            self.decoder = TransformerDecoder(decoder_layer, cfg.n_layers_decode, decoder_norm)
        else:  # "lstm"
            self.fc_hc = nn.Linear(cfg.dim_z, 2 * cfg.d_model)
            self.decoder = nn.LSTM(cfg.d_model, cfg.d_model, dropout=cfg.dropout)

        args_dim = 2 * cfg.args_dim if cfg.rel_targets else cfg.args_dim + 1
        self.fcn = FCN(cfg.d_model, cfg.n_commands, cfg.n_args, args_dim)

    def _get_initial_state(self, z):
        hidden, cell = torch.split(torch.tanh(self.fc_hc(z)), self.cfg.d_model, dim=2)
        hidden_cell = hidden.contiguous(), cell.contiguous()
        return hidden_cell

    def forward(self, z, commands, args, label=None, hierarch_logits=None, return_hierarch=False):
        N = z.size(2)
        l = self.label_embedding(label).unsqueeze(0) if self.cfg.label_condition else None
        if hierarch_logits is None:
            z = _pack_group_batch(z)

        if self.cfg.decode_stages == 2:
            if hierarch_logits is None:
                src = self.hierarchical_embedding(z)
                out = self.hierarchical_decoder(src, z, tgt_mask=None, tgt_key_padding_mask=None, memory2=l)
                hierarch_logits, z = self.hierarchical_fcn(out)

            if self.cfg.label_condition: l = l.unsqueeze(0).repeat(1, z.size(1), 1, 1)

            hierarch_logits, z, l = _pack_group_batch(hierarch_logits, z, l)

            if return_hierarch:
                return _unpack_group_batch(N, hierarch_logits, z)

        if self.cfg.pred_mode == "autoregressive":
            S = commands.size(0)
            commands, args = _pack_group_batch(commands, args)

            group_mask = _get_group_mask(commands, seq_dim=0)

            src = self.embedding(commands, args, group_mask)

            if self.cfg.model_type == "transformer":
                key_padding_mask = _get_key_padding_mask(commands, seq_dim=0)
                out = self.decoder(src, z, tgt_mask=self.square_subsequent_mask[:S, :S], tgt_key_padding_mask=key_padding_mask, memory2=l)
            else:  # "lstm"
                hidden_cell = self._get_initial_state(z)  # TODO: reinject intermediate state
                out, _ = self.decoder(src, hidden_cell)

        else:  # "one_shot"
            src = self.embedding(z)
            out = self.decoder(src, z, tgt_mask=None, tgt_key_padding_mask=None, memory2=l)

        command_logits, args_logits = self.fcn(out)

        out_logits = (command_logits, args_logits) + ((hierarch_logits,) if self.cfg.decode_stages == 2 else ())

        return _unpack_group_batch(N, *out_logits)


class SVGTransformer(nn.Module):
    def __init__(self, cfg: _DefaultConfig):
        super(SVGTransformer, self).__init__()

        self.cfg = cfg
        self.args_dim = 2 * cfg.args_dim if cfg.rel_targets else cfg.args_dim + 1

        if self.cfg.encode_stages > 0:

            self.encoder = Encoder(cfg)

            if cfg.use_resnet:
                self.resnet = ResNet(cfg.d_model)

            if cfg.use_vae:
                self.vae = VAE(cfg)
            else:
                self.bottleneck = Bottleneck(cfg)

        self.decoder = Decoder(cfg)

        self.register_buffer("cmd_args_mask", SVGTensor.CMD_ARGS_MASK)

    def perfect_matching(self, command_logits, args_logits, hierarch_logits, tgt_commands, tgt_args):
        with torch.no_grad():
            N, G, S, n_args = tgt_args.shape
            visibility_mask = _get_visibility_mask(tgt_commands, seq_dim=-1)
            padding_mask = _get_padding_mask(tgt_commands, seq_dim=-1, extended=True) * visibility_mask.unsqueeze(-1)

            # Unsqueeze
            tgt_commands, tgt_args, tgt_hierarch = tgt_commands.unsqueeze(2), tgt_args.unsqueeze(2), visibility_mask.unsqueeze(2)
            command_logits, args_logits, hierarch_logits = command_logits.unsqueeze(1), args_logits.unsqueeze(1), hierarch_logits.unsqueeze(1).squeeze(-2)

            # Loss
            tgt_hierarch, hierarch_logits = tgt_hierarch.repeat(1, 1, self.cfg.num_groups_proposal), hierarch_logits.repeat(1, G, 1, 1)
            tgt_commands, command_logits = tgt_commands.repeat(1, 1, self.cfg.num_groups_proposal, 1), command_logits.repeat(1, G, 1, 1, 1)
            tgt_args, args_logits = tgt_args.repeat(1, 1, self.cfg.num_groups_proposal, 1, 1), args_logits.repeat(1, G, 1, 1, 1, 1)

            padding_mask, mask = padding_mask.unsqueeze(2).repeat(1, 1, self.cfg.num_groups_proposal, 1), self.cmd_args_mask[tgt_commands.long()]

            loss_args = F.cross_entropy(args_logits.reshape(-1, self.args_dim), tgt_args.reshape(-1).long() + 1, reduction="none").reshape(N, G, self.cfg.num_groups_proposal, S, n_args)    # shift due to -1 PAD_VAL
            loss_cmd = F.cross_entropy(command_logits.reshape(-1, self.cfg.n_commands), tgt_commands.reshape(-1).long(), reduction="none").reshape(N, G, self.cfg.num_groups_proposal, S)
            loss_hierarch = F.cross_entropy(hierarch_logits.reshape(-1, 2), tgt_hierarch.reshape(-1).long(), reduction="none").reshape(N, G, self.cfg.num_groups_proposal)

            loss_args = (loss_args * mask).sum(dim=[-1, -2]) / mask.sum(dim=[-1, -2])
            loss_cmd = (loss_cmd * padding_mask).sum(dim=-1) / padding_mask.sum(dim=-1)

            loss = 2.0 * loss_args + 1.0 * loss_cmd + 1.0 * loss_hierarch

        # Iterate over the batch-dimension
        assignment_list = []

        full_set = set(range(self.cfg.num_groups_proposal))
        for i in range(N):
            costs = loss[i]
            mask = visibility_mask[i]
            _, assign = linear_sum_assignment(costs[mask].cpu())
            assign = assign.tolist()
            assignment_list.append(assign + list(full_set - set(assign)))

        assignment = torch.tensor(assignment_list, device=command_logits.device)

        return assignment.unsqueeze(-1).unsqueeze(-1)

    def forward(self, commands_enc, args_enc, commands_dec, args_dec, label=None,
                z=None, hierarch_logits=None,
                return_tgt=True, params=None, encode_mode=False, return_hierarch=False):
        commands_enc, args_enc = _make_seq_first(commands_enc, args_enc)  # Possibly None, None
        commands_dec_, args_dec_ = _make_seq_first(commands_dec, args_dec)

        if z is None:
            z = self.encoder(commands_enc, args_enc, label)

            if self.cfg.use_resnet:
                z = self.resnet(z)

            if self.cfg.use_vae:
                z, mu, logsigma = self.vae(z)
            else:
                z = self.bottleneck(z)
        else:
            z = _make_seq_first(z)

        if encode_mode: return z

        if return_tgt:  # Train mode
            commands_dec_, args_dec_ = commands_dec_[:-1], args_dec_[:-1]

        out_logits = self.decoder(z, commands_dec_, args_dec_, label, hierarch_logits=hierarch_logits,
                                  return_hierarch=return_hierarch)

        if return_hierarch:
            return out_logits

        out_logits = _make_batch_first(*out_logits)

        if return_tgt and self.cfg.self_match:  # Assignment
            assert self.cfg.decode_stages == 2  # Self-matching expects two-stage decoder
            command_logits, args_logits, hierarch_logits = out_logits

            assignment = self.perfect_matching(command_logits, args_logits, hierarch_logits, commands_dec[..., 1:], args_dec[..., 1:, :])

            command_logits = torch.gather(command_logits, dim=1, index=assignment.expand_as(command_logits))
            args_logits = torch.gather(args_logits, dim=1, index=assignment.unsqueeze(-1).expand_as(args_logits))
            hierarch_logits = torch.gather(hierarch_logits, dim=1, index=assignment.expand_as(hierarch_logits))

            out_logits = (command_logits, args_logits, hierarch_logits)

        res = {
            "command_logits": out_logits[0],
            "args_logits": out_logits[1]
        }

        if self.cfg.decode_stages == 2:
            res["visibility_logits"] = out_logits[2]

        if return_tgt:
            res["tgt_commands"] = commands_dec
            res["tgt_args"] = args_dec

            if self.cfg.use_vae:
                res["mu"] = _make_batch_first(mu)
                res["logsigma"] = _make_batch_first(logsigma)

        return res

    def greedy_sample(self, commands_enc=None, args_enc=None, commands_dec=None, args_dec=None, label=None,
                      z=None, hierarch_logits=None,
                      concat_groups=True, temperature=0.0001):
        if self.cfg.pred_mode == "one_shot":
            res = self.forward(commands_enc, args_enc, commands_dec, args_dec, label=label, z=z, hierarch_logits=hierarch_logits, return_tgt=False)
            commands_y, args_y = _sample_categorical(temperature, res["command_logits"], res["args_logits"])
            args_y -= 1  # shift due to -1 PAD_VAL
            visibility_y = _threshold_sample(res["visibility_logits"], threshold=0.7).bool().squeeze(-1) if self.cfg.decode_stages == 2 else None
            commands_y, args_y = self._make_valid(commands_y, args_y, visibility_y)
        else:
            if z is None:
                z = self.forward(commands_enc, args_enc, None, None, label=label, encode_mode=True)

            PAD_VAL = -1
            commands_y, args_y = z.new_zeros(1, 1, 1).fill_(SVGTensor.COMMANDS_SIMPLIFIED.index("SOS")).long(), z.new_ones(1, 1, 1, self.cfg.n_args).fill_(PAD_VAL).long()

            for i in range(self.cfg.max_total_len):
                res = self.forward(None, None, commands_y, args_y, label=label, z=z, hierarch_logits=hierarch_logits, return_tgt=False)
                commands_new_y, args_new_y = _sample_categorical(temperature, res["command_logits"], res["args_logits"])
                args_new_y -= 1  # shift due to -1 PAD_VAL
                _, args_new_y = self._make_valid(commands_new_y, args_new_y)

                commands_y, args_y = torch.cat([commands_y, commands_new_y[..., -1:]], dim=-1), torch.cat([args_y, args_new_y[..., -1:, :]], dim=-2)

            commands_y, args_y = commands_y[..., 1:], args_y[..., 1:, :]  # Discard SOS token

        if self.cfg.rel_targets:
            args_y = self._make_absolute(commands_y, args_y)

        if concat_groups:
            N = commands_y.size(0)
            padding_mask_y = _get_padding_mask(commands_y, seq_dim=-1).bool()
            commands_y, args_y = commands_y[padding_mask_y].reshape(N, -1), args_y[padding_mask_y].reshape(N, -1, self.cfg.n_args)

        return commands_y, args_y

    def _make_valid(self, commands_y, args_y, visibility_y=None, PAD_VAL=-1):
        if visibility_y is not None:
            S = commands_y.size(-1)
            commands_y[~visibility_y] = commands_y.new_tensor([SVGTensor.COMMANDS_SIMPLIFIED.index("m"), *[SVGTensor.COMMANDS_SIMPLIFIED.index("EOS")] * (S - 1)])
            args_y[~visibility_y] = PAD_VAL

        mask = self.cmd_args_mask[commands_y.long()].bool()
        args_y[~mask] = PAD_VAL

        return commands_y, args_y

    def _make_absolute(self, commands_y, args_y):

        mask = self.cmd_args_mask[commands_y.long()].bool()
        args_y[mask] -= self.cfg.args_dim - 1

        real_commands = commands_y < SVGTensor.COMMANDS_SIMPLIFIED.index("EOS")

        args_real_commands = args_y[real_commands]
        end_pos = args_real_commands[:-1, SVGTensor.IndexArgs.END_POS].cumsum(dim=0)

        args_real_commands[1:, SVGTensor.IndexArgs.CONTROL1] += end_pos
        args_real_commands[1:, SVGTensor.IndexArgs.CONTROL2] += end_pos
        args_real_commands[1:, SVGTensor.IndexArgs.END_POS] += end_pos

        args_y[real_commands] = args_real_commands

        _, args_y = self._make_valid(commands_y, args_y)

        return args_y
