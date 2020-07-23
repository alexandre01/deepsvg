from deepsvg.difflib.tensor import SVGTensor


class _DefaultConfig:
    """
    Model config.
    """
    def __init__(self):
        self.args_dim = 256              # Coordinate numericalization, default: 256 (8-bit)
        self.n_args = 11                 # Tensor nb of arguments, default: 11 (rx,ry,phi,fA,fS,qx1,qy1,qx2,qy2,x1,x2)
        self.n_commands = len(SVGTensor.COMMANDS_SIMPLIFIED)  # m, l, c, a, EOS, SOS, z

        self.dropout = 0.1                # Dropout rate used in basic layers and Transformers

        self.model_type = "transformer"  # "transformer" ("lstm" implementation is work in progress)

        self.encode_stages = 1           # One-stage or two-stage: 1 | 2
        self.decode_stages = 1           # One-stage or two-stage: 1 | 2

        self.use_resnet = True           # Use extra fully-connected residual blocks after Encoder

        self.use_vae = True              # Sample latent vector (with reparametrization trick) or use encodings directly

        self.pred_mode = "one_shot"      # Feed-forward (one-shot) or autogressive: "one_shot" | "autoregressive"
        self.rel_targets = False         # Predict coordinates in relative or absolute format

        self.label_condition = False     # Make all blocks conditional on the label
        self.n_labels = 100              # Number of labels (when used)
        self.dim_label = 64              # Label embedding dimensionality

        self.self_match = False          # Use Hungarian (self-match) or Ordered assignment

        self.n_layers = 4                # Number of Encoder blocks
        self.n_layers_decode = 4         # Number of Decoder blocks
        self.n_heads = 8                 # Transformer config: number of heads
        self.dim_feedforward = 512       # Transformer config: FF dimensionality
        self.d_model = 256               # Transformer config: model dimensionality

        self.dim_z = 256                 # Latent vector dimensionality

        self.max_num_groups = 8          # Number of paths (N_P)
        self.max_seq_len = 30            # Number of commands (N_C)
        self.max_total_len = self.max_num_groups * self.max_seq_len  # Concatenated sequence length for baselines

        self.num_groups_proposal = self.max_num_groups  # Number of predicted paths, default: N_P

    def get_model_args(self):
        model_args = []

        model_args += ["commands_grouped", "args_grouped"] if self.encode_stages <= 1 else ["commands", "args"]

        if self.rel_targets:
            model_args += ["commands_grouped", "args_rel_grouped"] if self.decode_stages == 1 else ["commands", "args_rel"]
        else:
            model_args += ["commands_grouped", "args_grouped"] if self.decode_stages == 1 else ["commands", "args"]

        if self.label_condition:
            model_args.append("label")

        return model_args


class SketchRNN(_DefaultConfig):
    # LSTM - Autoregressive - One-stage
    def __init__(self):
        super().__init__()

        self.model_type = "lstm"

        self.pred_mode = "autoregressive"
        self.rel_targets = True


class Sketchformer(_DefaultConfig):
    # Transformer - Autoregressive - One-stage
    def __init__(self):
        super().__init__()

        self.pred_mode = "autoregressive"
        self.rel_targets = True


class OneStageOneShot(_DefaultConfig):
    # Transformer - One-shot - One-stage
    def __init__(self):
        super().__init__()

        self.encode_stages = 1
        self.decode_stages = 1


class Hierarchical(_DefaultConfig):
    # Transformer - One-shot - Two-stage - Ordered
    def __init__(self):
        super().__init__()

        self.encode_stages = 2
        self.decode_stages = 2


class HierarchicalSelfMatching(_DefaultConfig):
    # Transformer - One-shot - Two-stage - Hungarian
    def __init__(self):
        super().__init__()

        self.encode_stages = 2
        self.decode_stages = 2
        self.self_match = True
