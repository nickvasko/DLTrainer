from torch import nn


class SimpleConfig:
    def __init__(self, args):
        self.input_size = args.input_size if args.input_size else 2
        self.hidden_size = args.hidden_size if args.hidden_size else 10
        self.output_size = args.output_size if args.output_size else 1


class SimpleModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_size = config.input_size
        self.hidden_size = config.hidden_size
        self.output_size = config.output_size

        self.hidden_layer = nn.Linear(self.input_size, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)
        self.activation_fn = nn.ReLU()

        self.criterion = nn.BCELoss()

    def forward(self, x, y=None):
        hidden_state = self.hidden_layer(x)
        hidden_state = self.activation_fn(hidden_state)
        output = self.output_layer(hidden_state)

        loss = None
        if y is not None:
            loss = self.criterion(output, y)
        return dict(loss=loss,
                    output=output)
