import chainer

from paint_transfer.config import ModelConfig
from paint_transfer.model import Models


class Forwarder(chainer.ChainList):
    def __init__(
            self,
            config: ModelConfig,
            models: Models,
    ):
        super().__init__(config)
        self.config = config
        self.models = models

    def __call__(self, *args, **kwargs):
        return self.forward_generator(*args, **kwargs, test=True)['generated']

    def forward(self, input, raw_line, raw_line_mismatch, test):
        output = self.forward_generator(input=input, raw_line=raw_line, test=test)
        z, generated = output['z'], output['generated']
        match, _ = self.models.mismatch_discriminator({'z': z, 'raw_line': raw_line}, test)
        mismatch, _ = self.models.mismatch_discriminator({'z': z, 'raw_line': raw_line_mismatch}, test)
        return {
            'z': z,
            'generated': generated,
            'match': match,
            'mismatch': mismatch,
        }

    def forward_generator(self, input, raw_line, test):
        z, _ = self.models.encoder(input, test)
        generated, _ = self.models.generator({'z': z, 'raw_line': raw_line}, test)
        return {
            'z': z,
            'generated': generated,
        }
