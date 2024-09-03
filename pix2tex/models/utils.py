import torch
import torch.nn as nn
import torch.nn.functional as F

from . import hybrid
from . import vit
from . import transformer


class Model(nn.Module):
    def _init_(self, encoder, decoder, args):
        super()._init_()
        self.encoder = encoder
        self.decoder = decoder
        self.args = args

    def data_parallel(self, x: torch.Tensor, device_ids, output_device=None, **kwargs):
        if not device_ids or len(device_ids) == 1:
            return self(x, **kwargs)
        if output_device is None:
            output_device = device_ids[0]
        replicas = nn.parallel.replicate(self, device_ids)
        inputs = nn.parallel.scatter(x, device_ids)  # Slices tensors into approximately equal chunks and distributes them across given GPUs.
        kwargs = nn.parallel.scatter(kwargs, device_ids)  # Duplicates references to objects that are not tensors.
        replicas = replicas[:len(inputs)]
        kwargs = kwargs[:len(inputs)]
        outputs = nn.parallel.parallel_apply(replicas, inputs, kwargs)
        return nn.parallel.gather(outputs, output_device).mean()

    def forward(self, x: torch.Tensor, tgt_seq: torch.Tensor,  **kwargs):
        encoded = self.encoder(x)
        out = self.decoder(tgt_seq, context=encoded, **kwargs)
        return out

    def beam_search(self, x: torch.Tensor, beam_width: int = 5, max_seq_len: int = 20, temperature: float = 0.25):
        # Initialize beam search parameters
        eos_token = self.args.eos_token
        bos_token = self.args.bos_token
        device = x.device

        # Encode the input sequence
        encoded = self.encoder(x)
        
        # Initialize beams
        beams = [(torch.LongTensor([bos_token]).to(device), 0)]  # (sequence, score)

        for _ in range(max_seq_len):
            new_beams = []
            for seq, score in beams:
                # Generate the next token probabilities
                outputs = self.decoder.generate(seq.unsqueeze(0), 1, eos_token=eos_token, context=encoded, temperature=temperature)
                logits = outputs[0, -1]  # Get the logits for the next token

                # Calculate the probabilities and scores
                probs = F.log_softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, beam_width, dim=-1)

                for i in range(beam_width):
                    next_token = topk_indices[0, i]
                    next_score = score + topk_probs[0, i].item()
                    new_seq = torch.cat([seq, next_token.unsqueeze(0)])
                    new_beams.append((new_seq, next_score))

            # Keep only the top beam_width beams
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

            # Check if all beams end with eos_token
            if all(seq[-1].item() == eos_token for seq, _ in beams):
                break

        # Return the sequence with the highest score
        best_seq = max(beams, key=lambda x: x[1])[0]
        return best_seq

    @torch.no_grad()
    def generate(self, x: torch.Tensor, beam_width: int = 5, temperature: float = 0.25):
        return self.beam_search(x, beam_width=beam_width, temperature=temperature)


def get_model(args):
    if args.encoder_structure.lower() == 'vit':
        encoder = vit.get_encoder(args)
    elif args.encoder_structure.lower() == 'hybrid':
        encoder = hybrid.get_encoder(args)
    else:
        raise NotImplementedError('Encoder structure "%s" not supported.' % args.encoder_structure)
    decoder = transformer.get_decoder(args)
    encoder.to(args.device)
    decoder.to(args.device)
    model = Model(encoder, decoder, args)
    if args.wandb:
        import wandb
        wandb.watch(model)

    return model
