"""Serve fixed-budget LUT checkpoints with SGLang's Qwen vision processor."""

import torch
from sglang.srt.model_executor.forward_context import get_req_to_token_pool
from sglang.srt.models.qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration as QwenVL,
)


def force_token(scores, token, force):
    """Constrain selected rows while preserving answer logits in the other rows."""
    selected = scores.gather(1, token[:, None])
    scores.masked_fill_(force[:, None], -torch.inf)
    scores.scatter_(1, token[:, None], selected.masked_fill(force[:, None], 0))


class Qwen2_5_VLForConditionalGeneration(QwenVL):
    def load_weights(self, weights):
        # The training checkpoint uses the Transformers 5 parameter names.
        return super().load_weights(
            (
                name.replace("model.language_model.", "model.", 1).replace(
                    "model.visual.", "visual.", 1
                ),
                tensor,
            )
            for name, tensor in weights
        )

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch, **kwargs):
        start = self.config.lut_start_id
        end = self.config.lut_end_id
        steps = self.config.lut_steps
        slots = forward_batch.req_pool_indices
        if not hasattr(self, "remaining"):
            size = get_req_to_token_pool().size + 1
            self.remaining = input_ids.new_zeros(size)
            self.previous = self.model.embed_tokens.weight.new_zeros(
                size, self.config.hidden_size
            )

        if not forward_batch.forward_mode.is_decode():
            # Evicted continuous inputs cannot be reconstructed from the token IDs.
            if ((input_ids == start) | (input_ids == end)).any():
                raise ValueError(
                    "LUT cannot replay generated latents during prefill. "
                    "Retry with a smaller batch size."
                )
            self.remaining[slots] = 0
            output = super().forward(input_ids, positions, forward_batch, **kwargs)
            output.next_token_logits.fill_(-torch.inf)
            output.next_token_logits[:, start] = 0
            return output

        remaining = self.remaining[slots]
        active = remaining > 0
        opening = (input_ids == start) & ~active
        embeds = torch.where(
            active[:, None], self.previous[slots], self.model.embed_tokens(input_ids)
        )
        hidden = self.model(
            input_ids, forward_batch.mrope_positions, forward_batch, input_embeds=embeds
        )
        self.previous[slots] = hidden
        remaining = torch.where(active, remaining - 1, opening * steps)
        self.remaining[slots] = remaining
        output = self.logits_processor(input_ids, hidden, self.lm_head, forward_batch)
        scores = output.next_token_logits
        # The opening ID also serves as the placeholder during the recurrent span.
        token = torch.where(remaining > 0, start, end)
        force_token(scores, token, active | opening)
        scores[:, start].masked_fill_(~(active | opening), -torch.inf)
        scores[:, end].masked_fill_(~(active | opening), -torch.inf)
        return output


EntryClass = Qwen2_5_VLForConditionalGeneration
