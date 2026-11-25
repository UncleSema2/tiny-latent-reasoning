"""Fixed-step decoding for the released LVR-7B checkpoint."""

import torch
from sglang.srt.model_executor.forward_context import get_req_to_token_pool
from sglang.srt.models.qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration as QwenVL,
)


class Qwen2_5_VLForConditionalGeneration(QwenVL):
    """Keep SGLang's Qwen vision processor and replace only the decoding step."""

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch, **kwargs):
        slots = forward_batch.req_pool_indices
        if not hasattr(self, "remaining"):
            if self.config.lvr_head or self.config.latent_end_token:
                raise ValueError(
                    "This decoder requires the released headless LVR model"
                )
            size = get_req_to_token_pool().size + 1
            self.remaining = input_ids.new_zeros(size)
            self.previous = self.model.embed_tokens.weight.new_zeros(
                size, self.config.hidden_size
            )

        if not forward_batch.forward_mode.is_decode():
            self.remaining[slots] = 0
            return super().forward(input_ids, positions, forward_batch, **kwargs)

        remaining = self.remaining[slots]
        active = remaining > 0
        embeds = torch.where(
            active[:, None], self.previous[slots], self.model.embed_tokens(input_ids)
        )
        hidden = self.model(
            input_ids, forward_batch.mrope_positions, forward_batch, input_embeds=embeds
        )
        self.previous[slots] = hidden
        remaining = torch.where(
            active,
            remaining - 1,
            (input_ids == self.config.lvr_start_id) * self.config.lvr_steps,
        )
        self.remaining[slots] = remaining
        output = self.logits_processor(input_ids, hidden, self.lm_head, forward_batch)
        # The reference ignores stop tokens until the latent budget is exhausted.
        # During those steps the next embedding comes entirely from hidden.
        scores = output.next_token_logits
        for stop_id in (self.config.eos_token_id, self.config.bos_token_id):
            scores[:, stop_id] = scores[:, stop_id].masked_fill(
                remaining > 0, -torch.inf
            )
        return output


EntryClass = Qwen2_5_VLForConditionalGeneration
