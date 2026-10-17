"""PD multiplexing (--enable-pdmux) under EAGLE3 MTP: split-prefill support for the speculative path.

Applied to a copy of up_edf_floor (tree up_pdmux_mtp). Under pdmux a prefill runs as a layer-split batch on the
prefill SM group (scheduler.run_batch -> tp_worker.forward_batch_split_prefill), which ran only the target model:
no EAGLE3 aux hidden states, no draft extend, so prefill requests merged into the running batch without draft
state and the first draft step asserted (topk1.py). This makes the split path do what EAGLEWorkerV2's normal
prefill does: target with CaptureHiddenMode.FULL (EAGLE3 aux layers captured across splits), then
draft_extend_for_prefill after the last split.

  python apply_pdmux_mtp.py <tree>/python/sglang/srt
"""
import sys
srt = sys.argv[1]


def patch(rel, old, new, count=1):
    p = f"{srt}/{rel}"
    s = open(p).read()
    if new in s:
        print("already", rel); return
    assert s.count(old) == count, (rel, s.count(old), old[:80])
    open(p, "w").write(s.replace(old, new))
    print("patched", rel)


# 1. target split prefill takes a hidden-state capture mode (fixed at the first split, when the ForwardBatch is built)
patch("managers/tp_worker.py",
'''    def forward_batch_split_prefill(self, batch: ScheduleBatch):
        if batch.split_index == 0:
            forward_batch = ForwardBatch.init_new(
                batch,
                self.model_runner,
                return_hidden_states_before_norm=False,
            )''',
'''    def forward_batch_split_prefill(
        self,
        batch: ScheduleBatch,
        capture_hidden_mode: Optional[CaptureHiddenMode] = None,
    ):
        if batch.split_index == 0:
            forward_batch = ForwardBatch.init_new(
                batch,
                self.model_runner,
                capture_hidden_mode=capture_hidden_mode,
                return_hidden_states_before_norm=False,
            )''')

# 2. llama: capture EAGLE3 aux hidden states across splits, as the full forward does
patch("models/llama.py",
'''        start, end = split_interval
        # embed
        if start == 0:
            residual_batch.start(forward_batch)
            if input_embeds is None:
                forward_batch.hidden_states = self.model.embed_tokens(input_ids)
            else:
                forward_batch.hidden_states = input_embeds
        # decoder layer
        for i in range(start, end):
            layer = self.model.layers[i]
            forward_batch.hidden_states = layer(
                positions,
                forward_batch.hidden_states,
                forward_batch,
            )''',
'''        start, end = split_interval
        # embed
        if start == 0:
            residual_batch.start(forward_batch)
            if input_embeds is None:
                forward_batch.hidden_states = self.model.embed_tokens(input_ids)
            else:
                forward_batch.hidden_states = input_embeds
            # EAGLE3 reads the hidden states of model.layers_to_capture; a split prefill reaches them across
            # several calls, so the list lives on the ForwardBatch until the last split.
            forward_batch.split_aux_hidden_states = (
                AuxHiddenStateList() if self.capture_aux_hidden_states else None
            )
        aux = getattr(forward_batch, "split_aux_hidden_states", None)
        # decoder layer
        for i in range(start, end):
            layer = self.model.layers[i]
            forward_batch.hidden_states = layer(
                positions,
                forward_batch.hidden_states,
                forward_batch,
                capture_output=aux.capture
                if aux is not None and i in getattr(self.model, "layers_to_capture", ())
                else None,
            )''')
patch("models/llama.py",
'''            # logits process
            result = self.logits_processor(
                input_ids, forward_batch.hidden_states, self.lm_head, forward_batch
            )
        else:
            result = None''',
'''            # logits process
            result = self.logits_processor(
                input_ids,
                forward_batch.hidden_states,
                self.lm_head,
                forward_batch,
                aux if aux else None,
            )
            forward_batch.split_aux_hidden_states = None
        else:
            result = None''')

# 3. EAGLEWorkerV2: split prefill = target split with FULL capture, then draft extend after the last split
patch("speculative/eagle_worker_v2.py",
'''    def _forward_dp_spec_prefill_coordination(''',
'''    def forward_batch_split_prefill(self, batch):
        """One layer-split slice of a prefill under PD multiplexing. The target runs this slice on the prefill SM
        group; after the last slice the draft extend runs as in _forward_prefill_batch, so the requests carry
        draft state when they join the running (decode) batch."""
        target_capture_mode = (
            CaptureHiddenMode.NULL
            if self.speculative_algorithm.is_standalone()
            else CaptureHiddenMode.FULL
        )
        batch_output = self.target_worker.forward_batch_split_prefill(
            batch, capture_hidden_mode=target_capture_mode
        )
        if batch_output.logits_output is None:
            return batch_output
        batch_output.new_seq_lens = batch.seq_lens
        if self._draft_worker is None:
            return batch_output
        # The draft extend covers the whole prompt in one normal extend; left at SPLIT_PREFILL it would be routed
        # through the draft model's forward_split_prefill.
        split_mode, batch.forward_mode = batch.forward_mode, ForwardMode.EXTEND
        try:
            with (
                draft_tp_context(self.draft_worker.draft_owns_attention),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
                spec_stage_span("draft_extend"),
            ):
                batch_output.next_draft_input = self.draft_worker._draft_extend_for_prefill(
                    batch,
                    batch_output.logits_output.hidden_states,
                    batch_output.next_token_ids,
                    batch_output.logits_output.mm_input_embeds,
                )
        finally:
            batch.forward_mode = split_mode
        return batch_output

    def _forward_dp_spec_prefill_coordination(''')

# 4. scheduler: route split prefills through the speculative worker when speculation is on
patch("managers/scheduler.py",
'''            elif self.enable_pdmux and batch.forward_mode.is_split_prefill():
                resolve_forward_inputs(batch, self.future_map)
                batch_result = self.tp_worker.forward_batch_split_prefill(batch)''',
'''            elif self.enable_pdmux and batch.forward_mode.is_split_prefill():
                resolve_forward_inputs(batch, self.future_map)
                if batch.spec_algorithm.is_none():
                    batch_result = self.tp_worker.forward_batch_split_prefill(batch)
                else:
                    batch_result = self.model_worker.forward_batch_split_prefill(batch)''')
print("done")
