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
LLAMA_SPLIT_PATCHES = [
    ('''        start, end = split_interval
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
            )'''),
    ('''            # logits process
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
            result = None'''),
]
for old, new in LLAMA_SPLIT_PATCHES:
    patch("models/llama.py", old, new)

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
        # The scheduler drops batch.input_ids after every split slice (the target's ForwardBatch, built at the first
        # slice, keeps them); the draft extend rotates the prompt ids, so take them back from there.
        if batch.input_ids is None:
            batch.input_ids = batch.split_forward_batch.input_ids
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
                    batch_result = self.model_worker.forward_batch_split_prefill(batch)
                    # After the last split the draft extend produced the requests' first draft state; install it as
                    # the speculative branch below does, so they join the decode batch with an EagleDraftInput.
                    if getattr(batch_result, "next_draft_input", None) is not None:
                        batch.spec_info = batch_result.next_draft_input''')
# 5. eager decode: the per-stream pdmux decode backend applies to the target runner only. The scheduler switches the
#    target runner's backend per stream group (multiplexing_mixin.adjust_stream_groups); the draft runner is not
#    switched and its draft decode runs under the multi-step draft backends that draft() planned, so overriding its
#    context with the (never planned) pdmux decode backend left forward_metadata None.
patch("model_executor/runner/eager_runner.py",
'''        model_runner = self.model_runner
        if self.enable_pdmux:
            return model_runner.decode_attn_backend, forward_context(''',
'''        model_runner = self.model_runner
        if self.enable_pdmux and not model_runner.is_draft_worker:
            return model_runner.decode_attn_backend, forward_context(''')
# 6. qwen2 (the Qwen2.5 target): the same EAGLE3 aux capture across splits as llama (patch 2); its forward_split_prefill
#    is textually the same and its decoder layer takes capture_output.
for old, new in LLAMA_SPLIT_PATCHES:
    patch("models/qwen2.py", old, new)

# 7. target verify under pdmux runs on the per-stream decode backend, like decode. Verify is an extend-mode forward and
#    planned the target's attn_backend -- the backend the split prefill planned at its first slice -- so a verify
#    between two prefill slices re-planned it ("q.shape[0] (19885) does not match qo_indptr[-1] (12)"). The group
#    backends are full backends of the same type (attention_backend_setup), so they plan target verify.
patch("model_executor/runner/eager_runner.py",
'''        model_runner = self.model_runner
        kwargs = model_runner._extend_forward_kwargs(forward_batch, pp_proxy_tensors)

        if not self.enable_pdmux:
            forward_batch = self.load_batch(forward_batch, pp_proxy_tensors)''',
'''        model_runner = self.model_runner
        kwargs = model_runner._extend_forward_kwargs(forward_batch, pp_proxy_tensors)
        extend_backend, extend_ctx = model_runner.attn_backend, contextlib.nullcontext()
        if (
            self.enable_pdmux
            and not model_runner.is_draft_worker
            and forward_batch.forward_mode.is_target_verify()
        ):
            extend_backend = model_runner.decode_attn_backend
            extend_ctx = forward_context(ForwardContext(attn_backend=extend_backend))

        if not self.enable_pdmux:
            forward_batch = self.load_batch(forward_batch, pp_proxy_tensors)''')
patch("model_executor/runner/eager_runner.py",
'''            model_runner.attn_backend.init_forward_metadata(forward_batch)
            model_runner.attn_backend.prepare_prefill_shared_read_snapshot(''',
'''            extend_backend.init_forward_metadata(forward_batch)
            extend_backend.prepare_prefill_shared_read_snapshot(''')
patch("model_executor/runner/eager_runner.py",
'''        with device_timer_ctx(model_runner.device_timer, category):
            pcg_runner = model_runner.prefill_cuda_graph_runner''',
'''        with device_timer_ctx(model_runner.device_timer, category), extend_ctx:
            pcg_runner = model_runner.prefill_cuda_graph_runner''')
print("done")
