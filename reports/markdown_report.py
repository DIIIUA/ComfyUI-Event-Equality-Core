from ..core.packet import ensure_packet, packet_summary
from ..core.signal import signal_public_summary
from ..adapters.wan.wan_report import build_wan_report_block


def _format_signature_value(value):
    if isinstance(value, float):
        return f"{value:.6g}"
    return value


def _compact_line_from_dict(data):
    if not isinstance(data, dict):
        return str(data)
    parts = []
    for key, value in data.items():
        if value is None:
            continue
        parts.append(f"{key}={_format_signature_value(value)}")
    return ", ".join(parts)


def _format_semantic_summary(summary: dict) -> str:
    if not isinstance(summary, dict):
        return ""
    keys = ["char_count", "word_count", "line_count", "has_text"]
    parts = []
    for key in keys:
        if key in summary:
            parts.append(f"{key}={summary[key]}")
    return ", ".join(parts)


def _format_numeric_summary(summary: dict) -> str:
    if not isinstance(summary, dict):
        return ""
    keys = ["latent_format", "image_format", "noise_format", "delta_format", "python_type", "shape", "dtype", "device", "mean", "std", "norm", "delta_norm", "relative_delta"]
    parts = []
    for key in keys:
        if key in summary and summary[key] is not None:
            value = summary[key]
            if isinstance(value, float):
                value = f"{value:.6g}"
            parts.append(f"{key}={value}")
    return ", ".join(parts)


def build_markdown_report(packet=None) -> str:
    """Build a safe markdown report. Never prints raw_ref."""
    packet = ensure_packet(packet)
    summary = packet_summary(packet)

    lines = []
    lines.append("# Event Equality Report")

    event_core_body = packet.get("metadata", {}).get("event_core_body", {})
    if event_core_body:
        summary_top = event_core_body.get("summary", {}) if isinstance(event_core_body.get("summary", {}), dict) else {}
        gate = event_core_body.get("completion_gate", {}) if isinstance(event_core_body.get("completion_gate", {}), dict) else {}
        runtime_summary = event_core_body.get("runtime_monitor_summary", {}) if isinstance(event_core_body.get("runtime_monitor_summary", {}), dict) else {}
        lines.append("")
        lines.append("## Event Core Body Top Summary")
        lines.append(f"- runtime_body_version: {event_core_body.get('body_version')}")
        lines.append(f"- audit_gate: {summary_top.get('audit_gate', gate.get('status', 'UNKNOWN'))}")
        lines.append(f"- one_node_ok: {summary_top.get('one_node_ok')}")
        lines.append(f"- stage_order_ok: {summary_top.get('stage_order_ok')}")
        lines.append(f"- missing_total: {summary_top.get('missing_total')}")
        lines.append(f"- stage_math_count: {summary_top.get('stage_math_count')}")
        lines.append(f"- boundary_math_count: {summary_top.get('boundary_math_count')}")
        lines.append(f"- live_route_count: {summary_top.get('live_route_count')}")
        lines.append(f"- runtime_monitor_count: {summary_top.get('runtime_monitor_count')}")
        lines.append(f"- local_sstate_count: {summary_top.get('local_sstate_count')}")
        lines.append(f"- event_conflict_count: {summary_top.get('event_conflict_count')}")
        if runtime_summary:
            lines.append(f"- runtime_observer_span_seconds: {runtime_summary.get('observed_stage_span_seconds')}")
            lines.append(f"- runtime_observer_only: {runtime_summary.get('observer_only')}")

        local_sstates = event_core_body.get("local_sstates", [])
        if local_sstates:
            lines.append("")
            lines.append("## Local SState Formula Role Breakdown")
            for item in local_sstates:
                lines.append(
                    f"- {item.get('name')}: role={item.get('formula_role')} "
                    f"present={item.get('stage_present')} status={item.get('status')} "
                    f"granularity={item.get('granularity')}"
                )

        live_timeline = event_core_body.get("live_route_timeline", [])
        if live_timeline:
            lines.append("")
            lines.append("## Live Route Timeline")
            for item in live_timeline[:60]:
                memory = item.get("memory", {}) if isinstance(item.get("memory", {}), dict) else {}
                mem_bits = []
                if "process_rss_mb" in memory:
                    mem_bits.append(f"rss_mb={memory.get('process_rss_mb')}")
                if memory.get("torch_cuda_available"):
                    mem_bits.append(f"cuda_alloc_mb={memory.get('cuda_allocated_mb')} cuda_reserved_mb={memory.get('cuda_reserved_mb')}")
                mem_text = f" {' '.join(mem_bits)}" if mem_bits else ""
                lines.append(
                    f"- [{item.get('index')}] {item.get('stage')} type={item.get('record_type')} "
                    f"status={item.get('status')} route={item.get('route_id')}{mem_text}"
                )
            if len(live_timeline) > 60:
                lines.append(f"- ... {len(live_timeline) - 60} more live route records")

        sidecars = packet.get("metadata", {}).get("runtime_monitor_sidecars")
        if isinstance(sidecars, dict) and sidecars:
            lines.append("")
            lines.append("## Runtime Monitor Sidecars")
            lines.append(f"- status: {sidecars.get('status')}")
            lines.append(f"- json_path: {sidecars.get('json_path')}")
            lines.append(f"- csv_path: {sidecars.get('csv_path')}")
            lines.append(f"- diff_path: {sidecars.get('diff_path')}")
            lines.append(f"- previous_json_path: {sidecars.get('previous_json_path')}")
            lines.append(f"- settings_signature: {sidecars.get('settings_signature')}")
            lines.append(f"- observer_only: {sidecars.get('observer_only')}")

        barrier_summary = packet.get("metadata", {}).get("branch_barrier_summary")
        if isinstance(barrier_summary, dict) and barrier_summary:
            lines.append("")
            lines.append("## Smart Branch Barrier Summary")
            lines.append(f"- record_count: {barrier_summary.get('record_count')}")
            lines.append(f"- strategy_state_checks: {barrier_summary.get('strategy_state_checks')}")
            lines.append(f"- strategy_state_preserved_count: {barrier_summary.get('strategy_state_preserved_count')}")
            lines.append(f"- observer_only: {barrier_summary.get('observer_only')}")
            phase_counts = barrier_summary.get("phase_counts", {})
            if isinstance(phase_counts, dict) and phase_counts:
                for phase, count in phase_counts.items():
                    lines.append(f"- phase_{phase}: {count}")

        barrier_records = packet.get("metadata", {}).get("branch_barrier_records")
        if isinstance(barrier_records, list) and barrier_records:
            lines.append("")
            lines.append("## Smart Branch Barrier Records")
            for item in barrier_records[:24]:
                phase = item.get("barrier_phase")
                preserved = item.get("strategy_state_preserved")
                released = item.get("released", {}).get("actions", [])
                released_text = ", ".join([str(x) for x in released]) if released else "none"
                rss_delta = item.get("memory_delta", {}).get("process_rss_mb", "")
                lines.append(
                    f"- phase={phase} preserved={preserved} released={released_text} rss_delta_mb={rss_delta}"
                )
            if len(barrier_records) > 24:
                lines.append(f"- ... {len(barrier_records) - 24} more barrier records")

    lines.append("")
    lines.append("## Packet Summary")
    for key, value in summary.items():
        lines.append(f"- {key}: {value}")

    frozen_meta = packet.get("metadata", {}).get("frozen", {})
    if frozen_meta:
        lines.append("")
        lines.append("## Frozen Node")
        lines.append(f"- mode: {frozen_meta.get('mode')}")
        lines.append(f"- report_detail: {frozen_meta.get('report_detail')}")
        lines.append(f"- run_id: {frozen_meta.get('run_id')}")
        lines.append(f"- semantic_mode: {frozen_meta.get('semantic_mode')}")

        event_program_status = frozen_meta.get("event_program_status")
        if event_program_status:
            lines.append("")
            lines.append("## Event Program Status")
            for key, value in event_program_status.items():
                lines.append(f"- {key}: {value}")

        observability = frozen_meta.get("observability")
        if observability:
            lines.append("")
            lines.append("## Observability")
            for key, value in observability.items():
                lines.append(f"- {key}: {value}")

        signatures = frozen_meta.get("input_signatures")
        if signatures:
            lines.append("")
            lines.append("## Input Signatures")
            for key, value in signatures.items():
                lines.append(f"- {key}: {_compact_line_from_dict(value)}")

        passthrough = frozen_meta.get("passthrough_status")
        if passthrough:
            lines.append("")
            lines.append("## Passthrough Status")
            for key, value in passthrough.items():
                lines.append(f"- {key}: {_compact_line_from_dict(value)}")

        mechanics = frozen_meta.get("semantic_relation_mechanics")
        if mechanics:
            lines.append("")
            lines.append("## Semantic Relation Mechanics")
            lines.append(f"- reading_mode: {mechanics.get('reading_mode')}")
            lines.append(f"- sstate_meaning: {mechanics.get('sstate_meaning')}")
            lines.append(f"- verdict_state: {mechanics.get('verdict_state')}")
            shared_targets = mechanics.get("shared_targets", {})
            if shared_targets:
                lines.append("- shared_targets:")
                for target_id, claims in shared_targets.items():
                    lines.append(f"  - {target_id}:")
                    for claim in claims:
                        lines.append(
                            f"    - {claim.get('relation_type')} via {claim.get('relation_id')} "
                            f"local={claim.get('local_strategy_id')}"
                        )

    root_event_program_status = packet.get("metadata", {}).get("event_program_status")
    if root_event_program_status:
        lines.append("")
        lines.append("## Event Program Status")
        for key, value in root_event_program_status.items():
            lines.append(f"- {key}: {value}")

    wan_interface = packet.get("metadata", {}).get("wan_workflow_interface")
    if wan_interface:
        lines.append("")
        lines.append("## Wan Workflow Interface")
        for key, value in wan_interface.items():
            if isinstance(value, dict):
                lines.append(f"- {key}:")
                for sub_key, sub_value in value.items():
                    lines.append(f"  - {sub_key}: {sub_value}")
            else:
                lines.append(f"- {key}: {value}")

    wan_topology = packet.get("metadata", {}).get("wan_event_internal_topology")
    if wan_topology:
        lines.append("")
        lines.append("## Wan Internal Planned Topology")
        for key, value in wan_topology.items():
            lines.append(f"- {key}: {value}")

    cleanup_records = packet.get("metadata", {}).get("cleanup_records")
    if cleanup_records:
        lines.append("")
        lines.append("## Memory Cleanup Records")
        for record in cleanup_records:
            lines.append(f"- {record}")

    execution_records = packet.get("metadata", {}).get("execution_records")
    if execution_records:
        lines.append("")
        lines.append("## Execution Records")
        for record in execution_records:
            lines.append(f"- {record}")

    output_policy = packet.get("metadata", {}).get("program_output_policy")
    if output_policy:
        lines.append("")
        lines.append("## Program Output Policy")
        for key, value in output_policy.items():
            lines.append(f"- {key}: {value}")


    result_status = packet.get("metadata", {}).get("result_status")
    if result_status:
        lines.append("")
        lines.append("## Result Status")
        for key, value in result_status.items():
            lines.append(f"- {key}: {value}")

    ui_preview = packet.get("metadata", {}).get("ui_preview")
    if ui_preview:
        lines.append("")
        lines.append("## UI Preview")
        for key, value in ui_preview.items():
            lines.append(f"- {key}: {value}")

    lines.append("")
    lines.append("## Signals")
    signals = packet.get("signals", {})
    if not signals:
        lines.append("- none")
    else:
        for sig_id, sig in signals.items():
            safe_sig = signal_public_summary(sig)
            lines.append(
                f"- {sig_id}: "
                f"{safe_sig.get('technical_type')} / "
                f"{safe_sig.get('formula_role')} / "
                f"{safe_sig.get('representation_space')} / "
                f"route={safe_sig.get('route_id')}"
            )

    lines.append("")
    lines.append("## Projections")
    projections = packet.get("projections", {})
    if not projections:
        lines.append("- none")
    else:
        for proj_id, proj in projections.items():
            sem = _format_semantic_summary(proj.get("semantic_summary", {}))
            num = _format_numeric_summary(proj.get("numeric_summary", {}))
            details = " / ".join([x for x in [sem, num] if x])
            suffix = f" / {details}" if details else ""
            lines.append(
                f"- {proj_id}: "
                f"{proj.get('operator_name')} "
                f"confidence={proj.get('confidence')}"
                f"{suffix}"
            )
            if proj.get("metadata", {}).get("warning"):
                lines.append(f"  - warning: {proj.get('metadata', {}).get('warning')}")

    noise_lines = []
    for proj_id, proj in projections.items():
        if proj.get("operator_name") == "NoisePossibilityReader":
            num = proj.get("numeric_summary", {})
            noise_lines.append(
                f"- {proj_id}: shape={num.get('shape')} mean={num.get('mean')} std={num.get('std')} "
                f"norm={num.get('norm')} seed={num.get('seed')} strength={num.get('noise_strength')} "
                f"mode={num.get('noise_mode')}"
            )
    if noise_lines:
        lines.append("")
        lines.append("## Noise Strategy")
        lines.extend(noise_lines)

    structured_lines = []
    priority_lines = []
    route_hint_lines = []
    expectation_lines = []
    for proj_id, proj in projections.items():
        sem = proj.get("semantic_summary", {})
        meta = proj.get("metadata", {})
        if sem.get("is_structured_strategy"):
            role = proj.get("role_vector", {})
            structured_lines.append(
                f"- {proj_id}: main={role.get('has_main_strategy')} anchors={role.get('has_anchors')} "
                f"active={role.get('has_active_changes')} contact={role.get('has_contact_rule')} "
                f"endpoint={role.get('has_endpoint')} forbidden_drift={role.get('has_forbidden_drift')}"
            )
            priority_map = meta.get("priority_map", {})
            if priority_map:
                for key, value in priority_map.items():
                    compact = str(value or "").replace("\n", " ").strip()
                    if len(compact) > 160:
                        compact = compact[:157] + "..."
                    priority_lines.append(f"- {key}: {compact if compact else '<empty>'}")
            route_hints = meta.get("route_hints", {})
            if route_hints:
                for key, value in route_hints.items():
                    compact = str(value or "").replace("\n", " ").strip()
                    if len(compact) > 160:
                        compact = compact[:157] + "..."
                    route_hint_lines.append(f"- {key}: {compact if compact else '<empty>'}")
            expectations = meta.get("strategy_expectations", {})
            if expectations:
                for key, value in expectations.items():
                    compact = str(value or "").replace("\n", " ").strip()
                    if len(compact) > 160:
                        compact = compact[:157] + "..."
                    expectation_lines.append(f"- {key}: {compact if compact else '<empty>'}")

    if structured_lines:
        lines.append("")
        lines.append("## Structured Strategy")
        lines.extend(structured_lines)
    if priority_lines:
        lines.append("")
        lines.append("## Priority Map")
        lines.extend(priority_lines)
    if route_hint_lines:
        lines.append("")
        lines.append("## Route Hints")
        lines.extend(route_hint_lines)
    if expectation_lines:
        lines.append("")
        lines.append("## Strategy Expectations")
        lines.extend(expectation_lines)

    boundary_lines = []
    for proj_id, proj in projections.items():
        if proj.get("operator_name") == "DeltaReader":
            num = proj.get("numeric_summary", {})
            boundary_lines.append(
                f"- {proj_id}: delta_norm={num.get('delta_norm')} relative_delta={num.get('relative_delta')} shape={num.get('shape')}"
            )
    if boundary_lines:
        lines.append("")
        lines.append("## Boundary Metrics")
        lines.extend(boundary_lines)

    sampler_lines = []
    for rel_id, rel in packet.get("relations", {}).items():
        meta = rel.get("metadata", {})
        if meta.get("boundary_type") == "sampler_summary":
            sampler_lines.append(
                f"- {rel_id}: sampler={meta.get('sampler_name')} scheduler={meta.get('scheduler')} "
                f"steps={meta.get('steps')} cfg={meta.get('cfg')} denoise={meta.get('denoise')} "
                f"seed={meta.get('seed')} delta_norm={meta.get('delta_norm')} relative_delta={meta.get('relative_delta')}"
            )
    if sampler_lines:
        lines.append("")
        lines.append("## Sampler Boundary Summary")
        lines.extend(sampler_lines)

    lines.append("")
    lines.append("## Relations")
    relations = packet.get("relations", {})
    if not relations:
        lines.append("- none")
    else:
        for rel_id, rel in relations.items():
            conflict_suffix = f" conflicts={rel.get('conflict_ids')}" if rel.get("conflict_ids") else ""
            lines.append(
                f"- {rel_id}: {rel.get('relation_type')} "
                f"sources={rel.get('source_signal_ids')} "
                f"targets={rel.get('target_signal_ids')} "
                f"local={rel.get('local_strategy_id')} "
                f"status={rel.get('equality_status')}"
                f"{conflict_suffix}"
            )
            if rel.get("formula_meaning"):
                lines.append(f"  - meaning: {rel.get('formula_meaning')}")

    lines.append("")
    lines.append("## SStates")
    sstates = packet.get("sstates", {})
    if not sstates:
        lines.append("- none")
    else:
        for s_id, sstate in sstates.items():
            conflict_suffix = f" conflicts={sstate.get('conflict_ids')}" if sstate.get("conflict_ids") else ""
            lines.append(
                f"- {s_id}: position={sstate.get('position')} "
                f"active_signals={len(sstate.get('active_signal_ids', []))} "
                f"active_relations={len(sstate.get('active_relation_ids', []))}"
                f"{conflict_suffix}"
            )
            if sstate.get("local_strategies"):
                lines.append("  - local strategies:")
                for local_id, local in sstate.get("local_strategies", {}).items():
                    lines.append(f"    - {local_id}: relation={local.get('relation_id')} type={local.get('relation_type')}")

    route_memory = packet.get("route_memory", {})
    alpha_records = []
    for record in packet.get("route_memory", {}).get("stage_records", []):
        if record.get("stage_name") == "EventCoreNodeAlpha":
            alpha_records.append(record)
    if alpha_records:
        lines.append("")
        lines.append("## EventCoreNodeAlpha")
        for record in alpha_records:
            lines.append(f"- {record.get('action')}: {record.get('observed_behavior')}")
            meta = record.get("metadata", {})
            if meta.get("adapter_mode"):
                lines.append(f"  - adapter_mode: {meta.get('adapter_mode')}")
            if meta.get("relation_ids"):
                lines.append(f"  - relation_ids: {meta.get('relation_ids')}")

    lines.append("")
    lines.append("## Route Memory")
    lines.append(f"- route_memory_id: {route_memory.get('id')}")
    lines.append(f"- signal_ids: {len(route_memory.get('signal_ids', []))}")
    lines.append(f"- projection_ids: {len(route_memory.get('projection_ids', []))}")
    lines.append(f"- relation_ids: {len(route_memory.get('relation_ids', []))}")
    lines.append(f"- sstate_ids: {len(route_memory.get('sstate_ids', []))}")
    lines.append(f"- conflict_ids: {len(route_memory.get('conflict_ids', []))}")
    lines.append(f"- stage_records: {len(route_memory.get('stage_records', []))}")
    lines.append(f"- sampler_step_records: {len(route_memory.get('sampler_step_records', []))}")

    stage_records = route_memory.get("stage_records", [])
    if stage_records:
        lines.append("")
        lines.append("### Stage Records")
        for index, record in enumerate(stage_records):
            stage = record.get("stage_name", "unknown")
            action = record.get("action", "unknown")
            behavior = record.get("observed_behavior", "")
            conflicts = record.get("conflict_ids", [])
            conflict_text = f" conflicts={conflicts}" if conflicts else ""
            lines.append(f"- [{index}] {stage} / {action} / {behavior}{conflict_text}")

    wan_lines = build_wan_report_block(packet)
    if wan_lines:
        lines.append("")
        lines.extend(wan_lines)

    lines.append("")
    lines.append("## Conflicts")
    conflicts = packet.get("conflicts", {})
    if conflicts:
        for conflict_id, conflict in conflicts.items():
            lines.append(
                f"- {conflict_id}: {conflict.get('conflict_type')} "
                f"severity={conflict.get('severity')} "
                f"stage={conflict.get('stage_position')}"
            )
            if conflict.get("observed_symptom"):
                lines.append(f"  - symptom: {conflict.get('observed_symptom')}")
            if conflict.get("suggested_response"):
                lines.append(f"  - suggested: {conflict.get('suggested_response')}")
    else:
        lines.append("- none")

    return "\n".join(lines)
