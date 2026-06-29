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


def _stage_records(packet):
    metadata = packet.get("metadata", {}) if isinstance(packet, dict) else {}
    records = metadata.get("execution_records", [])
    return records if isinstance(records, list) else []


def _latest_stage(records, stage_name):
    for record in reversed(records or []):
        if isinstance(record, dict) and str(record.get("stage", "") or "") == stage_name:
            return record
    return {}


def _latest_prefix(records, prefix):
    for record in reversed(records or []):
        if isinstance(record, dict) and str(record.get("stage", "") or "").startswith(prefix):
            return record
    return {}


def _preferred_pixel_region_motion(records):
    candidates = [
        record for record in (records or [])
        if isinstance(record, dict)
        and str(record.get("stage", "") or "") == "EventPixelRegionMotionMap"
    ]
    if not candidates:
        return {}

    def selected(record, reason, rank):
        out = dict(record)
        out["selected_for_strategy_return"] = True
        out["selection_reason"] = str(reason)
        out["selection_rank"] = int(rank)
        out["available_pixel_region_map_count"] = len(candidates)
        return out

    for source_stage, reason, rank in [
        ("EventMath_concatenated_frame_motion", "final_concatenated_visible_outcome", 1),
        ("EventMath_decoded_frame_motion", "decoded_visible_outcome", 2),
    ]:
        for record in reversed(candidates):
            if str(record.get("source_stage", "") or "") == source_stage:
                return selected(record, reason, rank)

    for record in reversed(candidates):
        source_stage = str(record.get("source_stage", "") or "")
        if source_stage.startswith("EventMath_cascade_") and source_stage.endswith("_frame_motion"):
            return selected(record, "latest_cascade_visible_outcome", 3)

    return selected(candidates[-1], "fallback_latest_pixel_region_record", 9)


def _compact_list(value, limit=6):
    if value is None:
        return ""
    if not isinstance(value, (list, tuple)):
        return str(value)
    items = [str(item) for item in value[:limit]]
    if len(value) > limit:
        items.append(f"...(+{len(value) - limit})")
    return ", ".join(items)


def _compact_issue_reasons(value, limit=5):
    if not isinstance(value, list):
        return ""
    reasons = []
    for item in value[:limit]:
        if isinstance(item, dict):
            reasons.append(str(item.get("reason", "unknown") or "unknown"))
        else:
            reasons.append(str(item))
    if len(value) > limit:
        reasons.append(f"...(+{len(value) - limit})")
    return ", ".join(reasons)


def _append_top_release_summary(lines, packet):
    records = _stage_records(packet)
    metadata = packet.get("metadata", {}) if isinstance(packet, dict) else {}
    program_status = metadata.get("event_program_status", {})
    if not isinstance(program_status, dict):
        program_status = {}

    runtime = _latest_stage(records, "SingularityRuntimeVersion")
    workflow_binding = _latest_stage(records, "EventWorkflowBinding")
    gate = _latest_stage(records, "EventCoreBodyCompletionGate")
    report_card = _latest_stage(records, "EventCoreBodyReportCard")
    release_gate = _latest_stage(records, "EventPublicReleaseReadinessGate")
    release_manifest = _latest_stage(records, "EventPublicReleaseCandidateManifest")
    public_surface = _latest_stage(records, "EventPublicSurfaceContract")
    public_package_static = _latest_stage(records, "EventPublicPackageStaticScan")
    math_ledger = _latest_stage(records, "EventMathTopologyLedger")
    math_graph = _latest_stage(records, "EventMathTopologyDependencyGraph")
    math_summary = _latest_stage(records, "EventMathControlSummary")
    prompt_lock = _latest_stage(records, "EventPromptPurityLock")
    prompt_apply = _latest_stage(records, "EventPromptStrategyTranscodeApply")
    visible_motion_return = _latest_stage(records, "EventVisibleMotionStrategyReturnGate")
    true_region_topology = _latest_stage(records, "EventTrueRegionTopologyEvidence")
    fractal_intersections = _latest_stage(records, "EventFractalStrategyIntersectionMap")
    region_weighted_fractal = _latest_stage(records, "EventRegionWeightedFractalStrategyReturn")
    pixel_region_motion = _preferred_pixel_region_motion(records)
    cascade_seam_impulse = _latest_stage(records, "EventCascadeSeamImpulseReview")
    tail_next_source_continuity = _latest_stage(records, "EventTailNextSourceStrategyContinuityProposal")
    cascade_seam_phase = _latest_stage(records, "EventCascadeSeamPhaseClassifier")
    semantic_phase_schedule = _latest_stage(records, "EventCascadeSemanticPhaseScheduleProposal")
    selected_tail_source_package = _latest_stage(records, "EventSelectedTailSourceReconstructionPackage")
    max_risk_strategy_package = _latest_stage(records, "EventMaxRiskStrategyRingPackage")
    action_background = _latest_stage(records, "EventActionBackgroundSeparationEvidence")
    pixel_pressure_review = _latest_stage(records, "EventPixelPressureDisagreementReview")
    pressure_pixel_reweighting = _latest_stage(records, "EventPressurePixelReweightingProposal")
    pressure_pixel_active = _latest_prefix(records, "EventPressurePixelReweightingActiveCandidate_")
    segment_entry_bridge = _latest_stage(records, "EventSegmentEntryLatentMemoryBridge")
    source_noise_birth_shaping = _latest_prefix(records, "EventSourceNoiseBirthShaping_")
    input_normalization = _latest_stage(records, "EventInputNormalization")
    r126_route = _latest_prefix(records, "EventR126LowMidWindowSpatialControlRoute_")

    if not any([runtime, workflow_binding, gate, report_card, release_gate, release_manifest, public_surface, public_package_static, math_ledger, math_graph, math_summary, visible_motion_return, true_region_topology, fractal_intersections, region_weighted_fractal, pixel_region_motion, cascade_seam_impulse, tail_next_source_continuity, cascade_seam_phase, semantic_phase_schedule, selected_tail_source_package, max_risk_strategy_package, action_background, pixel_pressure_review, pressure_pixel_reweighting, pressure_pixel_active, segment_entry_bridge, source_noise_birth_shaping]):
        return

    def first_value(*values, default=""):
        for value in values:
            if value not in (None, ""):
                return value
        return default

    runtime_version = first_value(
        runtime.get("runtime_version"),
        workflow_binding.get("runtime_version"),
        release_gate.get("runtime_version"),
        program_status.get("runtime_version"),
    )
    runtime_name = first_value(
        runtime.get("runtime_name"),
        workflow_binding.get("runtime_name"),
        release_gate.get("runtime_name"),
        program_status.get("runtime_name"),
    )
    result_status = first_value(
        release_gate.get("result_status"),
        program_status.get("result_status"),
        metadata.get("result_status"),
    )
    saved_video_path = first_value(
        program_status.get("saved_video_path"),
        metadata.get("saved_video_path"),
    )
    math_mode = first_value(
        release_gate.get("math_control_mode"),
        math_summary.get("math_control_mode"),
    )
    active_math_path = first_value(
        release_gate.get("active_generation_math_path"),
        math_summary.get("active_generation_math_path"),
    )
    prompt_transcode_mode = first_value(
        release_gate.get("prompt_transcode_mode"),
        prompt_apply.get("prompt_transcode_mode"),
        prompt_lock.get("prompt_transcode_mode"),
    )

    lines.append("")
    lines.append("## Top Summary")
    lines.append(f"- runtime: {runtime_version} / {runtime_name}")
    if result_status:
        lines.append(f"- result_status: {result_status}")
    if saved_video_path:
        lines.append(f"- saved_video_path: {saved_video_path}")
    if gate:
        lines.append(
            f"- completion_gate: {gate.get('status')} "
            f"(route_complete={gate.get('route_complete')}, final_output_ok={gate.get('final_output_ok')}, "
            f"missing_total={gate.get('missing_total')})"
        )
        blocking = gate.get("blocking_reasons", [])
        if blocking:
            lines.append(f"  - blocking_reasons: {_compact_list(blocking)}")
    if release_gate:
        lines.append(
            f"- public_release_readiness: {release_gate.get('status')} / {release_gate.get('severity')}"
        )
        if release_gate.get("does_not_measure_visual_quality"):
            lines.append("  - visual_quality_note: structural gate only; inspect the mp4 separately")
        blockers = _compact_issue_reasons(release_gate.get("blockers", []))
        warnings = _compact_issue_reasons(release_gate.get("warnings", []))
        research_flags = _compact_issue_reasons(release_gate.get("research_flags", []))
        if blockers:
            lines.append(f"  - blockers: {blockers}")
        if warnings:
            lines.append(f"  - warnings: {warnings}")
        if research_flags:
            lines.append(f"  - research_flags: {research_flags}")
        if release_gate.get("next_action"):
            lines.append(f"  - next_action: {release_gate.get('next_action')}")
    if release_manifest:
        lines.append(
            f"- public_release_candidate: {release_manifest.get('status')} / {release_manifest.get('severity')} "
            f"(can_package={release_manifest.get('can_package_public_archive')}, "
            f"human_video_review={release_manifest.get('requires_human_video_review')})"
        )
        manifest_blockers = _compact_issue_reasons(release_manifest.get("blockers", []))
        manifest_warnings = _compact_issue_reasons(release_manifest.get("warnings", []))
        manifest_research = _compact_issue_reasons(release_manifest.get("research_flags", []))
        if manifest_blockers:
            lines.append(f"  - package_blockers: {manifest_blockers}")
        if manifest_warnings:
            lines.append(f"  - package_warnings: {manifest_warnings}")
        if manifest_research:
            lines.append(f"  - package_research_flags: {manifest_research}")
        if release_manifest.get("next_action"):
            lines.append(f"  - package_next_action: {release_manifest.get('next_action')}")
    if public_surface:
        lines.append(
            f"- public_surface_contract: {public_surface.get('status')} / {public_surface.get('severity')} "
            f"(warnings={len(public_surface.get('warnings', []) or [])}, "
            f"research_flags={len(public_surface.get('research_flags', []) or [])})"
        )
    if public_package_static:
        lines.append(
            f"- public_package_static_scan: {public_package_static.get('status')} / {public_package_static.get('severity')} "
            f"(missing_files={len(public_package_static.get('missing_required_files', []) or [])}, "
            f"forbidden_dirs={public_package_static.get('forbidden_dir_count')}, "
            f"forbidden_files={public_package_static.get('forbidden_file_count')})"
        )
        static_blockers = _compact_issue_reasons(public_package_static.get("blockers", []))
        static_warnings = _compact_issue_reasons(public_package_static.get("warnings", []))
        if static_blockers:
            lines.append(f"  - package_static_blockers: {static_blockers}")
        if static_warnings:
            lines.append(f"  - package_static_warnings: {static_warnings}")
    if math_ledger:
        lines.append(
            f"- math_topology_ledger: {math_ledger.get('status')} / {math_ledger.get('severity')} "
            f"(present={math_ledger.get('present_surface_count')}/{math_ledger.get('surface_count')}, "
            f"active={math_ledger.get('active_generation_surface_count')}, "
            f"research={math_ledger.get('research_surface_count')})"
        )
        active_ids = _compact_list(math_ledger.get("active_generation_surface_ids", []))
        research_ids = _compact_list(math_ledger.get("research_surface_ids", []))
        missing_ids = _compact_list(math_ledger.get("missing_required_surface_ids", []))
        if active_ids:
            lines.append(f"  - active_generation_surfaces: {active_ids}")
        if research_ids:
            lines.append(f"  - research_or_diagnostic_surfaces: {research_ids}")
        if missing_ids:
            lines.append(f"  - missing_required_surfaces: {missing_ids}")
    if math_graph:
        lines.append(
            f"- math_topology_graph: {math_graph.get('status')} / {math_graph.get('severity')} "
            f"(nodes={math_graph.get('present_node_count')}/{math_graph.get('node_count')}, "
            f"edges={math_graph.get('present_edge_count')}/{math_graph.get('edge_count')}, "
            f"active_edges={math_graph.get('active_generation_edge_count')}, "
            f"research_edges={math_graph.get('research_edge_count')})"
        )
        active_edge_ids = _compact_list(math_graph.get("active_generation_edge_ids", []))
        research_edge_ids = _compact_list(math_graph.get("research_edge_ids", []))
        missing_edge_ids = _compact_list(math_graph.get("required_missing_edge_ids", []))
        if active_edge_ids:
            lines.append(f"  - active_generation_edges: {active_edge_ids}")
        if research_edge_ids:
            lines.append(f"  - research_edges: {research_edge_ids}")
        if missing_edge_ids:
            lines.append(f"  - missing_required_edges: {missing_edge_ids}")
    if visible_motion_return:
        lines.append(
            f"- visible_motion_return: {visible_motion_return.get('status')} / {visible_motion_return.get('severity')} "
            f"(next={visible_motion_return.get('next_control_surface')}, "
            f"active_next={visible_motion_return.get('visible_motion_active_control_allowed_next')})"
        )
        coupling = _compact_list(visible_motion_return.get("coupling_evidence", []))
        if coupling:
            lines.append(f"  - coupling_evidence: {coupling}")
    if true_region_topology:
        lines.append(
            f"- true_region_topology: {true_region_topology.get('status')} / {true_region_topology.get('severity')} "
            f"(readiness={true_region_topology.get('region_readiness_score')}, "
            f"next={true_region_topology.get('next_control_surface')}, "
            f"active_next={true_region_topology.get('true_region_active_control_allowed_next')})"
        )
        if true_region_topology.get("dominant_region_id"):
            lines.append(
                f"  - dominant_region: {true_region_topology.get('dominant_region_id')} "
                f"pressure={true_region_topology.get('dominant_region_pressure')}"
            )
        missing_region = _compact_list(true_region_topology.get("missing_evidence", []))
        if missing_region:
            lines.append(f"  - missing_region_evidence: {missing_region}")
    if fractal_intersections:
        lines.append(
            f"- fractal_strategy_intersections: {fractal_intersections.get('status')} / {fractal_intersections.get('severity')} "
            f"(depth={fractal_intersections.get('fractal_depth')}, "
            f"alignment={fractal_intersections.get('final_layer_alignment_score')}, "
            f"convergence={fractal_intersections.get('convergence_state')}, "
            f"next={fractal_intersections.get('next_control_surface')})"
        )
        dominant_axis = _compact_list(fractal_intersections.get("dominant_intersection_axis", []))
        if dominant_axis:
            lines.append(f"  - dominant_intersection_axis: {dominant_axis}")
    if region_weighted_fractal:
        lines.append(
            f"- region_weighted_fractal: {region_weighted_fractal.get('status')} / {region_weighted_fractal.get('severity')} "
            f"(confidence={region_weighted_fractal.get('region_axis_confidence')}, "
            f"match={region_weighted_fractal.get('dominant_axis_evidence_match')}, "
            f"guard={region_weighted_fractal.get('background_overweight_guard')}, "
            f"axis={region_weighted_fractal.get('guarded_visible_evidence_axis')}, "
            f"next={region_weighted_fractal.get('next_control_surface')})"
        )
        if region_weighted_fractal.get("center_action_vs_edge_background_ratio") not in (None, ""):
            lines.append(
                f"  - center_vs_edge_ratio: {region_weighted_fractal.get('center_action_vs_edge_background_ratio')} "
                f"background_overweight={region_weighted_fractal.get('background_overweight_score')}"
            )
    if pixel_region_motion:
        lines.append(
            f"- pixel_region_motion: {pixel_region_motion.get('status')} / {pixel_region_motion.get('severity')} "
            f"(center_edge={pixel_region_motion.get('center_edge_pixel_ratio')}, "
            f"edge_center={pixel_region_motion.get('edge_center_pixel_ratio')}, "
            f"seam={pixel_region_motion.get('estimated_seam_ratio')}, "
            f"leakage={pixel_region_motion.get('background_pixel_leakage_score')}, "
            f"source={pixel_region_motion.get('source_stage')}, "
            f"selection={pixel_region_motion.get('selection_reason')}, "
            f"next={pixel_region_motion.get('next_control_surface')})"
        )
    if cascade_seam_impulse:
        lines.append(
            f"- cascade_seam_impulse: {cascade_seam_impulse.get('status')} / {cascade_seam_impulse.get('severity')} "
            f"(score={cascade_seam_impulse.get('seam_impulse_score')}, "
            f"vector={cascade_seam_impulse.get('vector_seam_impulse_score')}, "
            f"visible={cascade_seam_impulse.get('visible_seam_delta_score')}, "
            f"visible_ratio={cascade_seam_impulse.get('visible_seam_over_median_ratio')}, "
            f"visible_rank={cascade_seam_impulse.get('visible_window_top_transition_rank')}, "
            f"boundary={cascade_seam_impulse.get('worst_boundary_index')}, "
            f"jump={cascade_seam_impulse.get('boundary_jump_score')}, "
            f"entry_accel={cascade_seam_impulse.get('entry_acceleration_score')}, "
            f"direction_switch={cascade_seam_impulse.get('direction_switch_score')}, "
            f"direction_cos={cascade_seam_impulse.get('tail_entry_direction_cosine')}, "
            f"next={cascade_seam_impulse.get('next_control_surface')})"
        )
    if tail_next_source_continuity:
        lines.append(
            f"- tail_next_source_continuity: {tail_next_source_continuity.get('status')} / {tail_next_source_continuity.get('severity')} "
            f"(pressure={tail_next_source_continuity.get('continuity_pressure_score')}, "
            f"source_gap={tail_next_source_continuity.get('source_gap_score')}, "
            f"entry_ratio={tail_next_source_continuity.get('entry_ratio_score')}, "
            f"source_delta={tail_next_source_continuity.get('source_frame_abs_delta')}, "
            f"source_over_median={tail_next_source_continuity.get('source_delta_over_global_median')}, "
            f"visible={tail_next_source_continuity.get('visible_seam_delta_score')}, "
            f"boundary={tail_next_source_continuity.get('target_boundary_index')}, "
            f"next={tail_next_source_continuity.get('next_control_surface')})"
        )
    if cascade_seam_phase:
        scores = cascade_seam_phase.get("axis_scores", {})
        if not isinstance(scores, dict):
            scores = {}
        lines.append(
            f"- seam_phase_classifier: {cascade_seam_phase.get('status')} / {cascade_seam_phase.get('severity')} "
            f"(dominant={cascade_seam_phase.get('dominant_axis')}, "
            f"score={cascade_seam_phase.get('dominant_score')}, "
            f"semantic={scores.get('semantic_phase_reentry')}, "
            f"text_change={scores.get('prompt_text_change')}, "
            f"prompt={scores.get('prompt_phase_reentry')}, "
            f"latent={scores.get('latent_carrier_mismatch')}, "
            f"background={scores.get('background_anchor_conflict')}, "
            f"center={scores.get('center_action_overdrive')}, "
            f"sampler={scores.get('sampler_handoff_reset')}, "
            f"next={cascade_seam_phase.get('next_control_surface')})"
        )
    if semantic_phase_schedule:
        lines.append(
            f"- semantic_phase_schedule: {semantic_phase_schedule.get('status')} / {semantic_phase_schedule.get('severity')} "
            f"(semantic={semantic_phase_schedule.get('semantic_phase_reentry_score')}, "
            f"text_change={semantic_phase_schedule.get('prompt_text_change_score')}, "
            f"prompt_clean={semantic_phase_schedule.get('prompt_carrier_clean')}, "
            f"active_carrier={semantic_phase_schedule.get('active_phase_window_carrier')}, "
            f"segments={semantic_phase_schedule.get('requested_segments')}, "
            f"next={semantic_phase_schedule.get('next_control_surface')})"
        )
    if selected_tail_source_package:
        lines.append(
            f"- safe_math_package: {selected_tail_source_package.get('status')} / {selected_tail_source_package.get('severity')} "
            f"(dominant_axis={selected_tail_source_package.get('dominant_axis')}, "
            f"rebirth_risk={selected_tail_source_package.get('rebirth_risk_score')}, "
            f"source_inheritance={selected_tail_source_package.get('source_inheritance_score')}, "
            f"continuity={selected_tail_source_package.get('continuity_pressure_score')}, "
            f"next={selected_tail_source_package.get('next_control_surface')})"
        )
    if max_risk_strategy_package:
        lines.append(
            f"- max_risk_math_package: {max_risk_strategy_package.get('status')} / {max_risk_strategy_package.get('severity')} "
            f"(active={max_risk_strategy_package.get('active_control_allowed')}, "
            f"hard_guard_override={max_risk_strategy_package.get('hard_guard_override_allowed')}, "
            f"ring_pressure={max_risk_strategy_package.get('strategy_ring_pressure_score')}, "
            f"rebirth_risk={max_risk_strategy_package.get('rebirth_risk_score')}, "
            f"surface={max_risk_strategy_package.get('active_mutation_surface')})"
        )
    if action_background:
        lines.append(
            f"- action_background_separation: {action_background.get('status')} / {action_background.get('severity')} "
            f"(confidence={action_background.get('separation_confidence')}, "
            f"action_bg_ratio={action_background.get('action_to_background_ratio')}, "
            f"leakage={action_background.get('background_leakage_score')}, "
            f"pixel_agree={action_background.get('pressure_pixel_agreement')}, "
            f"pixel_interp={action_background.get('background_leakage_interpretation')}, "
            f"seam={action_background.get('seam_interference_score')}, "
            f"axis={action_background.get('recommended_axis')}, "
            f"next={action_background.get('next_control_surface')})"
        )
    if pixel_pressure_review:
        lines.append(
            f"- pixel_pressure_review: {pixel_pressure_review.get('status')} / {pixel_pressure_review.get('severity')} "
            f"(scalar_overweight={pixel_pressure_review.get('scalar_pressure_overweight_score')}, "
            f"corrected_leakage={pixel_pressure_review.get('corrected_background_leakage_score')}, "
            f"corrected_action={pixel_pressure_review.get('corrected_action_center_confidence')}, "
            f"seam_locality={pixel_pressure_review.get('seam_locality_score')}, "
            f"axis={pixel_pressure_review.get('recommended_axis')}, "
            f"next={pixel_pressure_review.get('next_control_surface')})"
        )
    if pressure_pixel_reweighting:
        lines.append(
            f"- pressure_pixel_reweighting: {pressure_pixel_reweighting.get('status')} / {pressure_pixel_reweighting.get('severity')} "
            f"(pixel_trust={pressure_pixel_reweighting.get('pixel_outcome_trust_weight')}, "
            f"pressure_trust={pressure_pixel_reweighting.get('scalar_pressure_trust_weight')}, "
            f"background_factor={pressure_pixel_reweighting.get('bounded_background_pressure_factor')}, "
            f"action_factor={pressure_pixel_reweighting.get('action_preservation_factor')}, "
            f"seam_guard={pressure_pixel_reweighting.get('seam_protection_weight')}, "
            f"candidate={pressure_pixel_reweighting.get('bounded_reweighting_candidate')}, "
            f"next={pressure_pixel_reweighting.get('next_control_surface')})"
        )
    if pressure_pixel_active:
        lines.append(
            f"- pressure_pixel_active_candidate: {pressure_pixel_active.get('status')} / {pressure_pixel_active.get('severity')} "
            f"(branch={pressure_pixel_active.get('branch_key')}, "
            f"delta={pressure_pixel_active.get('candidate_delta')}, "
            f"strength={pressure_pixel_active.get('candidate_effective_strength')}, "
            f"quality_guard={((pressure_pixel_active.get('quality_guard') or {}).get('status') if isinstance(pressure_pixel_active.get('quality_guard'), dict) else None)}, "
            f"guard_factor={((pressure_pixel_active.get('quality_guard') or {}).get('quality_guard_factor') if isinstance(pressure_pixel_active.get('quality_guard'), dict) else None)}, "
            f"spatial_guard={((pressure_pixel_active.get('local_spatial_pressure_guard') or {}).get('status') if isinstance(pressure_pixel_active.get('local_spatial_pressure_guard'), dict) else None)}, "
            f"same_run_oracle={pressure_pixel_active.get('same_run_oracle')}, "
            f"source={pressure_pixel_active.get('source_runtime_version')}, "
            f"next={pressure_pixel_active.get('next_evidence')})"
        )
    if segment_entry_bridge:
        bridge_controls = segment_entry_bridge.get("bridge_controls", {})
        if not isinstance(bridge_controls, dict):
            bridge_controls = {}
        admissibility_guard = bridge_controls.get("admissibility_guard", {})
        if not isinstance(admissibility_guard, dict):
            admissibility_guard = {}
        selected_tail_source_carrier = bridge_controls.get("selected_tail_source_carrier", {})
        if not isinstance(selected_tail_source_carrier, dict):
            selected_tail_source_carrier = {}
        regional_guard = selected_tail_source_carrier.get("regional_guard", {})
        if not isinstance(regional_guard, dict):
            regional_guard = {}
        wan_bridge = segment_entry_bridge.get("wan_latent_bridge", {})
        if not isinstance(wan_bridge, dict):
            wan_bridge = {}
        concat_bridge = segment_entry_bridge.get("concat_latent_bridge", {})
        if not isinstance(concat_bridge, dict):
            concat_bridge = {}
        lines.append(
            f"- segment_entry_bridge: {segment_entry_bridge.get('status')} "
            f"(version={segment_entry_bridge.get('bridge_version')}, "
            f"mode={segment_entry_bridge.get('mode')}, "
            f"micro={segment_entry_bridge.get('source_noise_micro_bridge_requested')}, "
            f"package={bridge_controls.get('package')}, "
            f"surface={bridge_controls.get('micro_bridge_surface')}, "
            f"wan_alpha={bridge_controls.get('effective_wan_alpha')}, "
            f"concat_alpha={bridge_controls.get('effective_concat_alpha')}, "
            f"window={bridge_controls.get('entry_window_slices')}, "
            f"post_drop={bridge_controls.get('post_drop_entry_echo')}, "
            f"tail_source={selected_tail_source_carrier.get('status')}, "
            f"tail_pressure={selected_tail_source_carrier.get('selected_tail_source_pressure')}, "
            f"decay_floor={selected_tail_source_carrier.get('post_drop_decay_floor')}, "
            f"region_guard={regional_guard.get('status')}, "
            f"region_mean={regional_guard.get('mean')}, "
            f"region_top={regional_guard.get('top_band_mean')}, "
            f"region_center={regional_guard.get('center_mean')}, "
            f"guard={admissibility_guard.get('status')}, "
            f"micro_override={admissibility_guard.get('source_noise_micro_hard_guard_override')}, "
            f"wan={wan_bridge.get('status')}, "
            f"concat={concat_bridge.get('status')})"
        )
    if source_noise_birth_shaping:
        gain_policy = source_noise_birth_shaping.get("gain_policy", {})
        if not isinstance(gain_policy, dict):
            gain_policy = {}
        source_carrier = source_noise_birth_shaping.get("source_image_birth_carrier", {})
        if not isinstance(source_carrier, dict):
            source_carrier = {}
        conditioning_carrier = source_noise_birth_shaping.get("source_conditioning_birth_carrier", {})
        if not isinstance(conditioning_carrier, dict):
            conditioning_carrier = {}
        microdetail_guard = source_noise_birth_shaping.get("microdetail_guard", {})
        if not isinstance(microdetail_guard, dict):
            microdetail_guard = {}
        conditioning_microdetail_guard = conditioning_carrier.get("microdetail_guard", {})
        if not isinstance(conditioning_microdetail_guard, dict):
            conditioning_microdetail_guard = {}
        guard_status = microdetail_guard.get("status") or conditioning_microdetail_guard.get("status")
        guard_residual_ratio = microdetail_guard.get("high_frequency_residual_ratio")
        if guard_residual_ratio is None:
            guard_residual_ratio = conditioning_microdetail_guard.get("high_frequency_residual_ratio")
        spatial_protection = microdetail_guard.get("spatial_microdetail_protection", {})
        if not isinstance(spatial_protection, dict):
            spatial_protection = {}
        conditioning_spatial_protection = conditioning_carrier.get("spatial_microdetail_protection", {})
        if not isinstance(conditioning_spatial_protection, dict):
            conditioning_spatial_protection = {}
        spatial_protection_mean = spatial_protection.get("latent_mean")
        if spatial_protection_mean is None:
            spatial_protection_mean = conditioning_spatial_protection.get("conditioning_mean")
        spatial_protection_min = spatial_protection.get("latent_min")
        if spatial_protection_min is None:
            spatial_protection_min = conditioning_spatial_protection.get("conditioning_min")
        additive_anchor = microdetail_guard.get("additive_anchor_mask", {})
        if not isinstance(additive_anchor, dict):
            additive_anchor = {}
        conditioning_additive_anchor = conditioning_carrier.get("additive_anchor_mask", {})
        if not isinstance(conditioning_additive_anchor, dict):
            conditioning_additive_anchor = {}
        additive_anchor_mean = additive_anchor.get("latent_mean")
        if additive_anchor_mean is None:
            additive_anchor_mean = conditioning_additive_anchor.get("conditioning_mean")
        additive_anchor_min = additive_anchor.get("latent_min")
        if additive_anchor_min is None:
            additive_anchor_min = conditioning_additive_anchor.get("conditioning_min")
        lines.append(
            f"- source_noise_birth_shaping: {source_noise_birth_shaping.get('status')} "
            f"(active={source_noise_birth_shaping.get('active_tensor_mutation_applied')}, "
            f"conditioning_active={source_noise_birth_shaping.get('active_conditioning_mutation_applied')}, "
            f"intensity={source_noise_birth_shaping.get('intensity_multiplier')}, "
            f"source_carrier={source_carrier.get('status')}, "
            f"source_carrier_active={source_carrier.get('active')}, "
            f"source_carrier_pressure={source_carrier.get('pressure')}, "
            f"source_carrier_amplitude={source_carrier.get('amplitude')}, "
            f"conditioning_carrier={conditioning_carrier.get('status')}, "
            f"conditioning_amplitude={conditioning_carrier.get('amplitude')}, "
            f"microdetail_guard={guard_status}, "
            f"hf_residual_ratio={guard_residual_ratio}, "
            f"spatial_protect_min={spatial_protection_min}, "
            f"spatial_protect_mean={spatial_protection_mean}, "
            f"additive_anchor_min={additive_anchor_min}, "
            f"additive_anchor_mean={additive_anchor_mean}, "
            f"min_gain={gain_policy.get('min_gain')}, "
            f"mean_gain={gain_policy.get('mean_gain')}, "
            f"max_attenuation={gain_policy.get('max_attenuation')}, "
            f"route={source_noise_birth_shaping.get('route_label')})"
        )
    if math_mode or active_math_path:
        lines.append(f"- math_control: mode={math_mode}, active_path={active_math_path}")
    if prompt_transcode_mode:
        lines.append(f"- prompt_route: transcode_mode={prompt_transcode_mode}")
    if prompt_lock:
        lines.append(
            f"- prompt_purity_lock: {prompt_lock.get('status')} "
            f"(text_injection_allowed={prompt_lock.get('prompt_text_injection_allowed')}, "
            f"semantic_math_in_prompt_allowed={prompt_lock.get('semantic_math_in_prompt_allowed')})"
        )
    if input_normalization:
        lines.append(
            f"- input_normalization: adjustments={input_normalization.get('adjustment_count')} "
            f"signature={input_normalization.get('normalized_signature')}"
        )
    if r126_route:
        lines.append(
            f"- r126_low_mid_window_route: {r126_route.get('status')} "
            f"route={r126_route.get('route_key')} additional_sampler_calls={r126_route.get('additional_sampler_calls')}"
        )
    if report_card:
        lines.append(f"- report_card_next_action: {report_card.get('next_action')}")


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
    lines.append("# Singularity Report")

    _append_top_release_summary(lines, packet)

    # Event Core Body rich rendering heavily removed (physical cut #20)
    # The old comfort sections (Top Summary, Live Timeline, Barrier Records, etc.)
    # depended on data that has been systematically gutted.
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

    # cleanup_records rendering removed (physical cut #25): mechanism fully excised in previous cuts.
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

