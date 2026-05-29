def build_wan_report_block(packet):
    wan = packet.get("metadata", {}).get("wan_adapter")
    if not wan:
        return []

    lines = []
    lines.append("## Wan Adapter Diagnostics")
    lines.append(f"- enabled: {wan.get('enabled')}")
    lines.append(f"- mode: {wan.get('mode')}")

    route_status = wan.get("route_status", {})
    if route_status:
        lines.append("")
        lines.append("### Wan Route Status")
        for key, value in route_status.items():
            lines.append(f"- {key}: {value}")

    diagnostics = wan.get("diagnostics", {})
    if diagnostics:
        lines.append("")
        lines.append("### Wan Counts / Hints")
        for key, value in diagnostics.items():
            lines.append(f"- {key}: {value}")

    route_labels = wan.get("route_labels", {})
    if route_labels:
        lines.append("")
        lines.append("### Wan Route Labels")
        for key, value in route_labels.items():
            lines.append(f"- {key}: {value}")

    if wan.get("created_relation_ids"):
        lines.append("")
        lines.append("### Wan Created Relations")
        for rel_id in wan.get("created_relation_ids", []):
            lines.append(f"- {rel_id}")

    if wan.get("conflict_ids"):
        lines.append("")
        lines.append("### Wan Adapter Conflicts")
        for conflict_id in wan.get("conflict_ids", []):
            lines.append(f"- {conflict_id}")

    return lines
