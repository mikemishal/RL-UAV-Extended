from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import math
import os
import re
import struct
import tempfile
import xml.etree.ElementTree as ET
import zipfile
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from .application_error import ApplicationError
from .case_library import CaseLibrary
from .config import local_output_root
from .replay_contract import ReplayBundle, ReplayEpisode, ReplayFrame

EXPORT_ID_RE = re.compile(r"^[0-9a-f]{32}$")
SAFE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,100}$")
MAX_CAPTURE_BYTES = 30 * 1024 * 1024
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class CaseExportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["report", "episode_csv", "comparison_csv", "trajectory_svg", "telemetry_svg", "package_zip"]
    start_time_s: FiniteFloat | None = Field(default=None, ge=-1_000_000_000)
    end_time_s: FiniteFloat | None = Field(default=None, ge=-1_000_000_000)
    sync_mode: Literal["absolute", "detection"] = "absolute"
    camera: str = Field(default="recorded current view", max_length=100)

    @model_validator(mode="after")
    def valid_range(self):
        if self.start_time_s is not None and self.end_time_s is not None and self.start_time_s >= self.end_time_s:
            raise ValueError("Selected export range must have positive duration.")
        return self


class CaptureRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    replay_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    replay_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    png_base64: str = Field(min_length=16, max_length=MAX_CAPTURE_BYTES * 4 // 3 + 8)
    width: int = Field(ge=640, le=4096)
    height: int = Field(ge=360, le=4096)
    time_s: FiniteFloat = Field(ge=0)
    camera: str = Field(max_length=100)
    projection: Literal["perspective", "top-down"]


class ChartCaptureRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_filename: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,100}\.svg$")
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    png_base64: str = Field(min_length=16, max_length=MAX_CAPTURE_BYTES * 4 // 3 + 8)
    width: Literal[2100]
    height: int = Field(ge=200, le=4096)
    dpi: Literal[300] = 300


def _export_root(case_id: str) -> Path:
    root = local_output_root().resolve()
    target = root / "exports"
    if target.is_symlink():
        raise ApplicationError("INVALID_EXPORT_ROOT", "Export storage cannot be a symbolic link.", 500)
    target.mkdir(parents=True, exist_ok=True)
    target = target.resolve()
    if not target.is_relative_to(root):
        raise ApplicationError("INVALID_EXPORT_ROOT", "Export storage must remain under local output.", 500)
    case_root = target / case_id
    if case_root.is_symlink():
        raise ApplicationError("INVALID_EXPORT_ROOT", "Case export storage cannot be a symbolic link.", 403)
    case_root.mkdir(exist_ok=True)
    resolved = case_root.resolve()
    if not resolved.is_relative_to(target):
        raise ApplicationError("INVALID_EXPORT_ROOT", "Case export storage escaped local output.", 403)
    return resolved


def _atomic_bytes(path: Path, data: bytes) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".export-", suffix=".tmp", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        if temporary:
            temporary.unlink(missing_ok=True)
        raise ApplicationError("EXPORT_WRITE_FAILED", "Could not atomically write the requested export.", 500) from exc


def _finite(value: float | None) -> str | float:
    return "" if value is None else value


def _checked_png(content: bytes, expected_width: int, expected_height: int) -> bytes:
    if len(content) > MAX_CAPTURE_BYTES or not content.startswith(PNG_SIGNATURE):
        raise ApplicationError("INVALID_CAPTURE", "Capture must be a bounded PNG file.", 422)
    offset = len(PNG_SIGNATURE)
    dimensions = None
    has_data = False
    has_end = False
    chunks: list[tuple[bytes, bytes]] = []
    while offset + 12 <= len(content):
        length = int.from_bytes(content[offset:offset + 4], "big")
        end = offset + 12 + length
        if end > len(content):
            break
        kind = content[offset + 4:offset + 8]
        data = content[offset + 8:offset + 8 + length]
        checksum = int.from_bytes(content[offset + 8 + length:end], "big")
        if zlib.crc32(kind + data) & 0xffffffff != checksum:
            raise ApplicationError("INVALID_CAPTURE", "PNG contains an invalid chunk checksum.", 422)
        if kind == b"IHDR":
            if length != 13 or dimensions is not None:
                raise ApplicationError("INVALID_CAPTURE", "PNG has an invalid image header.", 422)
            dimensions = (int.from_bytes(data[:4], "big"), int.from_bytes(data[4:8], "big"))
        elif kind == b"IDAT":
            has_data = True
        elif kind == b"IEND":
            if length != 0 or end != len(content):
                raise ApplicationError("INVALID_CAPTURE", "PNG has an invalid end marker or trailing data.", 422)
            has_end = True
        chunks.append((kind, data))
        offset = end
        if has_end:
            break
    if not has_end or not has_data or dimensions != (expected_width, expected_height):
        raise ApplicationError("INVALID_CAPTURE", "PNG dimensions or required image chunks do not match the request.", 422)
    return content


def _png_with_dpi(content: bytes, dpi: int = 300) -> bytes:
    pixels_per_meter = round(dpi / 0.0254)
    payload = struct.pack(">IIB", pixels_per_meter, pixels_per_meter, 1)
    chunk_type = b"pHYs"
    chunk = struct.pack(">I", len(payload)) + chunk_type + payload + struct.pack(">I", zlib.crc32(chunk_type + payload) & 0xffffffff)
    chunks = []
    offset = len(PNG_SIGNATURE)
    inserted = False
    while offset < len(content):
        length = int.from_bytes(content[offset:offset + 4], "big")
        end = offset + 12 + length
        kind = content[offset + 4:offset + 8]
        if kind == b"pHYs":
            offset = end
            continue
        if kind == b"IEND" and not inserted:
            chunks.append(chunk)
            inserted = True
        chunks.append(content[offset:end])
        offset = end
    if not inserted:
        raise ApplicationError("INVALID_CAPTURE", "PNG end marker was not found.", 422)
    return PNG_SIGNATURE + b"".join(chunks)


def _native_angle(lead: tuple[float, float, float], requested: tuple[float, float, float]) -> float | None:
    lead_norm = math.sqrt(sum(item * item for item in lead))
    requested_norm = math.sqrt(sum(item * item for item in requested))
    if lead_norm <= 1e-12 or requested_norm <= 1e-12:
        return None
    cosine = sum(a * b for a, b in zip(lead, requested)) / (lead_norm * requested_norm)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _frame_rows(episode: ReplayEpisode, *, offset_s: float = 0, side: str = "single", start: float | None = None, end: float | None = None):
    cumulative_path = 0.0
    previous: tuple[float, float, float] | None = None
    for frame in episode.frames:
        position = frame.defender.position_m
        if previous is not None:
            cumulative_path += math.sqrt(sum((position[i] - previous[i]) ** 2 for i in range(3)))
        previous = position
        axis_time = frame.time_s - offset_s
        if (start is not None and axis_time < start) or (end is not None and axis_time > end):
            continue
        telemetry = frame.telemetry
        action_telemetry = None if frame.terminated or frame.truncated else telemetry
        lead = telemetry.lrppo.lead_direction if telemetry and telemetry.lrppo else None
        angle = action_telemetry.lrppo.native_adjustment_deg if action_telemetry and action_telemetry.lrppo else (
            _native_angle(lead, action_telemetry.requested_action) if action_telemetry and lead else None
        )
        policy_visible = telemetry.decision_policy_visible if telemetry else frame.policy_visible
        truth = frame.privileged_diagnostics.attacker_position_m if telemetry else None
        estimate = policy_visible.e_hat_m or policy_visible.enemy_measurement_m
        tracking_error = math.sqrt(sum((truth[index] - estimate[index]) ** 2 for index in range(3))) if truth and estimate else None
        yield {
            "case_side": side,
            "frame_step": frame.step,
            "time_s": frame.time_s,
            "comparison_axis_time_s": frame.time_s - offset_s,
            "defender_x_m": position[0], "defender_y_m": position[1], "defender_z_m": position[2],
            "attacker_x_m": frame.attacker.position_m[0], "attacker_y_m": frame.attacker.position_m[1], "attacker_z_m": frame.attacker.position_m[2],
            "asset_x_m": frame.asset.position_m[0], "asset_y_m": frame.asset.position_m[1], "asset_z_m": frame.asset.position_m[2],
            "defender_vx_mps": _finite(frame.defender.velocity_mps[0] if frame.defender.velocity_mps else None),
            "defender_vy_mps": _finite(frame.defender.velocity_mps[1] if frame.defender.velocity_mps else None),
            "defender_vz_mps": _finite(frame.defender.velocity_mps[2] if frame.defender.velocity_mps else None),
            "attacker_vx_mps": _finite(frame.attacker.velocity_mps[0] if frame.attacker.velocity_mps else None),
            "attacker_vy_mps": _finite(frame.attacker.velocity_mps[1] if frame.attacker.velocity_mps else None),
            "attacker_vz_mps": _finite(frame.attacker.velocity_mps[2] if frame.attacker.velocity_mps else None),
            "asset_vx_mps": _finite(frame.asset.velocity_mps[0] if frame.asset.velocity_mps else None),
            "asset_vy_mps": _finite(frame.asset.velocity_mps[1] if frame.asset.velocity_mps else None),
            "asset_vz_mps": _finite(frame.asset.velocity_mps[2] if frame.asset.velocity_mps else None),
            "detection": policy_visible.enemy_detected,
            "outcome": frame.outcome if frame.outcome != "ongoing" else "",
            "defender_attacker_range_m": frame.privileged_diagnostics.defender_attacker_range_m,
            "attacker_asset_range_m": frame.privileged_diagnostics.attacker_asset_range_m,
            "requested_x": _finite(action_telemetry.requested_action[0] if action_telemetry else None),
            "requested_y": _finite(action_telemetry.requested_action[1] if action_telemetry else None),
            "requested_z": _finite(action_telemetry.requested_action[2] if action_telemetry else None),
            "desired_x": _finite(action_telemetry.desired_command[0] if action_telemetry and action_telemetry.desired_command else None),
            "desired_y": _finite(action_telemetry.desired_command[1] if action_telemetry and action_telemetry.desired_command else None),
            "desired_z": _finite(action_telemetry.desired_command[2] if action_telemetry and action_telemetry.desired_command else None),
            "safe_x": _finite(action_telemetry.safe_command[0] if action_telemetry and action_telemetry.safe_command else None),
            "safe_y": _finite(action_telemetry.safe_command[1] if action_telemetry and action_telemetry.safe_command else None),
            "safe_z": _finite(action_telemetry.safe_command[2] if action_telemetry and action_telemetry.safe_command else None),
            "executed_vx_mps": _finite(telemetry.executed_velocity_mps[0] if telemetry else None),
            "executed_vy_mps": _finite(telemetry.executed_velocity_mps[1] if telemetry else None),
            "executed_vz_mps": _finite(telemetry.executed_velocity_mps[2] if telemetry else None),
            "requested_to_velocity_angle_deg": _finite(action_telemetry.requested_to_velocity_angle_deg if action_telemetry else None),
            "lrppo_native_3d_correction_deg": _finite(angle),
            "kalman_tracking_error_m": _finite(tracking_error if policy_visible.estimator == "kalman" else None),
            "safety_intervention": _finite(float(action_telemetry.avoidance_active) if action_telemetry else None),
            "decision_step": _finite(action_telemetry.decision_step if action_telemetry else None),
            "decision_action_applied": _finite(float(action_telemetry.controller_action_executed) if action_telemetry else None),
            "cumulative_defender_path_length_m": cumulative_path,
        }


CSV_COLUMNS = [
    "case_side", "frame_step", "time_s", "comparison_axis_time_s",
    "defender_x_m", "defender_y_m", "defender_z_m", "attacker_x_m", "attacker_y_m", "attacker_z_m",
    "asset_x_m", "asset_y_m", "asset_z_m",
    "defender_vx_mps", "defender_vy_mps", "defender_vz_mps",
    "attacker_vx_mps", "attacker_vy_mps", "attacker_vz_mps",
    "asset_vx_mps", "asset_vy_mps", "asset_vz_mps", "detection", "outcome",
    "defender_attacker_range_m", "attacker_asset_range_m", "requested_x", "requested_y", "requested_z",
    "desired_x", "desired_y", "desired_z", "safe_x", "safe_y", "safe_z",
    "executed_vx_mps", "executed_vy_mps", "executed_vz_mps", "requested_to_velocity_angle_deg",
    "lrppo_native_3d_correction_deg", "kalman_tracking_error_m", "safety_intervention",
    "decision_step", "decision_action_applied", "cumulative_defender_path_length_m",
]


def _csv_bytes(rows) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _svg_trajectory(
    episodes: list[ReplayEpisode],
    labels: list[str],
    statuses: list[str],
    *,
    start: float | None = None,
    end: float | None = None,
    sync_mode: str = "absolute",
) -> bytes:
    frames_by_episode = []
    for episode in episodes:
        offset = episode.summary.detection_time_s or 0.0 if sync_mode == "detection" else 0.0
        frames = [frame for frame in episode.frames
                  if (start is None or frame.time_s - offset >= start)
                  and (end is None or frame.time_s - offset <= end)]
        if not frames:
            raise ApplicationError("EMPTY_EXPORT_RANGE", "Selected time range contains no recorded frames for at least one case replay.", 422)
        frames_by_episode.append(frames)
    points = [frame for frames in frames_by_episode for frame in frames]
    xs = [frame.defender.position_m[0] for frame in points] + [frame.attacker.position_m[0] for frame in points] + [frame.asset.position_m[0] for frame in points]
    ys = [frame.defender.position_m[1] for frame in points] + [frame.attacker.position_m[1] for frame in points] + [frame.asset.position_m[1] for frame in points]
    for episode in episodes:
        for obstacle in episode.static_obstacles:
            xs.extend([obstacle.min_corner_m[0], obstacle.max_corner_m[0]])
            ys.extend([obstacle.min_corner_m[1], obstacle.max_corner_m[1]])
    low_x, high_x = min(xs) - 5, max(xs) + 5
    low_y, high_y = min(ys) - 5, max(ys) + 5
    width, height, margin = 1200, 820, 90
    scale = min((width - 2 * margin) / max(high_x - low_x, 1e-9), (height - 2 * margin) / max(high_y - low_y, 1e-9))
    def xy(x: float, y: float) -> tuple[float, float]:
        return margin + (x - low_x) * scale, height - margin - (y - low_y) * scale
    colors = ["#1583dc", "#d1495b"]
    chunks = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
              '<rect width="100%" height="100%" fill="white"/>',
              '<text x="600" y="34" text-anchor="middle" font-family="sans-serif" font-size="22">Recorded trajectories — XY projection (m)</text>',
              f'<text x="600" y="58" text-anchor="middle" font-family="sans-serif" font-size="13">{_xml(" · ".join(statuses))}</text>']
    for episode_index, (episode, frames) in enumerate(zip(episodes, frames_by_episode)):
        color = colors[episode_index % len(colors)]
        for obstacle in episode.static_obstacles:
            left, bottom = xy(obstacle.min_corner_m[0], obstacle.min_corner_m[1])
            right, top = xy(obstacle.max_corner_m[0], obstacle.max_corner_m[1])
            chunks.append(f'<rect x="{left:.2f}" y="{top:.2f}" width="{right-left:.2f}" height="{bottom-top:.2f}" fill="#9aa6ad" fill-opacity=".2" stroke="#7a858b"/>')
        for entity_name, entity_color, getter in (
            ("Defender", color, lambda frame: frame.defender.position_m),
            ("Attacker", "#d1495b" if episode_index == 0 else "#ee8b37", lambda frame: frame.attacker.position_m),
            ("Protected asset", "#368b5a", lambda frame: frame.asset.position_m),
        ):
            positions = [getter(frame) for frame in frames]
            polyline = " ".join(f"{xy(point[0], point[1])[0]:.2f},{xy(point[0], point[1])[1]:.2f}" for point in positions)
            chunks.append(f'<polyline points="{polyline}" fill="none" stroke="{entity_color}" stroke-width="2.5"/>')
            for point, label in ((positions[0], "start"), (positions[-1], "end")):
                px, py = xy(point[0], point[1])
                chunks.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="4" fill="{entity_color}"><title>{_xml(labels[episode_index])} {entity_name} {label}</title></circle>')
    chunks.extend([
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="black"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="black"/>',
        f'<text x="{width/2}" y="{height-24}" text-anchor="middle" font-family="sans-serif">X (m)</text>',
        f'<text transform="translate(24 {height/2}) rotate(-90)" text-anchor="middle" font-family="sans-serif">Y (m)</text>',
        '<text x="92" y="790" font-family="sans-serif" font-size="12">XY projection; altitude is omitted from this plane. Trajectories use recorded native coordinates.</text>',
    ])
    for tick in range(6):
        x_value = low_x + (high_x - low_x) * tick / 5
        y_value = low_y + (high_y - low_y) * tick / 5
        x, _ = xy(x_value, low_y)
        _, y = xy(low_x, y_value)
        chunks.extend([
            f'<line x1="{x:.2f}" y1="{height-margin}" x2="{x:.2f}" y2="{height-margin+5}" stroke="black"/>',
            f'<text x="{x:.2f}" y="{height-margin+20}" text-anchor="middle" font-family="sans-serif" font-size="11">{x_value:.3g}</text>',
            f'<line x1="{margin-5}" y1="{y:.2f}" x2="{margin}" y2="{y:.2f}" stroke="black"/>',
            f'<text x="{margin-9}" y="{y+4:.2f}" text-anchor="end" font-family="sans-serif" font-size="11">{y_value:.3g}</text>',
        ])
    chunks.append("</svg>")
    return "".join(chunks).encode("utf-8")


def _xml(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _svg_chart(
    episodes: list[ReplayEpisode],
    labels: list[str],
    metric: str,
    title: str,
    unit: str,
    getter,
    *,
    start: float | None = None,
    end: float | None = None,
    sync_mode: str = "absolute",
) -> bytes:
    series = []
    for label, episode in zip(labels, episodes):
        offset = episode.summary.detection_time_s or 0.0 if sync_mode == "detection" else 0.0
        values = [(frame.time_s - offset, getter(frame)) for frame in episode.frames
                  if (start is None or frame.time_s - offset >= start)
                  and (end is None or frame.time_s - offset <= end)]
        series.append((label, [(time_s, float(value)) for time_s, value in values if value is not None and math.isfinite(float(value))]))
    all_values = [value for _, values in series for _, value in values]
    all_times = [time_s for _, values in series for time_s, _ in values]
    width, height, left, right, top, bottom = 1200, 700, 95, 40, 70, 90
    min_x, max_x = min(all_times, default=0), max(all_times, default=1)
    if math.isclose(min_x, max_x):
        max_x = min_x + 1
    min_y, max_y = min(all_values, default=0), max(all_values, default=1)
    if math.isclose(min_y, max_y):
        max_y = min_y + 1
    def px(x: float) -> float: return left + (x - min_x) / (max_x - min_x) * (width - left - right)
    def py(y: float) -> float: return height - bottom - (y - min_y) / (max_y - min_y) * (height - top - bottom)
    chunks = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
              '<rect width="100%" height="100%" fill="white"/>',
              f'<text x="{width/2}" y="35" text-anchor="middle" font-family="sans-serif" font-size="22">{_xml(title)}</text>']
    for tick in range(6):
        value = min_y + (max_y - min_y) * tick / 5
        y = py(value)
        chunks.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#dce2e5"/>')
        chunks.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{value:.3g}</text>')
    chunks.extend([f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="black"/>',
                   f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="black"/>',
                   f'<text x="{width/2}" y="{height-28}" text-anchor="middle" font-family="sans-serif">{"Comparison" if sync_mode == "detection" and len(episodes) == 2 else "Detection-relative" if sync_mode == "detection" else "Simulation"} time (s)</text>',
                   f'<text transform="translate(22 {height/2}) rotate(-90)" text-anchor="middle" font-family="sans-serif">{_xml(metric)} ({_xml(unit)})</text>'])
    colors = ["#1769aa", "#d1495b"]
    for index, (label, values) in enumerate(series):
        path = " ".join(("M" if point == 0 else "L") + f"{px(x):.2f},{py(y):.2f}" for point, (x, y) in enumerate(values))
        chunks.append(f'<path d="{path}" fill="none" stroke="{colors[index % 2]}" stroke-width="2.5"/>')
        if values:
            offset = episodes[index].summary.detection_time_s or 0.0 if sync_mode == "detection" else 0.0
            terminal_x = px(episodes[index].summary.duration_s - offset)
            chunks.append(f'<line x1="{terminal_x:.2f}" y1="{top}" x2="{terminal_x:.2f}" y2="{height-bottom}" stroke="{colors[index % 2]}" stroke-dasharray="5 4"/>')
        chunks.append(f'<text x="{left + index*250}" y="60" font-family="sans-serif" font-size="14" fill="{colors[index % 2]}">{_xml(label)}</text>')
    chunks.append('</svg>')
    return "".join(chunks).encode("utf-8")


def _report(case: dict, bundles: list[ReplayBundle]) -> bytes:
    lines = [
        f"# {case['title']}", "",
        f"- Case ID: `{case['case_id']}`",
        f"- Case type: {case['case_type']}",
        f"- Case validation: {case['comparison_validation']['status'] if case['comparison_validation'] else 'single replay'}",
        f"- Description: {case['description'] or 'Not provided.'}",
        "",
        "## Recorded episode evidence", "",
    ]
    for index, (reference, bundle) in enumerate(zip(case["replay_references"], bundles), start=1):
        episode = bundle.episodes[0]
        label = f"Policy {'AB'[index-1]}" if len(bundles) == 2 else "Episode"
        lines.extend([
            f"### {label} — {episode.defender.policy_id}", "",
            f"- Replay ID and gzip SHA-256: `{reference['replay_id']}` / `{reference['sha256']}`",
            f"- Defender generation/seed/checkpoint: {episode.defender.generation or 'n/a'} / {episode.defender.training_seed if episode.defender.training_seed is not None else 'n/a'} / `{episode.defender.checkpoint.sha256 if episode.defender.checkpoint else 'analytical'}`",
            f"- Attacker: {episode.attacker.policy_id}; clutter/perception: {episode.environment.clutter}/{episode.environment.awareness}; evaluation seed: {episode.simulation_seed}",
            f"- Native outcome: `{episode.summary.outcome}`; duration: {episode.summary.duration_s:.3f} s; detection: {episode.summary.detection_time_s if episode.summary.detection_time_s is not None else 'not recorded'} s",
            f"- Interception time: {episode.summary.interception_time_s if episode.summary.interception_time_s is not None else 'not recorded'} s; final defender-attacker range: {episode.summary.final_defender_attacker_distance_m:.4f} m; attacker-asset range: {episode.summary.final_attacker_asset_distance_m:.4f} m",
            f"- Source commit: `{bundle.provenance.simulation_source_commit}`; source dirty flag: `{bundle.provenance.source_dirty}`; visualizer dirty flag: `{bundle.provenance.visualizer_dirty}`; Python: `{bundle.provenance.python_version}`",
            f"- Resolved environment fingerprint: `{episode.environment.config_sha256}`; replay schema: `{bundle.schema_version}`",
            "",
        ])
    if case["comparison_validation"]:
        lines.extend(["## Comparison eligibility", "", f"- Classification: **{case['comparison_validation']['status']}**"])
        lines.extend(f"- {reason}" for reason in case["comparison_validation"]["reasons"])
        lines.append("")
    lines.extend(["## Researcher annotations", ""])
    if case["notes"]:
        lines.extend([case["notes"], ""])
    for bookmark in case["bookmarks"]:
        lines.extend([
            f"### {bookmark['title']} — t={bookmark['time_s']:.3f} s (frame {bookmark['frame_step']})",
            f"- Observation (researcher-entered): {bookmark['observation'] or 'Not provided.'}",
            f"- Interpretation / hypothesis (not established causation): {bookmark['interpretation'] or 'Not provided.'}",
            f"- Tags: {', '.join(bookmark['tags']) or 'none'}", "",
        ])
    lines.extend([
        "## Figures", "",
        "- `trajectory-xy.svg` — native recorded coordinates projected onto XY; altitude is not represented in this projection.",
        "- `telemetry-lrppo-correction.svg` and `telemetry-kalman-tracking.svg` — included only where source data is available.",
        "",
        "## Limitations", "",
        "- This case is a single episode or exploratory matched-seed example, not a population estimate.",
        "- Dirty-source flags and comparison eligibility are reproduced as recorded; historical provenance is not rewritten.",
        "- Later attacker, asset, and stochastic trajectories are not assumed coupled by a shared initial seed.",
        "- Controller-internal metrics absent from the replay remain unavailable.",
        "- Policy-visible rendering is a partial reconstruction; visual evidence alone does not establish causal mechanism.",
        "- No policy-superiority or statistical-significance claim is generated.",
        "",
    ])
    return "\n".join(lines).encode("utf-8")


def _manifest(case: dict, bundles: list[ReplayBundle], files: dict[str, bytes], request: CaseExportRequest) -> dict:
    episodes = [bundle.episodes[0] for bundle in bundles]
    return {
        "schema_version": 1,
        "case_id": case["case_id"],
        "case_version": case["version"],
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "replay_traces_included": False,
        "replay_references": case["replay_references"],
        "selected_time_range_s": {
            "start": request.start_time_s,
            "end": request.end_time_s,
        },
        "timeline_sync_mode": request.sync_mode,
        "camera_or_projection": request.camera,
        "simulator_source_commits": sorted({bundle.provenance.simulation_source_commit for bundle in bundles}),
        "coordinate_system": "simulation_xyz_z_up_m",
        "units": {"position": "m", "velocity": "m/s", "time": "s", "angle": "degrees"},
        "derivations": {
            "defender_path_length": "sum of Euclidean distances between consecutive recorded native 3D defender positions",
            "lrppo_native_3d_correction": "recorded native_adjustment_deg; if absent, acos(clamp(dot(lead, requested)/(norm(lead)*norm(requested)))) in degrees",
            "kalman_tracking_error": "Euclidean distance between policy-visible e_hat_m or measurement and recorded privileged attacker position; researcher-only export",
        },
        "artifact_types": {
            "trajectory-xy.svg": "derived XY projection from recorded native positions and static obstacle footprints",
            "telemetry-*.svg": "derived chart over recorded sample times; no samples after terminal state",
            "episode.csv": "exact frame values; action fields describe recorded k→k+1 decisions; terminal frame action cells are blank",
        },
        "outcomes": [episode.summary.outcome for episode in episodes],
        "files": {name: hashlib.sha256(content).hexdigest() for name, content in sorted(files.items())},
    }


def _telemetry_exports(
    episodes: list[ReplayEpisode],
    labels: list[str],
    start: float | None,
    end: float | None,
    sync_mode: str,
) -> dict[str, bytes]:
    common = {"start": start, "end": end, "sync_mode": sync_mode}
    return {
        "telemetry-defender-attacker-distance.svg": _svg_chart(episodes, labels, "Distance", "Defender to attacker range", "m", lambda frame: frame.privileged_diagnostics.defender_attacker_range_m, **common),
        "telemetry-attacker-asset-distance.svg": _svg_chart(episodes, labels, "Distance", "Attacker to protected asset range", "m", lambda frame: frame.privileged_diagnostics.attacker_asset_range_m, **common),
        "telemetry-lrppo-correction.svg": _svg_chart(episodes, labels, "Net native correction", "LR-PPO native 3D correction angle", "degrees", lambda frame: frame.telemetry.lrppo.native_adjustment_deg if frame.telemetry and frame.telemetry.lrppo else None, **common),
        "telemetry-kalman-tracking.svg": _svg_chart(episodes, labels, "Tracking error", "PPO-Kalman position tracking error", "m", lambda frame: frame.privileged_diagnostics.tracking_error_m, **common),
        "telemetry-requested-turn.svg": _svg_chart(episodes, labels, "Action turn", "Requested to executed movement angle", "degrees", lambda frame: frame.telemetry.requested_to_velocity_angle_deg if frame.telemetry else None, **common),
    }


def _saved_chart_pngs(case_id: str) -> dict[str, bytes]:
    pngs: dict[str, bytes] = {}
    for manifest in list_exports(case_id):
        for item in manifest.get("files", []):
            name = item.get("name")
            if item.get("kind") != "derived_svg_rasterization" or not isinstance(name, str) or not name.endswith(".png"):
                continue
            path = export_file_path(case_id, manifest["export_id"], name)
            try:
                content = path.read_bytes()
            except OSError as exc:
                raise ApplicationError("EXPORT_READ_FAILED", "Could not read a previously rasterized chart for the package.", 500) from exc
            if hashlib.sha256(content).hexdigest() != item.get("sha256"):
                raise ApplicationError("EXPORT_HASH_MISMATCH", "A previously rasterized chart changed on disk.", 409)
            pngs[name] = content
    return pngs


def build_export(case_library: CaseLibrary, case_id: str, request: CaseExportRequest) -> dict:
    case, bundles = case_library.load_for_export(case_id)
    episodes = [bundle.episodes[0] for bundle in bundles]
    start = request.start_time_s
    outputs: dict[str, bytes] = {}
    labels = [f"{episode.defender.policy_id} {episode.defender.generation or ''} seed {episode.defender.training_seed}" for episode in episodes]
    if request.kind == "episode_csv" or request.kind == "comparison_csv":
        if request.kind == "episode_csv" and len(episodes) != 1:
            raise ApplicationError("EXPORT_KIND_MISMATCH", "Episode CSV requires a single-episode case.", 422)
        if request.kind == "comparison_csv" and len(episodes) != 2:
            raise ApplicationError("EXPORT_KIND_MISMATCH", "Comparison CSV requires a comparison case.", 422)
        rows = []
        for index, episode in enumerate(episodes):
            offset = (episode.summary.detection_time_s or 0.0) if request.sync_mode == "detection" else 0.0
            rows.extend(_frame_rows(episode, offset_s=offset, side="AB"[index] if len(episodes) == 2 else "single",
                                    start=start, end=request.end_time_s))
        outputs["comparison.csv" if len(episodes) == 2 else "episode.csv"] = _csv_bytes(rows)
    elif request.kind == "trajectory_svg":
        outputs["trajectory-xy.svg"] = _svg_trajectory(
            episodes, labels, [case["comparison_validation"]["status"] if case["comparison_validation"] else "single replay"],
            start=start, end=request.end_time_s, sync_mode=request.sync_mode,
        )
    elif request.kind == "telemetry_svg":
        outputs.update(_telemetry_exports(episodes, labels, start, request.end_time_s, request.sync_mode))
    elif request.kind == "report":
        outputs["case-report.md"] = _report(case, bundles)
        outputs["trajectory-xy.svg"] = _svg_trajectory(
            episodes, labels, [case["comparison_validation"]["status"] if case["comparison_validation"] else "single replay"],
            start=start, end=request.end_time_s, sync_mode=request.sync_mode,
        )
        outputs.update(_telemetry_exports(episodes, labels, start, request.end_time_s, request.sync_mode))
    elif request.kind == "package_zip":
        rows = []
        for index, episode in enumerate(episodes):
            offset = (episode.summary.detection_time_s or 0.0) if request.sync_mode == "detection" else 0.0
            rows.extend(_frame_rows(
                episode, offset_s=offset, side="AB"[index] if len(episodes) == 2 else "single",
                start=start, end=request.end_time_s,
            ))
        outputs["case-report.md"] = _report(case, bundles)
        outputs["comparison.csv" if len(episodes) == 2 else "episode.csv"] = _csv_bytes(rows)
        outputs["trajectory-xy.svg"] = _svg_trajectory(
            episodes, labels, [case["comparison_validation"]["status"] if case["comparison_validation"] else "single replay"],
            start=start, end=request.end_time_s, sync_mode=request.sync_mode,
        )
        outputs.update(_telemetry_exports(episodes, labels, start, request.end_time_s, request.sync_mode))
        outputs["annotations.json"] = json.dumps({
            "notes": case["notes"], "bookmarks": case["bookmarks"], "tags": case["tags"],
        }, ensure_ascii=False, indent=2).encode("utf-8")
        rasterized_charts = _saved_chart_pngs(case_id)
        outputs["rasterized-chart-index.json"] = json.dumps({
            name: {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content), "dpi": 300}
            for name, content in sorted(rasterized_charts.items())
        }, indent=2).encode("utf-8")
    if not outputs:
        raise ApplicationError("EXPORT_FAILED", "No scientific export artifacts were produced.", 500)
    manifest = _manifest(case, bundles, outputs, request)
    outputs["manifest.json"] = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
    export_id = uuid4().hex
    root = _export_root(case_id)
    destination = root / export_id
    destination.mkdir()
    stored = {}
    for name, content in outputs.items():
        if not SAFE_NAME_RE.fullmatch(name):
            raise ApplicationError("INVALID_EXPORT_NAME", "Generated export filename is not safe.", 500)
        _atomic_bytes(destination / name, content)
        stored[name] = content
    if request.kind == "package_zip":
        zip_buffer = io.BytesIO()
        rasterized_charts = _saved_chart_pngs(case_id)
        with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, content in stored.items():
                archive.writestr(name, content)
            for name, content in rasterized_charts.items():
                archive.writestr(f"charts_png/{name}", content)
        zip_name = "case-package.zip"
        zip_content = zip_buffer.getvalue()
        _atomic_bytes(destination / zip_name, zip_content)
        stored[zip_name] = zip_content
    export_manifest = {
        "export_id": export_id,
        "case_id": case_id,
        "kind": request.kind,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": [{"name": name, "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content),
                   "download_url": f"/api/cases/{case_id}/exports/{export_id}/{name}"}
                  for name, content in stored.items()],
    }
    _atomic_bytes(destination / "export.json", json.dumps(export_manifest, indent=2).encode("utf-8"))
    return export_manifest


def save_scene_capture(case_library: CaseLibrary, case_id: str, request: CaptureRequest) -> dict:
    case, _bundles = case_library.load_for_export(case_id)
    reference = next((item for item in case["replay_references"] if item["replay_id"] == request.replay_id), None)
    if reference is None or reference["sha256"] != request.replay_sha256:
        raise ApplicationError("CAPTURE_REPLAY_MISMATCH", "Scene capture replay does not belong to this unchanged case.", 409)
    try:
        content = base64.b64decode(request.png_base64, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise ApplicationError("INVALID_CAPTURE", "Trusted-client capture must contain valid base64 PNG data.", 422) from exc
    _checked_png(content, request.width, request.height)
    export_id = uuid4().hex
    destination = _export_root(case_id) / export_id
    destination.mkdir()
    filename = "battlefield-scene.png"
    _atomic_bytes(destination / filename, content)
    metadata = {
        "export_id": export_id, "case_id": case_id, "kind": "trusted_webgl_scene_capture",
        "replay_id": request.replay_id, "replay_sha256": request.replay_sha256,
        "time_s": request.time_s, "camera": request.camera, "projection": request.projection,
        "width": request.width, "height": request.height, "artifact_type": "visualization-only screenshot",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": [{"name": filename, "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content),
                   "download_url": f"/api/cases/{case_id}/exports/{export_id}/{filename}"}],
    }
    _atomic_bytes(destination / "export.json", json.dumps(metadata, indent=2).encode("utf-8"))
    return metadata


def save_chart_capture(
    case_id: str,
    export_id: str,
    request: ChartCaptureRequest,
) -> dict:
    svg_path = export_file_path(case_id, export_id, request.source_filename)
    try:
        svg_content = svg_path.read_bytes()
    except OSError as exc:
        raise ApplicationError("EXPORT_READ_FAILED", "Could not read the source SVG for rasterization.", 500) from exc
    if hashlib.sha256(svg_content).hexdigest() != request.source_sha256:
        raise ApplicationError("EXPORT_HASH_MISMATCH", "Source SVG changed before PNG rasterization.", 409)
    try:
        root = ET.fromstring(svg_content)
        view_box = [float(value) for value in root.attrib["viewBox"].replace(",", " ").split()]
    except (KeyError, ValueError, ET.ParseError) as exc:
        raise ApplicationError("INVALID_SOURCE_SVG", "Source chart does not contain a valid SVG viewBox.", 422) from exc
    if len(view_box) != 4 or not all(math.isfinite(value) for value in view_box) or view_box[2] <= 0 or view_box[3] <= 0:
        raise ApplicationError("INVALID_SOURCE_SVG", "Source chart viewBox must have positive finite dimensions.", 422)
    expected_height = round(request.width * view_box[3] / view_box[2])
    if request.height != expected_height:
        raise ApplicationError("INVALID_CAPTURE", "PNG dimensions must preserve the source chart aspect ratio at 7 inches × 300 DPI.", 422)
    try:
        content = base64.b64decode(request.png_base64, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise ApplicationError("INVALID_CAPTURE", "Rasterized chart must contain valid base64 PNG data.", 422) from exc
    content = _png_with_dpi(_checked_png(content, request.width, request.height), request.dpi)
    png_name = Path(request.source_filename).stem + ".png"
    if not SAFE_NAME_RE.fullmatch(png_name):
        raise ApplicationError("INVALID_EXPORT_NAME", "Generated chart filename is not safe.", 500)
    export_path = export_file_path(case_id, export_id, request.source_filename)
    destination = export_path.parent
    _atomic_bytes(destination / png_name, content)
    metadata_path = export_file_path(case_id, export_id, "export.json")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        scientific_manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ApplicationError("INVALID_EXPORT_MANIFEST", "Export metadata is missing or invalid.", 409) from exc
    file_record = {
        "name": png_name,
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
        "download_url": f"/api/cases/{case_id}/exports/{export_id}/{png_name}",
        "kind": "derived_svg_rasterization",
        "source_svg": request.source_filename,
        "source_sha256": request.source_sha256,
        "dpi": request.dpi,
        "width": request.width,
        "height": request.height,
    }
    existing = next((item for item in metadata.get("files", []) if item.get("name") == png_name), None)
    if existing:
        metadata["files"].remove(existing)
    metadata.setdefault("files", []).append(file_record)
    scientific_manifest.setdefault("files", {})[png_name] = file_record["sha256"]
    scientific_manifest.setdefault("rasterized_charts", {})[png_name] = {
        "source_svg": request.source_filename,
        "dpi": request.dpi,
        "width": request.width,
        "height": request.height,
    }
    _atomic_bytes(destination / "manifest.json", json.dumps(scientific_manifest, ensure_ascii=False, indent=2).encode("utf-8"))
    _atomic_bytes(metadata_path, json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8"))
    return metadata


def list_exports(case_id: str) -> list[dict]:
    root = _export_root(case_id)
    result = []
    for directory in sorted(root.iterdir(), key=lambda path: path.name, reverse=True):
        if not directory.is_dir() or directory.is_symlink() or not EXPORT_ID_RE.fullmatch(directory.name):
            continue
        try:
            manifest = json.loads((directory / "export.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("case_id") == case_id and manifest.get("export_id") == directory.name:
            result.append(manifest)
    return result


def export_file_path(case_id: str, export_id: str, filename: str) -> Path:
    if not EXPORT_ID_RE.fullmatch(export_id) or not SAFE_NAME_RE.fullmatch(filename):
        raise ApplicationError("INVALID_EXPORT_ID", "Export identifiers and filenames must be safe opaque values.", 400)
    root = _export_root(case_id)
    directory = root / export_id
    if directory.is_symlink() or not directory.resolve().is_relative_to(root):
        raise ApplicationError("INVALID_EXPORT_PATH", "Export file escaped its local case directory.", 403)
    path = directory / filename
    if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()) or not path.is_file():
        raise ApplicationError("EXPORT_NOT_FOUND", "Requested export artifact was not found.", 404)
    return path
