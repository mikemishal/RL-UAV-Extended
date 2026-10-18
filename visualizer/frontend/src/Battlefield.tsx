import { Component, useEffect, useMemo, useRef, useState } from 'react';
import type { ErrorInfo, ReactNode, RefObject } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { Html, OrbitControls, OrthographicCamera as DreiOrthographicCamera } from '@react-three/drei';
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib';
import {
  BufferAttribute,
  BoxGeometry,
  BufferGeometry,
  Color,
  ConeGeometry,
  DoubleSide,
  Group,
  Line,
  LineBasicMaterial,
  Mesh,
  OrthographicCamera,
  PerspectiveCamera,
  Vector3,
} from 'three';
import type {
  ReplayEpisode,
  ReplayEvent,
  ReplayFrame,
  ReplayMovingObstacle,
  ReplayPolicyTelemetry,
  Vec3,
} from './api';
import type { DecisionInterval } from './diagnostics';
import {
  dynamicObstaclePaths,
  orientationYaw,
  sampleAtTime,
  simVectorToThree,
  scenarioBounds,
  simToThree,
  staticObstacleBounds,
  trajectoryPath,
} from './playback';
import type { Bounds3, DisplaySample, ThreeVector } from './playback';
import type { DiagnosticVisibility, InformationMode } from './PolicyDiagnostics';

export type CameraMode = 'free' | 'follow-defender' | 'follow-attacker' | 'top-down';

export interface TrailVisibility {
  defender: boolean;
  attacker: boolean;
  asset: boolean;
  obstacles: boolean;
}

export interface CameraPose {
  position: [number, number, number];
  quaternion: [number, number, number, number];
  up: [number, number, number];
  target: [number, number, number];
}

export interface BattlefieldProps {
  episode: ReplayEpisode | null;
  timeS: number;
  cameraMode: CameraMode;
  resetCameraToken: number;
  showFullTrajectory: boolean;
  trails: TrailVisibility;
  showDetectionVolume: boolean;
  wireframeObstacles: boolean;
  transparentObstacles: boolean;
  informationMode: InformationMode;
  diagnosticVisibility: DiagnosticVisibility;
  decision: DecisionInterval | null;
  onWebGlError: (message: string) => void;
  onResetCamera: () => void;
  panelId?: string;
  heading?: string;
  linkedCamera?: boolean;
  cameraPose?: CameraPose | null;
  cameraLeader?: boolean;
  onCameraPose?: (pose: CameraPose) => void;
  sharedBoundsEpisode?: ReplayEpisode | null;
}

interface ErrorBoundaryState {
  error: Error | null;
}

class SceneErrorBoundary extends Component<{ children: ReactNode; onError: (message: string) => void }, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, _info: ErrorInfo) {
    this.props.onError(error.message);
  }

  render() {
    if (this.state.error) return <div className="scene-error" role="alert">3D scene failed: {this.state.error.message}</div>;
    return this.props.children;
  }
}

function vector(value: ThreeVector): Vector3 {
  return new Vector3(value[0], value[1], value[2]);
}

const cameraForward = new Vector3();

function CameraProbe({
  controls,
  output,
}: {
  controls: RefObject<OrbitControlsImpl | null>;
  output: RefObject<HTMLDivElement | null>;
}) {
  const { camera } = useThree();
  useFrame(() => {
    if (!output.current) return;
    output.current.setAttribute(
      'data-camera-position',
      camera.position.toArray().map(value => Number(value.toFixed(4))).join(','),
    );
    output.current.setAttribute('data-camera-up', camera.up.toArray().map(value => Number(value.toFixed(4))).join(','));
    output.current.setAttribute(
      'data-camera-forward',
      cameraForward.set(0, 0, -1).applyQuaternion(camera.quaternion)
        .toArray().map(value => Number(value.toFixed(4))).join(','),
    );
    const target = controls.current?.target;
    if (target) {
      output.current.setAttribute(
        'data-camera-target',
        target.toArray().map(value => Number(value.toFixed(4))).join(','),
      );
    }
  });
  return null;
}

function LinkedCamera({
  controls,
  enabled,
  leader,
  pose,
  onPose,
}: {
  controls: RefObject<OrbitControlsImpl | null>;
  enabled: boolean;
  leader: boolean;
  pose: CameraPose | null;
  onPose?: (pose: CameraPose) => void;
}) {
  const { camera } = useThree();
  const applying = useRef(false);
  const initialized = useRef(false);
  useEffect(() => {
    if (!enabled) {
      initialized.current = false;
      return;
    }
    if (!pose || !controls.current) return;
    initialized.current = true;
    applying.current = true;
    camera.position.fromArray(pose.position);
    camera.quaternion.fromArray(pose.quaternion);
    camera.up.fromArray(pose.up);
    controls.current.target.fromArray(pose.target);
    controls.current.update();
    applying.current = false;
  }, [camera, controls, enabled, pose]);
  useEffect(() => {
    const orbit = controls.current;
    if (!enabled || !orbit || !onPose) return;
    const publish = () => {
      if (applying.current || !controls.current) return;
      initialized.current = true;
      onPose({
        position: camera.position.toArray() as CameraPose['position'],
        quaternion: camera.quaternion.toArray() as CameraPose['quaternion'],
        up: camera.up.toArray() as CameraPose['up'],
        target: controls.current.target.toArray() as CameraPose['target'],
      });
    };
    orbit.addEventListener('change', publish);
    if (leader && !pose && !initialized.current) publish();
    return () => orbit.removeEventListener('change', publish);
  }, [camera, controls, enabled, leader, onPose, pose]);
  return null;
}

function makeLine(points: ThreeVector[], color: string, opacity = 0.95): Line {
  const geometry = new BufferGeometry();
  const buffer = new Float32Array(Math.max(points.length + 1, 2) * 3);
  for (let i = 0; i < points.length; i += 1) {
    buffer.set(points[i], i * 3);
  }
  geometry.setAttribute('position', new BufferAttribute(buffer, 3));
  geometry.setDrawRange(0, 0);
  const material = new LineBasicMaterial({ color, transparent: opacity < 1, opacity });
  const line = new Line(geometry, material);
  line.frustumCulled = false;
  return line;
}

function useTrajectory(
  points: ThreeVector[],
  currentIndex: number,
  currentFraction: number,
  currentPosition: ThreeVector,
  show: boolean,
  full: boolean,
  color: string,
) {
  const line = useMemo(() => makeLine(points, color), [points, color]);
  const positionAttribute = line.geometry.getAttribute('position') as BufferAttribute;
  useEffect(() => () => {
    line.geometry.dispose();
    (line.material as LineBasicMaterial).dispose();
  }, [line]);
  useFrame(() => {
    if (!show) {
      line.visible = false;
      return;
    }
    line.visible = true;
    if (full) {
      line.geometry.setDrawRange(0, points.length);
      return;
    }
    if (currentFraction > 0 && currentIndex < points.length - 1) {
      positionAttribute.setXYZ(currentIndex + 1, ...currentPosition);
      positionAttribute.needsUpdate = true;
      line.geometry.setDrawRange(0, currentIndex + 2);
    } else {
      line.geometry.setDrawRange(0, currentIndex + 1);
    }
  });
  return line;
}

function Trajectory({
  episode,
  sample,
  visible,
  full,
  entity,
}: {
  episode: ReplayEpisode;
  sample: DisplaySample;
  visible: boolean;
  full: boolean;
  entity: 'defender' | 'attacker' | 'asset';
}) {
  const points = useMemo(() => trajectoryPath(episode, entity), [episode, entity]);
  return (
    <TrajectoryLine
      points={points}
      currentIndex={sample.lowerIndex}
      currentFraction={sample.fraction}
      currentPosition={sample[entity]}
      visible={visible}
      full={full}
      color={{ defender: '#65acff', attacker: '#fa777d', asset: '#65c998' }[entity]}
    />
  );
}

function TrajectoryLine({
  points,
  currentIndex,
  currentFraction,
  currentPosition,
  visible,
  full,
  color,
}: {
  points: ThreeVector[];
  currentIndex: number;
  currentFraction: number;
  currentPosition: ThreeVector;
  visible: boolean;
  full: boolean;
  color: string;
}) {
  const line = useTrajectory(points, currentIndex, currentFraction, currentPosition, visible, full, color);
  return <primitive object={line} />;
}

function DynamicObstacleTrails({
  episode,
  sample,
  visible,
  full,
}: {
  episode: ReplayEpisode;
  sample: DisplaySample;
  visible: boolean;
  full: boolean;
}) {
  const paths = useMemo(() => dynamicObstaclePaths(episode), [episode]);
  return (
    <>
      {[...paths].flatMap(([id, segments]) => segments.map(({ points, startStep }, segmentIndex) => {
        const currentStep = episode.frames[sample.lowerIndex].step;
        const localIndex = currentStep - startStep;
        if (points.length < 2 || (!full && localIndex < 0)) return null;
        const currentIndex = Math.min(points.length - 1, Math.max(0, localIndex));
        const currentFraction = localIndex >= 0 && localIndex < points.length - 1 ? sample.fraction : 0;
        return (
          <TrajectoryLine
            key={`${id}-${segmentIndex}`}
            points={points}
            currentIndex={currentIndex}
            currentFraction={currentFraction}
            currentPosition={sample.dynamicObstacles.get(id) ?? points[currentIndex]}
            visible={visible}
            full={full}
            color={id.toLowerCase().includes('bird') ? '#e9bc76' : '#bc8fe8'}
          />
        );
      }))}
    </>
  );
}

function DroneModel({ color }: { color: string }) {
  return (
    <group>
      <mesh>
        <boxGeometry args={[1.35, 0.28, 0.48]} />
        <meshStandardMaterial color={color} roughness={0.55} />
      </mesh>
      <mesh rotation={[0, Math.PI / 4, 0]}>
        <boxGeometry args={[1.75, 0.12, 0.12]} />
        <meshStandardMaterial color={color} roughness={0.6} />
      </mesh>
      <mesh rotation={[0, -Math.PI / 4, 0]}>
        <boxGeometry args={[1.75, 0.12, 0.12]} />
        <meshStandardMaterial color={color} roughness={0.6} />
      </mesh>
      {([-1, 1] as const).flatMap(x => ([-1, 1] as const).map(z => (
        <group key={`${x}-${z}`} position={[x * 0.63, 0.13, z * 0.63]}>
          <mesh>
            <cylinderGeometry args={[0.08, 0.08, 0.08, 8]} />
            <meshStandardMaterial color="#d3e0e8" />
          </mesh>
          <mesh position={[0, 0.055, 0]}>
            <boxGeometry args={[0.62, 0.035, 0.075]} />
            <meshStandardMaterial color="#c3d5df" />
          </mesh>
        </group>
      )))}
      <mesh position={[0, 0.04, 0.28]}>
        <coneGeometry args={[0.12, 0.3, 5]} />
        <meshStandardMaterial color="#d8e4ed" />
      </mesh>
    </group>
  );
}

function Vehicle({
  episode,
  sample,
  entityName,
  label,
  color,
  visible = true,
}: {
  episode: ReplayEpisode;
  sample: DisplaySample;
  entityName: 'defender' | 'attacker';
  label: string;
  color: string;
  visible?: boolean;
}) {
  const ref = useRef<Group>(null);
  const position = sample[entityName];
  const frame = episode.frames[sample.lowerIndex];
  const nextFrame = episode.frames[sample.upperIndex];
  const fallback = [
    nextFrame[entityName].position_m[0] - frame[entityName].position_m[0],
    nextFrame[entityName].position_m[2] - frame[entityName].position_m[2],
    -(nextFrame[entityName].position_m[1] - frame[entityName].position_m[1]),
  ] as ThreeVector;
  const orientation = orientationYaw(
    frame[entityName],
    sample[`${entityName}Velocity`],
    fallback,
  );
  useFrame(() => {
    if (!ref.current) return;
    ref.current.visible = visible;
    if (!visible) return;
    ref.current.position.set(...position);
    ref.current.rotation.set(0, orientation.yaw, 0);
  });
  if (!visible) return null;
  return (
    <group ref={ref} visible={visible} data-testid={`${entityName}-vehicle`}>
      <DroneModel color={color} />
      <Html
        position={[0, 1.25, 0]}
        center
        data-testid={entityName === 'attacker' ? 'attacker-truth-label' : undefined}
        style={{ color: '#e8f0f5', whiteSpace: 'nowrap', fontSize: '11px', textShadow: '0 1px 4px #000' }}
      >
        {label}{orientation.source === 'display orientation derived from recorded positions'
          ? ' · display heading derived from recorded positions'
          : orientation.source === 'unavailable'
            ? ' · orientation unavailable'
            : ''}
      </Html>
    </group>
  );
}

function AssetMarker({ position }: { position: ThreeVector }) {
  const ref = useRef<Group>(null);
  useFrame(() => { if (ref.current) ref.current.position.set(...position); });
  return (
    <group ref={ref} data-testid="asset-marker">
      <mesh position={[0, 0.7, 0]}>
        <cylinderGeometry args={[0.9, 1.05, 1.4, 6]} />
        <meshStandardMaterial color="#43aa7d" roughness={0.8} />
      </mesh>
      <mesh position={[0, 1.46, 0]}>
        <cylinderGeometry args={[0.52, 0.68, 0.18, 6]} />
        <meshStandardMaterial color="#aad7bd" />
      </mesh>
      <Html position={[0, 2.1, 0]} center style={{ color: '#d5f4df', whiteSpace: 'nowrap', fontSize: '10px', textShadow: '0 1px 4px #000' }}>
        Protected asset
      </Html>
    </group>
  );
}

function StaticObstacle({
  episode,
  wireframe,
  transparent,
  informationMode,
}: {
  episode: ReplayEpisode;
  wireframe: boolean;
  transparent: boolean;
  informationMode: InformationMode;
}) {
  return (
    <>
      {episode.static_obstacles.filter(obstacle => informationMode === 'ground-truth' || obstacle.map_known).map(obstacle => {
        const bounds = staticObstacleBounds(obstacle.min_corner_m, obstacle.max_corner_m);
        const size: ThreeVector = [
          bounds.max[0] - bounds.min[0],
          bounds.max[1] - bounds.min[1],
          bounds.max[2] - bounds.min[2],
        ];
        return (
          <StaticObstacleBox
            key={obstacle.obstacle_id}
            position={bounds.center}
            size={size}
            color={obstacle.visual_kind === 'tree' ? '#557e65' : '#8e795d'}
            wireframe={wireframe}
            transparent={transparent}
          />
        );
      })}
    </>
  );
}

function StaticObstacleBox({
  position,
  size,
  color,
  wireframe,
  transparent,
}: {
  position: ThreeVector;
  size: ThreeVector;
  color: string;
  wireframe: boolean;
  transparent: boolean;
}) {
  const outline = useMemo(() => new BoxGeometry(...size), [size[0], size[1], size[2]]);
  useEffect(() => () => outline.dispose(), [outline]);
  return (
    <group position={position} data-testid="static-obstacle">
      <mesh>
        <boxGeometry args={size} />
        <meshStandardMaterial
          color={color}
          wireframe={wireframe}
          transparent={transparent || wireframe}
          opacity={transparent ? 0.36 : wireframe ? 0.8 : 0.84}
          depthWrite={!transparent}
          roughness={0.88}
        />
      </mesh>
      <lineSegments>
        <edgesGeometry args={[outline]} />
        <lineBasicMaterial color="#d4c4a4" transparent opacity={0.55} />
      </lineSegments>
    </group>
  );
}

function MovingObstacles({
  episode,
  sample,
}: {
  episode: ReplayEpisode;
  sample: DisplaySample;
}) {
  const refs = useRef(new Map<string, Mesh>());
  const current = sample;
  const latest = episode.frames[sample.lowerIndex].moving_obstacles;
  useFrame(() => {
    for (const obstacle of latest) {
      const mesh = refs.current.get(obstacle.obstacle_id);
      const position = current.dynamicObstacles.get(obstacle.obstacle_id);
      if (!mesh || !position) continue;
      mesh.position.set(...position);
    }
  });
  return (
    <>
      {latest.map((obstacle: ReplayMovingObstacle) => (
        <mesh
          key={obstacle.obstacle_id}
          ref={mesh => {
            if (mesh) refs.current.set(obstacle.obstacle_id, mesh);
            else refs.current.delete(obstacle.obstacle_id);
          }}
          data-testid="moving-obstacle"
          scale={obstacle.radius_m}
        >
          <sphereGeometry args={[1, 14, 10]} />
          <meshStandardMaterial
            color={obstacle.kind === 'bird' ? '#e9bc76' : '#bc8fe8'}
            transparent
            opacity={0.84}
            roughness={0.5}
          />
        </mesh>
      ))}
    </>
  );
}

function DetectionVolume({
  frame,
  position,
  visible,
}: {
  frame: ReplayFrame;
  position: ThreeVector;
  visible: boolean;
}) {
  if (!visible || frame.detection_radius_m <= 0) return null;
  return (
    <mesh position={position} scale={frame.detection_radius_m} data-testid="detection-volume">
      <sphereGeometry args={[1, 24, 16]} />
      <meshStandardMaterial color="#86bdde" transparent opacity={0.08} wireframe depthWrite={false} />
    </mesh>
  );
}

const eventLabels: Record<ReplayEvent['kind'], string> = {
  detection: 'Attacker detected',
  interception: 'Safe interception',
  unsafe_interception: 'Unsafe interception',
  asset_reached: 'Attacker reached asset',
  collision: 'Recorded collision',
  timeout: 'Timeout',
};

const outcomeEventLabels: Record<ReplayEpisode['summary']['outcome'], string> = {
  intercepted: 'Safe interception',
  soldier_caught: 'Attacker reached protected asset',
  unsafe_intercept: 'Unsafe interception',
  timeout: 'Timeout',
  defender_static_obstacle_collision: 'Defender / static obstacle collision',
  defender_dynamic_obstacle_collision: 'Defender / moving obstacle collision',
  enemy_static_obstacle_collision: 'Attacker / static obstacle collision',
  enemy_dynamic_obstacle_collision: 'Attacker / moving obstacle collision',
};

function EpisodeEventMarkers({ episode, timeS }: { episode: ReplayEpisode; timeS: number }) {
  return (
    <>
      {episode.events.filter(event => event.time_s <= timeS).map((event, index) => {
        const entity = event.entity ? episode.frames[event.step]?.[event.entity] : null;
        const position = event.position_m ?? entity?.position_m;
        if (!position) return null;
        const world = simToThree(position);
        const terminal = event.kind !== 'detection';
        const label = event.kind === 'collision' ? outcomeEventLabels[episode.summary.outcome] : eventLabels[event.kind];
        return (
          <group key={`${event.kind}-${event.step}-${index}`} position={world} data-testid="episode-event">
            <mesh>
              <sphereGeometry args={[terminal ? 1.15 : 0.8, 12, 10]} />
              <meshBasicMaterial color={event.kind === 'detection' ? '#79d6d0' : '#ffd270'} wireframe />
            </mesh>
            <Html position={[0, 1.8, 0]} center style={{ color: '#f4eee0', whiteSpace: 'nowrap', fontSize: '10px', textShadow: '0 1px 5px #000' }}>
              {label} · {event.time_s.toFixed(1)} s
              {event.position_m ? '' : ' · entity position'}
            </Html>
          </group>
        );
      })}
    </>
  );
}

const normalizedUp = new Vector3(0, 1, 0);

function DirectionIndicator({
  origin,
  direction,
  color,
  visible,
  label,
}: {
  origin: ThreeVector;
  direction: Vec3 | null | undefined;
  color: string;
  visible: boolean;
  label: string;
}) {
  const ref = useRef<Group>(null);
  const vectorRef = useRef(new Vector3());
  useFrame(() => {
    const group = ref.current;
    if (!group) return;
    const converted = direction ? simVectorToThree(direction) : null;
    const vector = converted ? vectorRef.current.set(...converted) : null;
    const magnitude = vector?.length() ?? 0;
    group.visible = visible && magnitude > 1e-9;
    if (!group.visible || !vector) return;
    vector.normalize();
    group.position.set(...origin);
    group.quaternion.setFromUnitVectors(normalizedUp, vector);
  });
  return (
    <group ref={ref} userData={{ diagnosticLabel: label }}>
      <mesh position={[0, 6, 0]} renderOrder={20}>
        <cylinderGeometry args={[0.14, 0.14, 10, 8]} />
        <meshBasicMaterial color={color} depthTest={false} />
      </mesh>
      <mesh position={[0, 11, 0]} renderOrder={20}>
        <coneGeometry args={[0.55, 1.5, 8]} />
        <meshBasicMaterial color={color} depthTest={false} />
      </mesh>
    </group>
  );
}

function CorrectionCone({ origin, direction, visible }: { origin: ThreeVector; direction: Vec3 | null | undefined; visible: boolean }) {
  const ref = useRef<Group>(null);
  const vectorRef = useRef(new Vector3());
  const geometry = useMemo(() => new ConeGeometry(12 * Math.tan(Math.PI / 6), 12, 32, 1, true), []);
  useEffect(() => () => geometry.dispose(), [geometry]);
  useFrame(() => {
    const group = ref.current;
    if (!group) return;
    const converted = direction ? simVectorToThree(direction) : null;
    const vector = converted ? vectorRef.current.set(...converted) : null;
    const magnitude = vector?.length() ?? 0;
    group.visible = visible && magnitude > 1e-9;
    if (!group.visible || !vector) return;
    vector.normalize();
    group.position.set(
      origin[0] + vector.x * 6,
      origin[1] + vector.y * 6,
      origin[2] + vector.z * 6,
    );
    group.quaternion.setFromUnitVectors(normalizedUp, vector.negate());
  });
  return (
    <group ref={ref} userData={{ diagnosticLabel: '30 degree native 3D correction allowance' }}>
      <mesh geometry={geometry} renderOrder={10}>
        <meshBasicMaterial color="#ffad32" transparent opacity={0.28} side={DoubleSide} depthWrite={false} depthTest={false} />
      </mesh>
    </group>
  );
}

function EstimateMarker({ position, color, label, visible = true }: { position: Vec3 | null; color: string; label: string; visible?: boolean }) {
  const ref = useRef<Mesh>(null);
  useFrame(() => {
    const mesh = ref.current;
    if (!mesh || !visible || !position) {
      if (mesh) mesh.visible = false;
      return;
    }
    mesh.visible = true;
    mesh.position.set(...simToThree(position));
  });
  return (
    <mesh ref={ref} userData={{ diagnosticLabel: label }}>
      <octahedronGeometry args={[0.85, 0]} />
      <meshBasicMaterial color={color} wireframe />
    </mesh>
  );
}

function TrackingErrorLine({ from, to, visible }: { from: Vec3 | null; to: Vec3 | null; visible: boolean }) {
  const ref = useRef<Line>(null);
  useFrame(() => {
    const line = ref.current;
    if (!line) return;
    const available = visible && from !== null && to !== null;
    line.visible = Boolean(available);
    if (!available) return;
    const start = simToThree(from);
    const end = simToThree(to);
    const attribute = line.geometry.getAttribute('position') as BufferAttribute;
    attribute.setXYZ(0, ...start);
    attribute.setXYZ(1, ...end);
    attribute.needsUpdate = true;
    line.geometry.computeBoundingSphere();
  });
  const line = useMemo(() => {
    const geometry = new BufferGeometry();
    geometry.setAttribute('position', new BufferAttribute(new Float32Array(6), 3));
    const material = new LineBasicMaterial({ color: '#ff77bb', linewidth: 2 });
    const result = new Line(geometry, material);
    result.frustumCulled = false;
    return result;
  }, []);
  useEffect(() => () => {
    line.geometry.dispose();
    (line.material as LineBasicMaterial).dispose();
  }, [line]);
  return <primitive ref={ref} object={line} />;
}

function GuidanceOverlays({
  origin,
  telemetry,
  visibility,
}: {
  origin: ThreeVector;
  telemetry: ReplayPolicyTelemetry | null;
  visibility: DiagnosticVisibility;
}) {
  if (!telemetry) return null;
  const lead = telemetry.lrppo?.lead_direction;
  return (
    <>
      {telemetry.lrppo && (
        <>
          <DirectionIndicator origin={origin} direction={lead} color="#f59e0b" visible={visibility.lead} label="Analytical Lead direction" />
          <DirectionIndicator origin={origin} direction={telemetry.requested_action} color="#c084fc" visible={visibility.requested} label="Final LR-PPO requested direction" />
          <DirectionIndicator origin={origin} direction={telemetry.executed_velocity_mps} color="#22d3ee" visible={visibility.executed} label="Actual post-transition velocity direction" />
          <CorrectionCone origin={origin} direction={lead} visible={visibility.cone} />
        </>
      )}
      {visibility.safetyMarker && telemetry.avoidance_active && (
        <mesh position={[origin[0], origin[1] + 2.2, origin[2]]} userData={{ diagnosticLabel: 'Recorded safety avoidance active' }}>
          <octahedronGeometry args={[0.7, 0]} />
          <meshBasicMaterial color="#ff9f43" wireframe />
        </mesh>
      )}
    </>
  );
}

function sceneBounds(episode: ReplayEpisode, informationMode: InformationMode) {
  if (informationMode === 'ground-truth') return scenarioBounds(episode);
  const configuredExtent = episode.environment.config.L;
  const configuredAltitude = episode.environment.config.max_altitude;
  if (typeof configuredExtent !== 'number' || !Number.isFinite(configuredExtent) || configuredExtent <= 0) {
    throw new Error('Policy-Visible framing unavailable: replay does not record a valid engagement extent.');
  }
  if (typeof configuredAltitude !== 'number' || !Number.isFinite(configuredAltitude) || configuredAltitude <= 0) {
    throw new Error('Policy-Visible framing unavailable: replay does not record a valid maximum altitude.');
  }
  const extent = configuredExtent;
  const altitude = configuredAltitude;
  const max = Math.max(extent * 2, altitude);
  return {
    min: [-extent, 0, -extent] as ThreeVector,
    max: [extent, altitude, extent] as ThreeVector,
    center: [0, altitude / 2, 0] as ThreeVector,
    extent: max,
  };
}

function sharedSceneBounds(
  episode: ReplayEpisode,
  otherEpisode: ReplayEpisode | null | undefined,
  informationMode: InformationMode,
): Bounds3 {
  const first = sceneBounds(episode, informationMode);
  if (!otherEpisode) return first;
  const second = sceneBounds(otherEpisode, informationMode);
  const min: ThreeVector = first.min.map((value, index) => Math.min(value, second.min[index])) as ThreeVector;
  const max: ThreeVector = first.max.map((value, index) => Math.max(value, second.max[index])) as ThreeVector;
  return {
    min,
    max,
    center: min.map((value, index) => (value + max[index]) / 2) as ThreeVector,
    extent: Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2]),
  };
}

function PerspectiveCameraRig({
  episode,
  boundsEpisode,
  sample,
  mode,
  informationMode,
  resetToken,
  controls,
}: {
  episode: ReplayEpisode;
  boundsEpisode: ReplayEpisode | null;
  sample: DisplaySample;
  mode: Exclude<CameraMode, 'top-down'>;
  informationMode: InformationMode;
  resetToken: number;
  controls: RefObject<OrbitControlsImpl | null>;
}) {
  const lastFollowPosition = useRef<Vector3 | null>(null);
  const { camera, size } = useThree();
  const bounds = useMemo(() => sharedSceneBounds(episode, boundsEpisode, informationMode), [boundsEpisode, episode, informationMode]);
  const target = useMemo(() => vector(bounds.center), [bounds]);
  const sampleRef = useRef(sample);
  sampleRef.current = sample;

  useEffect(() => {
    const control = controls.current;
    if (!control) return;
    const aspect = Math.max(0.5, size.width / Math.max(1, size.height));
    const fitExtent = Math.max(bounds.extent, 18);
    const radius = Math.hypot(
      (bounds.max[0] - bounds.min[0]) / 2,
      (bounds.max[1] - bounds.min[1]) / 2,
      (bounds.max[2] - bounds.min[2]) / 2,
    );
    const verticalFov = Math.PI / 180 * (camera as PerspectiveCamera).fov;
    const horizontalFov = 2 * Math.atan(Math.tan(verticalFov / 2) * aspect);
    const distance = Math.max(fitExtent * 0.8, radius / Math.sin(Math.min(verticalFov, horizontalFov) / 2) * 1.12);
    const currentSample = sampleRef.current;
    const nextTarget = mode === 'follow-defender' ? vector(currentSample.defender)
      : mode === 'follow-attacker' ? vector(currentSample.attacker) : target.clone();
    control.target.copy(nextTarget);
    camera.position.set(nextTarget.x + distance * 0.76, nextTarget.y + distance * 0.67, nextTarget.z + distance * 0.76);
    control.minPolarAngle = 0.04;
    control.maxPolarAngle = Math.PI - 0.03;
    camera.lookAt(nextTarget);
    camera.updateProjectionMatrix();
    lastFollowPosition.current = mode === 'follow-defender' ? vector(currentSample.defender)
      : mode === 'follow-attacker' ? vector(currentSample.attacker) : null;
    control.update();
  }, [bounds, camera, mode, resetToken, size.height, size.width, target]);

  useFrame(() => {
    const tracked = mode === 'follow-defender' ? sample.defender
      : mode === 'follow-attacker' ? sample.attacker : null;
    const control = controls.current;
    if (!tracked || !control) {
      lastFollowPosition.current = null;
      return;
    }
    const next = vector(tracked);
    const previous = lastFollowPosition.current;
    if (!previous) {
      lastFollowPosition.current = next;
      return;
    }
    const step = next.clone().sub(previous);
    camera.position.add(step);
    control.target.add(step);
    lastFollowPosition.current = next;
    control.update();
  });

  return (
    <OrbitControls
      key={resetToken}
      ref={controls}
      makeDefault
      enableDamping
      dampingFactor={0.12}
      minDistance={4}
      maxDistance={Math.max(bounds.extent * 8, 100)}
    />
  );
}

function TopDownCameraRig({
  episode,
  boundsEpisode,
  resetToken,
  controls,
  informationMode,
}: {
  episode: ReplayEpisode;
  boundsEpisode: ReplayEpisode | null;
  resetToken: number;
  controls: RefObject<OrbitControlsImpl | null>;
  informationMode: InformationMode;
}) {
  const cameraRef = useRef<OrthographicCamera>(null);
  const { size } = useThree();
  const bounds = useMemo(() => sharedSceneBounds(episode, boundsEpisode, informationMode), [boundsEpisode, episode, informationMode]);
  const target = useMemo(() => vector(bounds.center), [bounds]);
  const aspect = Math.max(0.5, size.width / Math.max(1, size.height));
  const halfHeight = Math.max(bounds.max[2] - bounds.min[2], (bounds.max[0] - bounds.min[0]) / aspect, 10) * 0.58;
  const halfWidth = halfHeight * aspect;
  const cameraDistance = Math.max(bounds.extent * 2, 40);
  useEffect(() => {
    const camera = cameraRef.current;
    const control = controls.current;
    if (!camera || !control) return;
    camera.up.set(0, 0, -1);
    camera.position.set(target.x, target.y + cameraDistance, target.z);
    camera.left = -halfWidth;
    camera.right = halfWidth;
    camera.top = halfHeight;
    camera.bottom = -halfHeight;
    camera.updateProjectionMatrix();
    control.target.copy(target);
    control.minPolarAngle = Math.PI / 2;
    control.maxPolarAngle = Math.PI / 2;
    control.update();
  }, [cameraDistance, controls, halfHeight, halfWidth, resetToken, target]);
  return (
    <>
      <DreiOrthographicCamera
        key={resetToken}
        ref={cameraRef}
        makeDefault
        position={[target.x, target.y + cameraDistance, target.z]}
        left={-halfWidth}
        right={halfWidth}
        top={halfHeight}
        bottom={-halfHeight}
        near={0.1}
        far={cameraDistance * 4}
      />
      <OrbitControls
        key={`controls-${resetToken}`}
        ref={controls}
        makeDefault
        enableDamping
        dampingFactor={0.12}
        minDistance={2}
        maxDistance={cameraDistance * 3}
        enableRotate={false}
        minPolarAngle={Math.PI / 2}
        maxPolarAngle={Math.PI / 2}
      />
    </>
  );
}

function CameraRig(props: {
  episode: ReplayEpisode;
  boundsEpisode: ReplayEpisode | null;
  sample: DisplaySample;
  mode: CameraMode;
  informationMode: InformationMode;
  resetToken: number;
  controls: RefObject<OrbitControlsImpl | null>;
}) {
  if (props.mode === 'top-down') {
    return <TopDownCameraRig episode={props.episode} boundsEpisode={props.boundsEpisode} informationMode={props.informationMode} resetToken={props.resetToken} controls={props.controls} />;
  }
  const mode = props.informationMode === 'policy-visible' && props.mode === 'follow-attacker'
    ? 'free'
    : props.mode;
  return <PerspectiveCameraRig {...props} mode={mode} />;
}

function SceneContents({
  episode,
  timeS,
  cameraMode,
  resetCameraToken,
  showFullTrajectory,
  trails,
  showDetectionVolume,
  wireframeObstacles,
  transparentObstacles,
  informationMode,
  diagnosticVisibility,
  decision,
  sharedBoundsEpisode,
  controls,
  cameraOutput,
}: BattlefieldProps & {
  controls: RefObject<OrbitControlsImpl | null>;
  cameraOutput: RefObject<HTMLDivElement | null>;
}) {
  const sample = useMemo(() => sampleAtTime(episode!, timeS), [episode, timeS]);
  const frame = episode!.frames[sample.lowerIndex];
  const bounds = useMemo(() => sharedSceneBounds(episode!, sharedBoundsEpisode, informationMode), [episode, informationMode, sharedBoundsEpisode]);
  const policyVisible = decision?.telemetry.decision_policy_visible;
  const estimatePosition = policyVisible?.enemy_detected
    ? policyVisible.estimator === 'kalman' ? policyVisible.e_hat_m : policyVisible.enemy_measurement_m
    : null;
  const estimateVelocity = policyVisible?.enemy_detected
    ? policyVisible.estimator === 'kalman' ? policyVisible.v_hat_mps : policyVisible.enemy_measurement_velocity_mps
    : null;
  const gridSize = Math.max(40, Math.ceil(bounds.extent * 1.15 / 10) * 10);
  return (
    <>
      <color attach="background" args={['#111c26']} />
      <ambientLight intensity={1.1} />
      <directionalLight position={[90, 130, 60]} intensity={2.1} castShadow />
      <hemisphereLight args={['#c8d7e3', '#536652', 0.52]} />
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.055, 0]} receiveShadow data-testid="simulator-ground">
        <planeGeometry args={[gridSize * 2, gridSize * 2]} />
        <meshStandardMaterial color="#293b37" roughness={0.92} />
      </mesh>
      <gridHelper args={[gridSize * 2, 40, new Color('#759a92'), new Color('#405b58')]} position={[0, 0, 0]} />
      <axesHelper args={[Math.max(8, bounds.extent * 0.07)]} position={[0, 0.02, 0]} />
      <StaticObstacle episode={episode!} wireframe={wireframeObstacles} transparent={transparentObstacles} informationMode={informationMode} />
      {informationMode === 'ground-truth' && <MovingObstacles episode={episode!} sample={sample} />}
      <Trajectory episode={episode!} sample={sample} visible={trails.defender && informationMode === 'ground-truth'} full={showFullTrajectory} entity="defender" />
      {informationMode === 'ground-truth' && <Trajectory episode={episode!} sample={sample} visible={trails.attacker} full={showFullTrajectory} entity="attacker" />}
      <Trajectory episode={episode!} sample={sample} visible={trails.asset && informationMode === 'ground-truth'} full={showFullTrajectory} entity="asset" />
      {informationMode === 'ground-truth' && <DynamicObstacleTrails episode={episode!} sample={sample} visible={trails.obstacles} full={showFullTrajectory} />}
      <DetectionVolume frame={frame} position={sample.defender} visible={showDetectionVolume} />
      <Vehicle episode={episode!} sample={sample} entityName="defender" label="Defender UAV" color="#2786df" />
      {informationMode === 'ground-truth' &&       <Vehicle
        episode={episode!}
        sample={sample}
        entityName="attacker"
        label="Attacker UAV · privileged ground truth"
        color="#df4e59"
        visible={informationMode === 'ground-truth'}
      />}
      <AssetMarker position={sample.asset} />
      {informationMode === 'ground-truth' && <EpisodeEventMarkers episode={episode!} timeS={timeS} />}
      <GuidanceOverlays origin={sample.defender} telemetry={decision?.telemetry ?? null} visibility={diagnosticVisibility} />
      {decision && (
        <>
          <EstimateMarker
            position={estimatePosition ?? null}
            color="#d68cff"
            label="Policy-visible target estimate or measurement"
            visible={diagnosticVisibility.estimate}
          />
          <EstimateMarker
            position={informationMode === 'ground-truth' && diagnosticVisibility.truthMarker
              ? decision.startFrame.privileged_diagnostics.attacker_position_m
              : null}
            color="#fa777d"
            label="Privileged ground-truth attacker position"
          />
          <TrackingErrorLine
            from={decision.telemetry.decision_policy_visible.enemy_detected
              ? (decision.telemetry.decision_policy_visible.estimator === 'kalman'
                ? decision.telemetry.decision_policy_visible.e_hat_m
                : decision.telemetry.decision_policy_visible.enemy_measurement_m)
              : null}
            to={informationMode === 'ground-truth' ? decision.startFrame.privileged_diagnostics.attacker_position_m : null}
            visible={diagnosticVisibility.trackingError && decision.telemetry.decision_policy_visible.enemy_detected}
          />
          {estimatePosition && estimateVelocity && (
            <DirectionIndicator
              origin={simToThree(estimatePosition)}
              direction={estimateVelocity}
              color="#d68cff"
              visible={diagnosticVisibility.estimateVelocity}
              label="Recorded policy-visible target velocity estimate"
            />
          )}
        </>
      )}
      <CameraRig episode={episode!} boundsEpisode={sharedBoundsEpisode ?? null} sample={sample} mode={cameraMode} informationMode={informationMode} resetToken={resetCameraToken} controls={controls} />
      <CameraProbe controls={controls} output={cameraOutput} />
    </>
  );
}

function EmptyScene({
  resetToken,
  controls,
  cameraOutput,
}: {
  resetToken: number;
  controls: RefObject<OrbitControlsImpl | null>;
  cameraOutput: RefObject<HTMLDivElement | null>;
}) {
  const { camera } = useThree();
  useEffect(() => {
    const control = controls.current;
    if (!control) return;
    control.reset();
    camera.up.set(0, 1, 0);
    camera.position.set(65, 52, 70);
    control.target.set(0, 0, 0);
    camera.lookAt(0, 0, 0);
    control.update();
  }, [camera, controls, resetToken]);
  return (
    <>
      <color attach="background" args={['#111c26']} />
      <ambientLight intensity={1.2} />
      <directionalLight position={[8, 12, 9]} intensity={2} />
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.05, 0]}>
        <planeGeometry args={[120, 120]} />
        <meshStandardMaterial color="#293b37" />
      </mesh>
      <gridHelper args={[120, 40, '#759a92', '#405b58']} />
      <OrbitControls key={resetToken} ref={controls} makeDefault enableDamping={false} />
      <CameraProbe controls={controls} output={cameraOutput} />
    </>
  );
}

export function Battlefield(props: BattlefieldProps) {
  const ref = useRef<HTMLDivElement>(null);
  const controls = useRef<OrbitControlsImpl>(null);
  const cameraOutput = useRef<HTMLDivElement>(null);
  const contextLost = useRef(false);
  const [sceneEpoch, setSceneEpoch] = useState(0);
  const displayedSample = useMemo(
    () => props.episode ? sampleAtTime(props.episode, props.timeS) : null,
    [props.episode, props.timeS],
  );
  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const onLost = (event: Event) => {
      event.preventDefault();
      contextLost.current = true;
      props.onWebGlError('WebGL context was lost. Reload the page to retry the scene.');
    };
    const onRestored = () => {
      contextLost.current = false;
      setSceneEpoch(epoch => epoch + 1);
      props.onWebGlError('');
    };
    element.addEventListener('webglcontextlost', onLost, true);
    element.addEventListener('webglcontextrestored', onRestored, true);
    return () => {
      element.removeEventListener('webglcontextlost', onLost, true);
      element.removeEventListener('webglcontextrestored', onRestored, true);
    };
  }, [props.onWebGlError]);

  return (
    <section className="battlefield panel" aria-label={props.heading ?? '3D episode battlefield'}>
      <div className="viewer-toolbar">
        <div><span className="eyebrow">AUTHORITATIVE REPLAY · NATIVE 3D</span><h2>{props.heading ?? 'Interactive battlefield'}</h2></div>
        <div className="camera-tools">
          <span className="camera-mode">{props.cameraMode === 'top-down' ? 'Top-down projection of 3D replay' : `Camera: ${props.cameraMode.replaceAll('-', ' ')}`}</span>
          <button type="button" aria-label="Reset camera" onClick={() => {
            controls.current?.reset();
            props.onResetCamera();
          }}>Reset camera</button>
        </div>
      </div>
      <div className="viewport" ref={ref} data-testid="battlefield-viewport">
        <SceneErrorBoundary key={sceneEpoch} onError={message => {
          if (!contextLost.current) props.onWebGlError(message);
        }}>
          <Canvas
            dpr={[1, 1.75]}
            shadows
            camera={{ position: [65, 52, 70], fov: 48, near: 0.1, far: 10000 }}
            gl={{ antialias: true, alpha: false }}
            onCreated={({ gl }) => {
              gl.setClearColor('#111c26');
            }}
          >
            {props.episode
              ? <SceneContents {...props} controls={controls} cameraOutput={cameraOutput} />
              : <EmptyScene resetToken={props.resetCameraToken} controls={controls} cameraOutput={cameraOutput} />}
            {props.linkedCamera && (
              <LinkedCamera
                controls={controls}
                enabled={props.linkedCamera}
                leader={props.cameraLeader ?? false}
                pose={props.cameraPose ?? null}
                onPose={props.onCameraPose}
              />
            )}
          </Canvas>
        </SceneErrorBoundary>
        {!props.episode && (
          <div className="empty-state" data-testid="no-episode-overlay">
            <span className="empty-icon" aria-hidden="true">◎</span>
            <h3>No episode loaded</h3>
            <p>Select a recorded episode below to load the validated 3D trace.</p>
          </div>
        )}
        {props.episode && displayedSample && (
          <output
            className="sr-only"
            data-testid={props.panelId ? `rendered-replay-state-${props.panelId}` : 'rendered-replay-state'}
            data-panel-id={props.panelId}
            data-step={props.episode.frames[displayedSample.lowerIndex].step}
            data-upper-step={props.episode.frames[displayedSample.upperIndex].step}
            data-interpolated={String(displayedSample.fraction > 0)}
            data-static-obstacle-count={props.informationMode === 'ground-truth'
              ? props.episode.static_obstacles.length
              : props.episode.static_obstacles.filter(obstacle => obstacle.map_known).length}
            data-moving-obstacle-count={props.informationMode === 'ground-truth'
              ? props.episode.frames[displayedSample.lowerIndex].moving_obstacles.length
              : 0}
            data-visible-event-count={props.informationMode === 'ground-truth'
              ? props.episode.events.filter(event => event.time_s <= displayedSample.timeS).length
              : 0}
            data-defender-position={displayedSample.defender.join(',')}
            data-attacker-position={props.informationMode === 'ground-truth' ? displayedSample.attacker.join(',') : 'unavailable'}
            data-asset-position={displayedSample.asset.join(',')}
          />
        )}
        <div className="sr-only" data-testid={props.panelId ? `camera-state-${props.panelId}` : 'camera-state'} ref={cameraOutput} />
        {props.episode && (
          <span className="viewport-caption">
            {props.informationMode === 'ground-truth'
              ? 'Ground Truth researcher view · attacker position is privileged · 1 unit = 1 m · (x, y, z) → (x, z, −y)'
              : 'Policy-Visible partial reconstruction · no privileged attacker position or trajectory · (x, y, z) → (x, z, −y)'}
          </span>
        )}
        {props.episode?.defender.policy_id === 'lr_ppo' && props.decision?.telemetry.lrppo && (
          <div className="guidance-legend" data-testid="guidance-legend" aria-label="LR-PPO guidance direction legend">
            <span><i className="lead-vector" />Lead direction</span>
            <span><i className="requested-vector" />Final requested direction</span>
            <span><i className="executed-vector" />Actual post-transition velocity</span>
            <span><i className="correction-cone" />30° native 3D allowance</span>
            <small>Arrows are normalized direction indicators, not magnitude.</small>
          </div>
        )}
      </div>
    </section>
  );
}
