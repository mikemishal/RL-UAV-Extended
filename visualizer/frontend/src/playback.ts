import type { EntityState, ReplayEpisode, Vec3 } from './api';

export type ThreeVector = [number, number, number];

export interface DisplaySample {
  timeS: number;
  lowerIndex: number;
  upperIndex: number;
  fraction: number;
  defender: ThreeVector;
  attacker: ThreeVector;
  asset: ThreeVector;
  defenderVelocity: ThreeVector | null;
  attackerVelocity: ThreeVector | null;
  assetVelocity: ThreeVector | null;
  dynamicObstacles: Map<string, ThreeVector>;
}

export interface Bounds3 {
  min: ThreeVector;
  max: ThreeVector;
  center: ThreeVector;
  extent: number;
}

function negateAxis(value: number): number {
  return value === 0 ? 0 : -value;
}

export function simToThree(position: Vec3): ThreeVector {
  return [position[0], position[2], negateAxis(position[1])];
}

export function simVectorToThree(vector: Vec3): ThreeVector {
  return [vector[0], vector[2], negateAxis(vector[1])];
}

export function interpolateVector(from: Vec3, to: Vec3, fraction: number): ThreeVector {
  const t = Math.max(0, Math.min(1, fraction));
  return [
    from[0] + (to[0] - from[0]) * t,
    from[1] + (to[1] - from[1]) * t,
    from[2] + (to[2] - from[2]) * t,
  ];
}

function interpolatedPosition(from: EntityState, to: EntityState, fraction: number): ThreeVector {
  return simToThree(interpolateVector(from.position_m, to.position_m, fraction));
}

function interpolatedVelocity(from: EntityState, to: EntityState, fraction: number): ThreeVector | null {
  if (!from.velocity_mps || !to.velocity_mps) return null;
  return simVectorToThree(interpolateVector(from.velocity_mps, to.velocity_mps, fraction));
}

export function sampleAtTime(episode: ReplayEpisode, timeS: number): DisplaySample {
  const frames = episode.frames;
  const clampedTime = Math.max(0, Math.min(episode.summary.duration_s, timeS));
  let low = 0;
  let high = frames.length - 1;
  while (low < high) {
    const middle = Math.ceil((low + high) / 2);
    if (frames[middle].time_s <= clampedTime) low = middle;
    else high = middle - 1;
  }
  const lowerIndex = low;
  const upperIndex = Math.min(lowerIndex + 1, frames.length - 1);
  const lower = frames[lowerIndex];
  const upper = frames[upperIndex];
  const span = upper.time_s - lower.time_s;
  const fraction = span > 0 ? Math.max(0, Math.min(1, (clampedTime - lower.time_s) / span)) : 0;
  const dynamicObstacles = new Map<string, ThreeVector>();
  for (const obstacle of lower.moving_obstacles) {
    const next = upper.moving_obstacles.find(candidate => candidate.obstacle_id === obstacle.obstacle_id);
    if (!next) {
      dynamicObstacles.set(obstacle.obstacle_id, simToThree(obstacle.position_m));
      continue;
    }
    dynamicObstacles.set(
      obstacle.obstacle_id,
      simToThree(interpolateVector(obstacle.position_m, next.position_m, fraction)),
    );
  }
  return {
    timeS: clampedTime,
    lowerIndex,
    upperIndex,
    fraction,
    defender: interpolatedPosition(lower.defender, upper.defender, fraction),
    attacker: interpolatedPosition(lower.attacker, upper.attacker, fraction),
    asset: interpolatedPosition(lower.asset, upper.asset, fraction),
    defenderVelocity: interpolatedVelocity(lower.defender, upper.defender, fraction),
    attackerVelocity: interpolatedVelocity(lower.attacker, upper.attacker, fraction),
    assetVelocity: interpolatedVelocity(lower.asset, upper.asset, fraction),
    dynamicObstacles,
  };
}

export function staticObstacleBounds(minCorner: Vec3, maxCorner: Vec3): Bounds3 {
  const min: ThreeVector = [minCorner[0], minCorner[2], negateAxis(maxCorner[1])];
  const max: ThreeVector = [maxCorner[0], maxCorner[2], negateAxis(minCorner[1])];
  const center: ThreeVector = [
    (min[0] + max[0]) / 2,
    (min[1] + max[1]) / 2,
    (min[2] + max[2]) / 2,
  ];
  return {
    min,
    max,
    center,
    extent: Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2]),
  };
}

export function scenarioBounds(episode: ReplayEpisode): Bounds3 {
  const points: ThreeVector[] = [];
  for (const frame of episode.frames) {
    points.push(
      simToThree(frame.defender.position_m),
      simToThree(frame.attacker.position_m),
      simToThree(frame.asset.position_m),
    );
    for (const obstacle of frame.moving_obstacles) {
      const point = simToThree(obstacle.position_m);
      points.push(
        [point[0] - obstacle.radius_m, point[1] - obstacle.radius_m, point[2] - obstacle.radius_m],
        [point[0] + obstacle.radius_m, point[1] + obstacle.radius_m, point[2] + obstacle.radius_m],
      );
    }
  }
  for (const obstacle of episode.static_obstacles) {
    const bounds = staticObstacleBounds(obstacle.min_corner_m, obstacle.max_corner_m);
    points.push(bounds.min, bounds.max);
  }
  const min: ThreeVector = [Infinity, Infinity, Infinity];
  const max: ThreeVector = [-Infinity, -Infinity, -Infinity];
  for (const point of points) {
    for (let axis = 0; axis < 3; axis += 1) {
      min[axis] = Math.min(min[axis], point[axis]);
      max[axis] = Math.max(max[axis], point[axis]);
    }
  }
  const center: ThreeVector = [
    (min[0] + max[0]) / 2,
    (min[1] + max[1]) / 2,
    (min[2] + max[2]) / 2,
  ];
  return {
    min,
    max,
    center,
    extent: Math.max(1, max[0] - min[0], max[1] - min[1], max[2] - min[2]),
  };
}

export function orientationYaw(entity: EntityState, velocity: ThreeVector | null, fallback: ThreeVector): {
  yaw: number;
  source: 'recorded heading' | 'recorded movement' | 'display orientation derived from recorded positions' | 'unavailable';
} {
  if (entity.horizontal_heading_deg !== null && entity.horizontal_heading_deg !== undefined) {
    const heading = entity.horizontal_heading_deg * Math.PI / 180;
    return { yaw: Math.atan2(Math.cos(heading), -Math.sin(heading)), source: 'recorded heading' };
  }
  const direction = velocity
    ?? (entity.movement_direction ? simVectorToThree(entity.movement_direction) : null)
    ?? fallback;
  if (!direction || (Math.abs(direction[0]) < 1e-9 && Math.abs(direction[2]) < 1e-9)) {
    return { yaw: 0, source: 'unavailable' };
  }
  return {
    yaw: Math.atan2(direction[0], direction[2]),
    source: velocity || entity.movement_direction ? 'recorded movement' : 'display orientation derived from recorded positions',
  };
}

export function trajectoryPath(episode: ReplayEpisode, entity: 'defender' | 'attacker' | 'asset'): ThreeVector[] {
  return episode.frames.map(frame => simToThree(frame[entity].position_m));
}

export interface DynamicObstaclePath {
  startStep: number;
  points: ThreeVector[];
}

export function dynamicObstaclePaths(episode: ReplayEpisode): Map<string, DynamicObstaclePath[]> {
  const paths = new Map<string, DynamicObstaclePath[]>();
  for (const frame of episode.frames) {
    for (const obstacle of frame.moving_obstacles) {
      const segments = paths.get(obstacle.obstacle_id) ?? [];
      const segment = segments[segments.length - 1];
      if (!segment || segment.startStep + segment.points.length !== frame.step) {
        segments.push({ startStep: frame.step, points: [] });
      }
      segments[segments.length - 1].points.push(simToThree(obstacle.position_m));
      paths.set(obstacle.obstacle_id, segments);
    }
  }
  return paths;
}
