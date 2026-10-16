import { Component, Suspense, useEffect, useRef, useState } from 'react';
import type { ComponentRef, ErrorInfo, ReactNode } from 'react';
import { Canvas, useThree } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';

class SceneErrorBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('Three.js scene failed to render', error, info);
  }

  render() {
    if (this.state.failed) {
      return <div className="scene-error" role="alert">3D scene unavailable. Enable WebGL and reload. See browser console for details.</div>;
    }
    return this.props.children;
  }
}

function WebGLStatus({ onLost }: { onLost: (lost: boolean) => void }) {
  const canvas = useThree(state => state.gl.domElement);
  useEffect(() => {
    const lost = () => onLost(true);
    const restored = () => onLost(false);
    canvas.addEventListener('webglcontextlost', lost);
    canvas.addEventListener('webglcontextrestored', restored);
    return () => {
      canvas.removeEventListener('webglcontextlost', lost);
      canvas.removeEventListener('webglcontextrestored', restored);
    };
  }, [canvas, onLost]);
  return null;
}

export function Battlefield() {
  const controls = useRef<ComponentRef<typeof OrbitControls>>(null);
  const [ready, setReady] = useState(false);
  const [failed, setFailed] = useState(false);

  return (
    <section className="battlefield panel" aria-label="Interactive 3D battlefield">
      <div className="viewer-toolbar">
        <div><span className="eyebrow">BATTLEFIELD</span><h2>Scene workspace</h2></div>
        <div className="camera-tools">
          <span className="camera-mode">Free orbit</span>
          <button disabled={!ready || failed} onClick={() => controls.current?.reset()}>Reset camera</button>
        </div>
      </div>
      <div className="viewport" data-testid="viewport" data-scene-ready={ready && !failed}>
        <SceneErrorBoundary>
          <Suspense fallback={<div className="scene-loading" role="status">Preparing 3D scene...</div>}>
            <Canvas
              frameloop="demand"
              camera={{ position: [76, 64, 76], fov: 48, near: 0.1, far: 500 }}
              dpr={[1, 2]}
              fallback={<div className="scene-error" role="alert">WebGL is unavailable. Enable hardware acceleration to view the scene.</div>}
              onCreated={({ gl }) => {
                setReady(true);
                gl.domElement.setAttribute('aria-label', 'Empty 3D battlefield with ground and coordinate grid');
              }}
            >
              <WebGLStatus onLost={setFailed} />
              <color attach="background" args={['#101b25']} />
              <ambientLight intensity={1.1} />
              <directionalLight position={[30, 60, 20]} intensity={2} />
              <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.025, 0]}>
                <planeGeometry args={[100, 100]} />
                <meshStandardMaterial color="#1c303b" roughness={1} />
              </mesh>
              <gridHelper args={[100, 20, '#5c96a4', '#35505e']} />
              <axesHelper args={[12]} />
              <OrbitControls
                ref={controls}
                makeDefault
                enableDamping={false}
                target={[0, 0, 0]}
                minDistance={12}
                maxDistance={230}
                maxPolarAngle={Math.PI / 2 - 0.02}
              />
            </Canvas>
          </Suspense>
        </SceneErrorBoundary>
        {failed && <div className="scene-error" role="alert">WebGL context lost. Reload to restore the scene.</div>}
        <div className="empty-state">
          <span className="empty-icon" aria-hidden="true">+</span>
          <h3>No episode loaded</h3>
          <p>This is an empty scene, not a simulated episode.<br />Real Python replay integration begins in Phase 2.</p>
        </div>
        <div className="viewport-caption">Horizontal projection view / simulation x,y mapped to scene X,-Z</div>
      </div>
      <div className="viewer-footer"><span>Drag to orbit / scroll to zoom / right-drag to pan</span><span>Grid: 5 m</span></div>
    </section>
  );
}
