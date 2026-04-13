import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import ForceGraph3D from 'react-force-graph-3d';
import * as THREE from 'three';
import { getEdgeStyle } from '../utils/edgeStyle';
import { getNodeRadius, getGlowMultiplier } from '../utils/nodeSize';
import { transformToForceGraph } from '../utils/graphTransforms';

export default function GraphCanvas({ graphJson, onNodeHover, onNodeSelect, onMouseMove, panelWidth = 0, isNodeSelected = false }) {
  const graphRef = useRef(null);
  const [dimensions, setDimensions] = useState({
    width: window.innerWidth,
    height: window.innerHeight,
  });

  // Track user interaction to pause auto-orbit
  const isInteracting = useRef(false);
  const isNodeSelectedRef = useRef(isNodeSelected);
  const interactionTimer = useRef(null);

  // Keep ref in sync with prop (no stale closure in orbit loop)
  useEffect(() => {
    isNodeSelectedRef.current = isNodeSelected;
    // When panel closes, clear manual interaction lock so orbit resumes
    if (!isNodeSelected) {
      isInteracting.current = false;
      clearTimeout(interactionTimer.current);
    }
  }, [isNodeSelected]);
  const orbitRef = useRef(null);
  const orbitAngle = useRef(0);

  // Convert backend JSON → force-graph format
  const graphData = useMemo(() => transformToForceGraph(graphJson), [graphJson]);

  // Resize handler
  useEffect(() => {
    const onResize = () =>
      setDimensions({ width: window.innerWidth, height: window.innerHeight });
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  // Mouse position forwarding
  useEffect(() => {
    const onMove = (e) => onMouseMove?.({ x: e.clientX, y: e.clientY });
    window.addEventListener('mousemove', onMove);
    return () => window.removeEventListener('mousemove', onMove);
  }, [onMouseMove]);

  // Scene setup + camera animation (runs once graph is mounted)
  useEffect(() => {
    const graph = graphRef.current;
    if (!graph) return;

    // ── Lighting ────────────────────────────────────────────────────────────
    const scene = graph.scene();

    const ambient = new THREE.AmbientLight(0x1a2040, 1.2);
    scene.add(ambient);

    const dir1 = new THREE.DirectionalLight(0x4466cc, 1.0);
    dir1.position.set(200, 200, 100);
    scene.add(dir1);

    const dir2 = new THREE.DirectionalLight(0x221133, 0.6);
    dir2.position.set(-200, -100, -200);
    scene.add(dir2);

    // ── Starfield ───────────────────────────────────────────────────────────
    const STAR_COUNT = 4000;
    const starPositions = new Float32Array(STAR_COUNT * 3);
    const starColors = new Float32Array(STAR_COUNT * 3);
    for (let i = 0; i < STAR_COUNT; i++) {
      const r = 1200 + Math.random() * 800;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      starPositions[i * 3]     = r * Math.sin(phi) * Math.cos(theta);
      starPositions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      starPositions[i * 3 + 2] = r * Math.cos(phi);
      // Vary star color slightly: white / blue-white / warm
      const hue = Math.random();
      starColors[i * 3]     = 0.7 + hue * 0.3;
      starColors[i * 3 + 1] = 0.7 + hue * 0.25;
      starColors[i * 3 + 2] = 0.85 + hue * 0.15;
    }
    const starGeo = new THREE.BufferGeometry();
    starGeo.setAttribute('position', new THREE.BufferAttribute(starPositions, 3));
    starGeo.setAttribute('color', new THREE.BufferAttribute(starColors, 3));
    const starMat = new THREE.PointsMaterial({
      size: 1.2,
      vertexColors: true,
      transparent: true,
      opacity: 0.7,
      sizeAttenuation: true,
    });
    scene.add(new THREE.Points(starGeo, starMat));

    // ── Initial camera position ─────────────────────────────────────────────
    graph.cameraPosition({ x: 0, y: 60, z: 220 }, { x: 0, y: 0, z: 0 }, 1200);

    // ── Slow auto-orbit ────────────────────────────────────────────────────
    const ORBIT_R = 220;
    const ORBIT_Y = 45;
    let lastTime = Date.now();

    const orbit = () => {
      if (!isInteracting.current && !isNodeSelectedRef.current) {
        const now = Date.now();
        const dt = (now - lastTime) / 1000;
        lastTime = now;
        orbitAngle.current += dt * 0.08; // radians per second
        const a = orbitAngle.current;
        graph.cameraPosition({
          x: ORBIT_R * Math.sin(a),
          y: ORBIT_Y,
          z: ORBIT_R * Math.cos(a),
        });
      } else {
        lastTime = Date.now();
      }
      orbitRef.current = requestAnimationFrame(orbit);
    };
    // Start orbit after initial zoom completes
    const startTimer = setTimeout(() => {
      orbitRef.current = requestAnimationFrame(orbit);
    }, 1800);

    return () => {
      clearTimeout(startTimer);
      if (orbitRef.current) cancelAnimationFrame(orbitRef.current);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [graphRef.current]);

  // Pause orbit on user interaction
  const pauseOrbit = useCallback(() => {
    isInteracting.current = true;
    clearTimeout(interactionTimer.current);
    interactionTimer.current = setTimeout(() => {
      isInteracting.current = false;
    }, 4000);
  }, []);

  // ── Custom node Three.js object ────────────────────────────────────────────
  const nodeThreeObject = useCallback((node) => {
    const radius = getNodeRadius(node.__size);
    const glowMult = getGlowMultiplier(node.__isMain);
    const color = node.color;
    const isMain = node.__isMain;
    const group = new THREE.Group();

    // Core sphere
    const coreMesh = new THREE.Mesh(
      new THREE.SphereGeometry(radius, 32, 32),
      new THREE.MeshPhongMaterial({
        color,
        emissive: color,
        emissiveIntensity: isMain ? 0.65 : 0.40,
        shininess: 120,
        specular: 0xffffff,
      })
    );
    group.add(coreMesh);

    // Glow halo (backside sphere, transparent)
    const haloMesh = new THREE.Mesh(
      new THREE.SphereGeometry(radius * glowMult, 20, 20),
      new THREE.MeshPhongMaterial({
        color,
        transparent: true,
        opacity: isMain ? 0.14 : 0.09,
        side: THREE.BackSide,
        depthWrite: false,
      })
    );
    group.add(haloMesh);

    // Main claim gets two rings for emphasis
    if (isMain) {
      const ringMat = new THREE.MeshPhongMaterial({
        color,
        emissive: color,
        emissiveIntensity: 0.7,
        transparent: true,
        opacity: 0.55,
      });
      const ring1 = new THREE.Mesh(
        new THREE.TorusGeometry(radius * 2.0, radius * 0.18, 8, 64),
        ringMat
      );
      group.add(ring1);

      const ring2 = new THREE.Mesh(
        new THREE.TorusGeometry(radius * 3.0, radius * 0.07, 8, 64),
        new THREE.MeshPhongMaterial({
          color,
          transparent: true,
          opacity: 0.22,
          depthWrite: false,
        })
      );
      group.add(ring2);
    }

    // Store offset for pulsing animation
    node.__pulseOffset = Math.random() * Math.PI * 2;
    node.__coreMesh = coreMesh;
    node.__haloMesh = haloMesh;

    return group;
  }, []);

  // ── Pulse animation per frame ──────────────────────────────────────────────
  const handleRenderFrame = useCallback(() => {
    const t = Date.now() / 1000;
    graphData.nodes.forEach((node) => {
      if (node.__coreMesh) {
        const base = node.__isMain ? 0.55 : 0.32;
        const amp  = node.__isMain ? 0.22 : 0.12;
        node.__coreMesh.material.emissiveIntensity =
          base + amp * Math.sin(t * 1.4 + (node.__pulseOffset ?? 0));
      }
      if (node.__haloMesh) {
        const base = node.__isMain ? 0.12 : 0.07;
        const amp  = node.__isMain ? 0.06 : 0.04;
        node.__haloMesh.material.opacity =
          base + amp * Math.sin(t * 1.4 + (node.__pulseOffset ?? 0));
      }
    });
  }, [graphData.nodes]);

  // ── Edge particle config ───────────────────────────────────────────────────
  const linkParticleCount = useCallback(
    (link) => getEdgeStyle(link.edge_type).particleCount,
    []
  );
  const linkParticleSpeed = useCallback(
    (link) => getEdgeStyle(link.edge_type).particleSpeed,
    []
  );

  // ── Hover handler ──────────────────────────────────────────────────────────
  const handleNodeHover = useCallback(
    (node) => {
      document.body.style.cursor = node ? 'pointer' : 'default';
      onNodeHover?.(node ?? null);
      pauseOrbit();
    },
    [onNodeHover, pauseOrbit]
  );

  // ── Click handler ──────────────────────────────────────────────────────────
  const handleNodeClick = useCallback(
    (node) => {
      onNodeSelect?.(node);
      pauseOrbit();
      // Fly camera to node
      if (graphRef.current) {
        const dist = 80;
        const { x = 0, y = 0, z = 0 } = node;
        graphRef.current.cameraPosition(
          { x: x + dist * 0.4, y: y + dist * 0.3, z: z + dist },
          { x, y, z },
          600
        );
      }
    },
    [onNodeSelect, pauseOrbit]
  );

  const handleBackgroundClick = useCallback(() => {
    onNodeSelect?.(null);
  }, [onNodeSelect]);

  return (
    <div
      style={{ position: 'absolute', inset: 0 }}
      onMouseDown={pauseOrbit}
      onWheel={pauseOrbit}
    >
      <ForceGraph3D
        ref={graphRef}
        graphData={graphData}
        width={dimensions.width - panelWidth}
        height={dimensions.height}
        backgroundColor="#060a10"
        // Nodes
        nodeId="id"
        nodeVal="val"
        nodeColor="color"
        nodeThreeObject={nodeThreeObject}
        nodeThreeObjectExtend={false}
        nodeLabel={() => ''}
        // Links
        linkColor={(link) => link.color}
        linkWidth={(link) => link.width}
        linkOpacity={0.75}
        linkDirectionalParticles={linkParticleCount}
        linkDirectionalParticleWidth={(link) => Math.max(link.width * 0.55, 0.8)}
        linkDirectionalParticleColor={(link) => link.color}
        linkDirectionalParticleSpeed={linkParticleSpeed}
        linkDirectionalArrowLength={5}
        linkDirectionalArrowRelPos={0.92}
        linkDirectionalArrowColor={(link) => link.color}
        // Interaction
        onNodeHover={handleNodeHover}
        onNodeClick={handleNodeClick}
        onBackgroundClick={handleBackgroundClick}
        // Physics
        d3AlphaDecay={0.022}
        d3VelocityDecay={0.35}
        cooldownTicks={120}
        // Animation
        onRenderFramePre={handleRenderFrame}
        // Controls
        enableNodeDrag={true}
        enableNavigationControls={true}
        showNavInfo={false}
      />
    </div>
  );
}
