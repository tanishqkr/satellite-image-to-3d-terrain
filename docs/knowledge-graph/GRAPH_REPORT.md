# Graph Report - DepthWizard  (2026-09-10)

## Summary
- 373 nodes · 590 edges · 23 communities (12 shown, 4 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 7 edges (avg confidence: 0.9)

## Community Hubs (Navigation)
- Frontend React UI & Geospatial Analysis Tools
- Backend Elevation Calibration, Geospatial Parser & Validation
- GAMUS Deep Learning Training & Losses
- FPV & TPP Drone Flight Simulator, Proximity Ray-Marching & HUD
- Frontend Build System & Dev Dependencies
- FastAPI REST Endpoints & GeoTIFF Export Services
- Inference Engine, Benchmark Validation & Configuration
- Voxel Terrain Quantization & Instanced Mesh Rendering
- TypeScript Compiler Configuration
- Presentation Diagram Generation
- SIH 2026 Presentation Builder
- Frontend Runtime Dependencies & 3D Engine
- CUDA Hardware Verification Suite
- Synthetic Aerial Benchmark Scene Generator
- Turbo Colormap Palette Shaders
- Backend Production Server Launcher

## God Nodes (most connected - your core abstractions)
1. `react` - 24 edges
2. `compilerOptions` - 16 edges
3. `calibrate_depth()` - 11 edges
4. `lucide-react` - 10 edges
5. `DepthWizard System Overview README.md` - 10 edges
6. `DepthEstimator` - 9 edges
7. `read_image()` - 9 edges
8. `build_mesh_data()` - 9 edges
9. `CombinedLoss` - 9 edges
10. `generateTurboPalette()` - 9 edges

## Surprising Connections (you probably didn't know these)
- `DepthWizard System Overview README.md` --references--> `UVP: FPV & TPP Drone Flight Simulator with Proximity Ray-Marching & Telemetry`  [EXTRACTED]
  README.md → reports/drone_and_navigation_mechanics_report.md
- `DepthWizard System Overview README.md` --references--> `UVP: Voxelized Digital Surface Model (DSM) Quantization & Instanced Rendering`  [EXTRACTED]
  README.md → reports/mesh_construction_and_project_architecture.md
- `DepthWizard Multi-Modal Geospatial Brief` --references--> `DepthWizard System Overview README.md`  [EXTRACTED]
  reports/project_brief.md → README.md
- `SIH26175 Verification Walkthrough` --references--> `DepthWizard System Overview README.md`  [EXTRACTED]
  WALKTHROUGH.md → README.md
- `test_cpu_autocast_guard()` --calls--> `DepthEstimator`  [EXTRACTED]
  backend/tests/test_pipeline.py → backend/app/services/depth_estimator.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Disaster Management & Topography UVP Suite** — uvp_flood_simulator, uvp_cross_section, uvp_contour_lines, uvp_geotiff_export [INFERRED 0.95]
- **FPV/TPP Drone Flight Simulation & Voxel Terrain Suite** — uvp_fpv_tpp_drone_flight, uvp_voxel_terrain_mode, uvp_terrain_quick_panel [INFERRED 0.95]

## Communities (23 total, 4 thin omitted)

### Community 0 - "Frontend React UI & Geospatial Analysis Tools"
Cohesion: 0.06
Nodes (41): App(), ColorBar(), ColorBarProps, ContourOverlay(), ContourOverlayProps, CrossSection(), CrossSectionProps, ProfilePoint (+33 more)

### Community 1 - "Backend Elevation Calibration, Geospatial Parser & Validation"
Cohesion: 0.07
Nodes (39): _process_pipeline(), Synchronous pipeline executed in threadpool to prevent event-loop blocking., calibrate_depth(), CalibrationResult, _metric_calibration(), ndarray, Two-Component Elevation Decomposition. DSM(x,y) = DTM_base(x,y) + α ·…, Keep as relative depth normalized to [0, 1]. (+31 more)

### Community 2 - "GAMUS Deep Learning Training & Losses"
Cohesion: 0.06
Nodes (25): GAMUSDataset, Dataset, PyTorch Dataset wrapping HuggingFace earthflow/GAMUS HDF5 files. Fixes applied:…, CombinedLoss, GradientMatchingLoss, Mathematically Stabilized Precision Loss for DepthWizard. Combines: 1.…, Log1p Scale-Invariant Logarithmic Loss. d = ln(1 + gamma * p) - ln(1 + gamma *…, StableSILogLoss (+17 more)

### Community 3 - "FPV & TPP Drone Flight Simulator, Proximity Ray-Marching & HUD"
Cohesion: 0.12
Nodes (30): Drone & Navigation Flight Mechanics Report, DroneCanvasProps, DroneFlightController(), DroneFlightControllerProps, FlightTelemetry, DroneHUD(), DroneHUDProps, DroneModel (+22 more)

### Community 4 - "Frontend Build System & Dev Dependencies"
Cohesion: 0.06
Nodes (34): devDependencies, tailwindcss, @tailwindcss/vite, @types/d3-contour, @types/d3-geo, @types/react, @types/react-dom, @types/three (+26 more)

### Community 5 - "FastAPI REST Endpoints & GeoTIFF Export Services"
Cohesion: 0.10
Nodes (25): export_dsm(), health(), get, Download computed DSM as GeoTIFF without Windows file lock race., set_estimator(), upload_image(), ErrorResponse, HealthResponse (+17 more)

### Community 6 - "Inference Engine, Benchmark Validation & Configuration"
Cohesion: 0.10
Nodes (17): compute_metrics(), ensure_val_tiles(), evaluate_benchmark(), ndarray, DepthWizard: Official GAMUS Validation Benchmark Script. Evaluates Base…, Downloads official val tiles from earthflow/GAMUS if not already on disk., Computes academic monocular depth estimation metrics on normalized elevation., download_tile_pair() (+9 more)

### Community 7 - "Voxel Terrain Quantization & Instanced Mesh Rendering"
Cohesion: 0.18
Nodes (14): 3D Mesh Construction & Project Architecture Report, VoxelMesh(), VoxelTerrain(), VoxelTerrainProps, generateTurboPalette(), rgbToHex(), sampleTurboRgb(), __dirname (+6 more)

### Community 8 - "TypeScript Compiler Configuration"
Cohesion: 0.11
Nodes (18): compilerOptions, allowImportingTsExtensions, isolatedModules, jsx, lib, module, moduleResolution, noEmit (+10 more)

### Community 9 - "Presentation Diagram Generation"
Cohesion: 0.17
Nodes (11): create_slide2_flowchart(), create_slide3_architecture(), create_slide4_swot_risks(), create_slide5_impact_ecosystem(), create_slide6_research_lineage(), Generate High-Resolution Diagrammatic Assets for DepthWizard SIH 2026…, Slide 4: Feasibility SWOT Matrix + Risk-Mitigation Chevrons., Slide 5: 4-Quadrant Impact Ecosystem Hub & Key Metrics. (+3 more)

### Community 10 - "SIH 2026 Presentation Builder"
Cohesion: 0.25
Nodes (10): add_card(), build_presentation(), find_shape_by_text(), format_bullet(), Build Official SIH 2026 Presentation for DepthWizard (SIH26175 - ISRO SAC).…, Safely removes a shape from a slide., Finds a shape containing specific text., Adds a styled rounded rectangular card container. (+2 more)

### Community 11 - "Frontend Runtime Dependencies & 3D Engine"
Cohesion: 0.20
Nodes (10): dependencies, d3-contour, d3-geo, lucide-react, react, react-dom, @react-three/drei, @react-three/fiber (+2 more)

## Knowledge Gaps
- **90 isolated node(s):** `name`, `private`, `version`, `type`, `dev` (+85 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 177 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `DepthWizard System Overview README.md` connect `GAMUS Deep Learning Training & Losses` to `Frontend React UI & Geospatial Analysis Tools`, `FPV & TPP Drone Flight Simulator, Proximity Ray-Marching & HUD`, `Voxel Terrain Quantization & Instanced Mesh Rendering`?**
  _High betweenness centrality (0.358) - this node is a cross-community bridge._
- **Why does `UVP: OGC Cloud-Optimized GeoTIFF Export` connect `GAMUS Deep Learning Training & Losses` to `FastAPI REST Endpoints & GeoTIFF Export Services`?**
  _High betweenness centrality (0.215) - this node is a cross-community bridge._
- **Why does `UVP: FPV & TPP Drone Flight Simulator with Proximity Ray-Marching & Telemetry` connect `FPV & TPP Drone Flight Simulator, Proximity Ray-Marching & HUD` to `GAMUS Deep Learning Training & Losses`?**
  _High betweenness centrality (0.147) - this node is a cross-community bridge._
- **What connects `name`, `private`, `version` to the rest of the system?**
  _90 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Frontend React UI & Geospatial Analysis Tools` be split into smaller, more focused modules?**
  _Cohesion score 0.06019871420222092 - nodes in this community are weakly interconnected._
- **Should `Backend Elevation Calibration, Geospatial Parser & Validation` be split into smaller, more focused modules?**
  _Cohesion score 0.07342995169082125 - nodes in this community are weakly interconnected._
- **Should `GAMUS Deep Learning Training & Losses` be split into smaller, more focused modules?**
  _Cohesion score 0.0613107822410148 - nodes in this community are weakly interconnected._