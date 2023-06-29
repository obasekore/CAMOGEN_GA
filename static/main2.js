import * as THREE from "three";
// Create the scene
const scene = new THREE.Scene();

// Create the camera
const camera = new THREE.PerspectiveCamera(
  75,
  window.innerWidth / window.innerHeight,
  0.1,
  1000
);
camera.position.z = 5;

// Create the renderer
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// Create the torus geometry
const torusGeometry = new THREE.TorusGeometry(1, 0.5, 32, 100);

// Create the material
const material = new THREE.MeshBasicMaterial({ color: 0x00ff00 });

// Create the mesh
const torusMesh = new THREE.Mesh(torusGeometry, material);
scene.add(torusMesh);

// Create a texture loader
const textureLoader = new THREE.TextureLoader();

// Load the baked UV map texture
textureLoader.load("uv_map_texture.png", function (texture) {
  // Set the texture to the material's map property
  material.map = texture;

  // Adjust the UV wrapping mode for seamless appearance
  material.map.wrapS = THREE.RepeatWrapping;
  material.map.wrapT = THREE.RepeatWrapping;

  // Set the UV scale to match the number of segments in the torus
  material.map.repeat.set(
    torusGeometry.parameters.radialSegments,
    torusGeometry.parameters.tubularSegments
  );

  // Update material's needsUpdate flag
  material.needsUpdate = true;

  // Render the scene
  renderer.render(scene, camera);
});

// Animation loop
function animate() {
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
}

// Start the animation loop
animate();
