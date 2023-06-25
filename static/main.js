import * as THREE from "three";

import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

import { UVsDebug } from "three/addons/utils/UVsDebug.js";
// import { UVsDebug } from "node_modules/three/examples/jsm/utils/UVsDebug.js";
// static\node_modules\three\examples\jsm\utils
// const scene = new THREE.Scene();
// const camera = new THREE.PerspectiveCamera(
//   75,
//   window.innerWidth / window.innerHeight,
//   0.1,
//   1000
// );

// const renderer = new THREE.WebGLRenderer();
// renderer.setSize(window.innerWidth, window.innerHeight);
// document.body.appendChild(renderer.domElement);

// const controls = new OrbitControls(camera, renderer.domElement);
// const loader = new GLTFLoader();
const geometry = new THREE.BoxGeometry(1, 1, 1);
const material = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
const cube = new THREE.Mesh(geometry, material);
// scene.add(cube);

// camera.position.z = 5;

// camera.position.z = 5;

// function animate() {
//   requestAnimationFrame(animate);

//   cube.rotation.x += 0.01;
//   cube.rotation.y += 0.01;

//   renderer.render(scene, camera);
// }

// animate();

function test(name, geometry) {
  const d = document.createElement("div");

  d.innerHTML = "<h3>" + name + "</h3>";

  d.appendChild(UVsDebug(geometry));

  document.body.appendChild(d);
}

test(
  "new THREE.TorusGeometry( 50, 20, 8, 8 )",
  new THREE.TorusGeometry(50, 20, 8, 8)
);

test("new THREE.BoxGeometry( 50, 20, 8, 8 )", geometry);
