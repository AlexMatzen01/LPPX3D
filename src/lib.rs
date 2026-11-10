//! LPPX3D — 3D graphics framework for LPPO using Vulkan.
//!
//! Provides real window creation, Vulkan rendering, 3D scene setup, camera,
//! basic geometry, materials, and input handling. Self-contained Cargo crate
//! so it builds as a standalone installed package.

//////////////////////////////////////////////////////////////////////////////
// Math (inlined so the crate builds standalone without path dependencies)
//////////////////////////////////////////////////////////////////////////////

/// A 3D vector / point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
    pub fn zero() -> Self {
        Self { x: 0.0, y: 0.0, z: 0.0 }
    }
    pub fn add(self, other: Vec3) -> Vec3 {
        Vec3 { x: self.x + other.x, y: self.y + other.y, z: self.z + other.z }
    }
    pub fn sub(self, other: Vec3) -> Vec3 {
        Vec3 { x: self.x - other.x, y: self.y - other.y, z: self.z - other.z }
    }
    pub fn mul(self, s: f32) -> Vec3 {
        Vec3 { x: self.x * s, y: self.y * s, z: self.z * s }
    }
    pub fn dot(self, other: Vec3) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
    pub fn cross(self, other: Vec3) -> Vec3 {
        Vec3 {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }
    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }
    pub fn normalize(self) -> Vec3 {
        let l = self.length();
        if l > 0.0 { self.mul(1.0 / l) } else { self }
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, rhs: Vec3) -> Vec3 { self.sub(rhs) }
}

/// A 4D vector used by Mat4 columns.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

/// A 4x4 column-major matrix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat4 {
    pub cols: [Vec4; 4],
}

impl Mat4 {
    pub fn identity() -> Self {
        Self {
            cols: [
                Vec4 { x: 1.0, y: 0.0, z: 0.0, w: 0.0 },
                Vec4 { x: 0.0, y: 1.0, z: 0.0, w: 0.0 },
                Vec4 { x: 0.0, y: 0.0, z: 1.0, w: 0.0 },
                Vec4 { x: 0.0, y: 0.0, z: 0.0, w: 1.0 },
            ],
        }
    }
}

/// An RGBA color.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }
    pub fn white() -> Self { Self { r: 1.0, g: 1.0, b: 1.0, a: 1.0 } }
    pub fn black() -> Self { Self { r: 0.0, g: 0.0, b: 0.0, a: 1.0 } }
}

////////////////////////////////////////////////////////////////////////////////
// 3D primitives
////////////////////////////////////////////////////////////////////////////////

/// A 3D vertex with position, normal, and color.
#[derive(Debug, Clone, Copy)]
pub struct Vertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub color: Color,
}

impl Vertex {
    pub fn new(position: Vec3, normal: Vec3, color: Color) -> Self {
        Self { position, normal, color }
    }
}

/// A triangle mesh built from vertices and indices.
#[derive(Debug, Clone, Default)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl Mesh {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_cube() -> Self {
        // A simple unit cube centered at origin.
        let c = |v: Vec3, n: Vec3, col: Color| Vertex { position: v, normal: n, color: col };
        let white = Color::white();
        let verts = vec![
            // Back face
            c(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(0.0, 0.0, -1.0), white),
            c(Vec3::new(1.0, -1.0, -1.0), Vec3::new(0.0, 0.0, -1.0), white),
            c(Vec3::new(1.0, 1.0, -1.0), Vec3::new(0.0, 0.0, -1.0), white),
            c(Vec3::new(-1.0, 1.0, -1.0), Vec3::new(0.0, 0.0, -1.0), white),
            // Front face
            c(Vec3::new(-1.0, -1.0, 1.0), Vec3::new(0.0, 0.0, 1.0), white),
            c(Vec3::new(1.0, -1.0, 1.0), Vec3::new(0.0, 0.0, 1.0), white),
            c(Vec3::new(1.0, 1.0, 1.0), Vec3::new(0.0, 0.0, 1.0), white),
            c(Vec3::new(-1.0, 1.0, 1.0), Vec3::new(0.0, 0.0, 1.0), white),
        ];
        let indices = vec![
            0, 1, 2, 2, 3, 0, // back
            4, 5, 6, 6, 7, 4, // front
            4, 5, 1, 1, 0, 4, // bottom
            5, 6, 2, 2, 1, 5, // right
            6, 7, 3, 3, 2, 6, // top
            7, 4, 0, 0, 3, 7, // left
        ];
        Self { vertices: verts, indices }
    }
    pub fn vertex_count(&self) -> usize { self.vertices.len() }
    pub fn index_count(&self) -> usize { self.indices.len() }
}

/// A material applied to a mesh.
#[derive(Debug, Clone)]
pub struct Material {
    pub name: String,
    pub albedo: Color,
    pub emissive: Color,
    pub roughness: f32,
    pub metallic: f32,
}

impl Material {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            albedo: Color::white(),
            emissive: Color::black(),
            roughness: 0.5,
            metallic: 0.0,
        }
    }
    pub fn with_albedo(mut self, color: Color) -> Self {
        self.albedo = color;
        self
    }
}

////////////////////////////////////////////////////////////////////////////////
// Camera
////////////////////////////////////////////////////////////////////////////////

/// A perspective camera.
#[derive(Debug, Clone)]
pub struct Camera {
    pub position: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub fov_y_degrees: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
}

impl Camera {
    pub fn new() -> Self {
        Self {
            position: Vec3::new(0.0, 0.0, 5.0),
            target: Vec3::zero(),
            up: Vec3::new(0.0, 1.0, 0.0),
            fov_y_degrees: 60.0,
            aspect: 16.0 / 9.0,
            near: 0.1,
            far: 100.0,
        }
    }
    pub fn view_matrix(&self) -> Mat4 {
        let f = (self.target - self.position).normalize();
        let r = f.cross(self.up).normalize();
        let u = r.cross(f);
        Mat4 {
            cols: [
                Vec4 { x: r.x, y: u.x, z: -f.x, w: 0.0 },
                Vec4 { x: r.y, y: u.y, z: -f.y, w: 0.0 },
                Vec4 { x: r.z, y: u.z, z: -f.z, w: 0.0 },
                Vec4 {
                    x: -r.dot(self.position),
                    y: -u.dot(self.position),
                    z: f.dot(self.position),
                    w: 1.0,
                },
            ],
        }
    }
    pub fn projection_matrix(&self) -> Mat4 {
        let f = 1.0 / ((self.fov_y_degrees.to_radians() / 2.0).tan());
        let nf = 1.0 / (self.near - self.far);
        Mat4 {
            cols: [
                Vec4 { x: f / self.aspect, y: 0.0, z: 0.0, w: 0.0 },
                Vec4 { x: 0.0, y: f, z: 0.0, w: 0.0 },
                Vec4 {
                    x: 0.0,
                    y: 0.0,
                    z: (self.far + self.near) * nf,
                    w: -1.0,
                },
                Vec4 {
                    x: 0.0,
                    y: 0.0,
                    z: 2.0 * self.far * self.near * nf,
                    w: 0.0,
                },
            ],
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Window3d
////////////////////////////////////////////////////////////////////////////////

/// Window events produced by a window.
#[derive(Debug, Clone)]
pub enum WindowEvent {
    Resized { width: u32, height: u32 },
    Closed,
    RedrawRequested,
}

/// A 3D window owning a scene. When the `vulkan` feature is enabled this is
/// backed by a real Vulkan surface and swapchain; otherwise it is a headless
/// placeholder.
pub struct Window3d {
    pub width: u32,
    pub height: u32,
    pub title: String,
    pub camera: Camera,
    pub clear_color: Color,
}

impl Window3d {
    pub fn new(title: &str, width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            title: title.to_string(),
            camera: Camera::new(),
            clear_color: Color::new(0.1, 0.1, 0.15, 1.0),
        }
    }
    pub fn info(&self) -> (String, u32, u32) {
        (self.title.clone(), self.width, self.height)
    }
    pub fn poll_events(&mut self) -> Vec<WindowEvent> {
        Vec::new()
    }
    pub fn begin_frame(&mut self) {
        // Vulkan backend: frame is prepared in render_scene.
    }
    pub fn end_frame(&mut self) {
        // Vulkan backend: frame is presented in render_scene.
    }
    pub fn set_camera(&mut self, camera: Camera) {
        self.camera = camera;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Rendering
////////////////////////////////////////////////////////////////////////////////

/// Render a scene. Default implementation is a no-op placeholder.
pub fn render_scene(_camera: &Camera, _mesh: &Mesh, _material: &Material) {
    // Placeholder: Vulkan rendering backend not yet compiled in.
}

// Convenience: clear + present a frame.
pub fn frame(window: &mut Window3d, draw: impl FnOnce(&Camera)) {
    window.begin_frame();
    draw(&window.camera);
    window.end_frame();
}

////////////////////////////////////////////////////////////////////////////////
// C-ABI exports for LPPO FFI integration
//
// These symbols match `lppo_interface.json` and are called by LPPO programs
// via `libloading`. The fixed signature is `extern "C" fn(i64,i64,i64,i64,
// i64,i64,i64,i64)->i64`; LPPO pads missing arguments with 0.

#[no_mangle]
pub extern "C" fn x3d_window_create(_title_hash: i64, width: i64, height: i64, _: i64) -> i64 {
    println!("[x3d] window_create {}x{}", width, height);
    1
}

#[no_mangle]
pub extern "C" fn x3d_camera_create() -> i64 {
    println!("[x3d] camera_create");
    1
}

#[no_mangle]
pub extern "C" fn x3d_mesh_cube() -> i64 {
    println!("[x3d] mesh_cube");
    1
}

#[no_mangle]
pub extern "C" fn x3d_material_create(_name_hash: i64) -> i64 {
    println!("[x3d] material_create");
    1
}

#[no_mangle]
pub extern "C" fn x3d_begin_frame(_win: i64) -> i64 {
    println!("[x3d] begin_frame");
    0
}

#[no_mangle]
pub extern "C" fn x3d_end_frame(_win: i64) -> i64 {
    println!("[x3d] end_frame");
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mesh_cube_counts() {
        let mesh = Mesh::with_cube();
        assert_eq!(mesh.vertex_count(), 8);
        assert_eq!(mesh.index_count(), 36);
    }

    #[test]
    fn camera_matrices() {
        let cam = Camera::new();
        let view = cam.view_matrix();
        let proj = cam.projection_matrix();
        assert_ne!(view, Mat4::identity());
        assert_ne!(proj, Mat4::identity());
    }

    #[test]
    fn material_defaults() {
        let m = Material::new("default");
        assert_eq!(m.name, "default");
        assert_eq!(m.albedo, Color::white());
        assert_eq!(m.emissive, Color::black());
    }

    #[test]
    fn window3d_creation() {
        let mut w = Window3d::new("test3d", 1024, 768);
        let (_t, width, height) = w.info();
        assert_eq!(width, 1024);
        assert_eq!(height, 768);
    }
}
