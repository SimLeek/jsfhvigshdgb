#![allow(non_upper_case_globals)]
#![allow(non_snake_case)]

use vulkano::instance::{Instance, PhysicalDevicesIter};
use vulkano::instance::InstanceExtensions;
use vulkano::instance::PhysicalDevice;
use std::process::exit;
use vulkano::device::Device;
use vulkano::device::DeviceExtensions;
use vulkano::device::Features;

use std::ptr;
use std::mem;
use std::os::raw::c_void;
use std::ffi::CStr;
use std::sync::Arc;

extern crate glfw;

use self::glfw::Context;

extern crate gl;

use self::gl::types::*;

const SCR_WIDTH: u32 = 1280;
const SCR_HEIGHT: u32 = 720;

mod shader;

use shader::Shader;

use vulkano::buffer::BufferUsage;
use vulkano::buffer::CpuAccessibleBuffer;
use vulkano::pipeline::ComputePipeline;

/// Macro to get c strings from literals without runtime overhead
/// Literal must not contain any interior nul bytes!
macro_rules! c_str {
    ($literal:expr) => {
        CStr::from_bytes_with_nul_unchecked(concat!($literal, "\0").as_bytes())
    }
}

fn vulkan_compute(instance: &Arc<Instance>) {
    let physical = PhysicalDevice::enumerate(&instance).next().expect("no device available");
    for device in PhysicalDevice::enumerate(&instance) {
        println!("GPU: [{}] is available!", device.name());
    }
    for family in physical.queue_families() {
        println!("Found a queue family with {:?} queue(s)", family.queues_count());
    }
    let queue_family = physical.queue_families()
        .find(|&q| q.supports_graphics())
        .expect("couldn't find a graphical queue family");
    let (device, mut queues) = {
        Device::new(physical, &Features::none(), &DeviceExtensions {
            khr_storage_buffer_storage_class: true,
            ..DeviceExtensions::none()
        },
                    [(queue_family, 0.5)].iter().cloned()).expect("failed to create device")
    };
    let queue = queues.next().unwrap();

    let data_iter = 0..65536;
    let data_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false,
                                                     data_iter).expect("failed to create buffer");

    mod cs {
        vulkano_shaders::shader! {
        ty: "compute",
        path: "src/shaders/dense_conv_forward_2d.glsl"
    }
    }

    let shader = cs::Shader::load(device.clone())
        .expect("failed to create shader module");

    let compute_pipeline = Arc::new(ComputePipeline::new(device.clone(), &shader.main_entry_point(), &())
        .expect("failed to create compute pipeline"));
}


fn opengl_compute() {
    // modified from: https://github.com/bwasty/learn-opengl-rs/blob/master/src/_4_advanced_opengl/_8_advanced_glsl_ubo.rs

    // glfw: initialize and configure
    // ------------------------------
    let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();
    glfw.window_hint(glfw::WindowHint::ContextVersion(4, 5));
    glfw.window_hint(glfw::WindowHint::OpenGlProfile(glfw::OpenGlProfileHint::Core));
    #[cfg(target_os = "macos")]
        glfw.window_hint(glfw::WindowHint::OpenGlForwardCompat(true));

    // glfw window creation
    // --------------------
    let (mut window, events) = glfw.create_window(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", glfw::WindowMode::Windowed)
        .expect("Failed to create GLFW window");

    window.make_current();
    window.set_framebuffer_size_polling(true);
    window.set_cursor_pos_polling(true);
    window.set_scroll_polling(true);

    // gl: load all OpenGL function pointers
    // ---------------------------------------
    gl::load_with(|symbol| window.get_proc_address(symbol) as *const _);

    let (shaderCompute, cubeVBO, cubeVAO, uboMatrices) = unsafe {
        // configure global opengl state
        // -----------------------------
        gl::Enable(gl::DEPTH_TEST);

        // build and compile shaders
        // -------------------------
        let shaderCompute = Shader::compute("src/shaders/dense_conv_forward_2d.glsl");

        // set up vertex data (and buffer(s)) and configure vertex attributes
        // ------------------------------------------------------------------

        // cube VAO
        let (mut cubeVAO, mut cubeVBO) = (0, 0);
        gl::GenVertexArrays(1, &mut cubeVAO);
        gl::GenBuffers(1, &mut cubeVBO);
        gl::BindVertexArray(cubeVAO);
        gl::BindBuffer(gl::ARRAY_BUFFER, cubeVBO);
        /*gl::BufferData(gl::ARRAY_BUFFER,
                       (cubeVertices.len() * mem::size_of::<GLfloat>()) as GLsizeiptr,
                       &cubeVertices[0] as *const f32 as *const c_void,
                       gl::STATIC_DRAW);*/
        let stride = 3 * mem::size_of::<GLfloat>() as GLsizei;
        gl::EnableVertexAttribArray(0);
        gl::VertexAttribPointer(0, 3, gl::FLOAT, gl::FALSE, stride, ptr::null());

        // configure a uniform buffer object
        // ---------------------------------
        // first. We get the relevant block indices
        //let uniformBlockIndexRed = gl::GetUniformBlockIndex(shaderRed.ID, c_str!("Matrices").as_ptr());
        //let uniformBlockIndexGreen = gl::GetUniformBlockIndex(shaderGreen.ID, c_str!("Matrices").as_ptr());
        //let uniformBlockIndexBlue = gl::GetUniformBlockIndex(shaderBlue.ID, c_str!("Matrices").as_ptr());
        //let uniformBlockIndexYellow = gl::GetUniformBlockIndex(shaderYellow.ID, c_str!("Matrices").as_ptr());
        // then we link each shader's uniform block to this uniform binding point
        //gl::UniformBlockBinding(shaderRed.ID, uniformBlockIndexRed, 0);
        //gl::UniformBlockBinding(shaderGreen.ID, uniformBlockIndexGreen, 0);
        //gl::UniformBlockBinding(shaderBlue.ID, uniformBlockIndexBlue, 0);
        //gl::UniformBlockBinding(shaderYellow.ID, uniformBlockIndexYellow, 0);
        // Now actually create the buffer
        let mut uboMatrices = 0;
        gl::GenBuffers(1, &mut uboMatrices);
        gl::BindBuffer(gl::UNIFORM_BUFFER, uboMatrices);
        //gl::BufferData(gl::UNIFORM_BUFFER, 2 * mem::size_of::<Matrix4<f32>>() as isize, ptr::null(), gl::STATIC_DRAW);
        // define the range of the buffer that links to a uniform binding point
        //gl::BindBufferRange(gl::UNIFORM_BUFFER, 0, uboMatrices, 0, 2 * 2 * mem::size_of::<Matrix4<f32>>() as isize);

        // store the projection matrix (we only do this once now) (note: we're not using zoom anymore by changing the FoV)
        //let projection: Matrix4<f32> = perspective(Deg(45.0), SCR_WIDTH as f32 / SCR_HEIGHT as f32 , 0.1, 100.0);
        gl::BindBuffer(gl::UNIFORM_BUFFER, uboMatrices);
        //gl::BufferSubData(gl::UNIFORM_BUFFER, 0, mem::size_of::<Matrix4<f32>>() as isize, projection.as_ptr() as *const c_void);
        gl::BindBuffer(gl::UNIFORM_BUFFER, 0);

        (shaderCompute, cubeVBO, cubeVAO, uboMatrices)
    };

    // render loop
    // -----------
    while !window.should_close() {
        // per-frame time logic
        // --------------------
        let currentFrame = glfw.get_time() as f32;
        //deltaTime = currentFrame - lastFrame;
        //lastFrame = currentFrame;

        // events
        // -----
        //process_events(&events, &mut firstMouse, &mut lastX, &mut lastY, &mut camera);

        // input
        // -----
        //processInput(&mut window, deltaTime, &mut camera);

        // render
        // ------
        unsafe {
            gl::ClearColor(0.1, 0.1, 0.1, 1.0);
            gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);

            // set the view and projection matrix in the uniform block - we only have to do this once per loop iteration.
            //let view = camera.GetViewMatrix();
            gl::BindBuffer(gl::UNIFORM_BUFFER, uboMatrices);
            //let size = mem::size_of::<Matrix4<f32>>() as isize;
            //gl::BufferSubData(gl::UNIFORM_BUFFER, size, size, view.as_ptr() as *const c_void);
            gl::BindBuffer(gl::UNIFORM_BUFFER, 0);
        }

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        window.swap_buffers();
        glfw.poll_events();
    }

    // optional: de-allocate all resources once they've outlived their purpose:
    // ------------------------------------------------------------------------
    unsafe {
        gl::DeleteVertexArrays(1, &cubeVAO);
        gl::DeleteBuffers(1, &cubeVBO);
    }
}

fn main() {
    opengl_compute();

    let instance = Instance::new(None, &InstanceExtensions::none(), None)
        .expect("failed to create instance");
    if PhysicalDevice::enumerate(&instance).len() == 0 {
        println!("Vulkan is not supported on any GPUs on this system.");
        println!("Please install a Vulkan ready driver");
        println!("Disabling your internal graphics card may help, too (but may black-out laptop displays).");
        println!("Switching to OpenGL based GLSL.");
        opengl_compute();
    } else {
        vulkan_compute(&instance);
    }
}