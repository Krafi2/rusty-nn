pub mod activation_functions;
pub mod allocator;
pub mod default_trainer;
pub mod helpers;
pub mod initializer;
pub mod layers;
pub mod loss_functions;
pub mod network;
pub mod optimizer;
pub mod trainer;

#[allow(non_camel_case_types)] // <- feck off
pub type f32s = packed_simd::f32x4;
/// A mask to go with our simd type of choice
#[allow(non_camel_case_types)]
pub type mask_s = packed_simd::m32x4;
/// Unsigned integer size of one simd lane
#[allow(non_camel_case_types)]
pub type usize_s = u32;
