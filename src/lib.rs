pub mod a_funcs;
pub mod storage;
pub mod helpers;
pub mod initializer;
pub mod layers;
pub mod loss_funcs;
pub mod network;
pub mod optimizer;
pub mod serde;
pub mod trainer;

#[allow(non_camel_case_types)]
pub type f32s = packed_simd::f32x4;
/// A mask to go with our simd type of choice
#[allow(non_camel_case_types)]
pub type mask_s = packed_simd::m32x4;
/// Unsigned integer size of one simd lane
#[allow(non_camel_case_types)]
pub type usize_s = u32;
