use crate::{f32s, mask_s, usize_s};
use rand::{rngs::SmallRng, seq::SliceRandom, SeedableRng};

use std::mem::{transmute, MaybeUninit};

///Shuffles an array of consequtive integers and allows yout to iterate through them
pub struct IndexShuffler {
    idxs: Box<[usize]>,
    idx: usize,
    rng: SmallRng,
}
impl IndexShuffler {
    pub fn new(size: usize) -> Self {
        let mut t = IndexShuffler {
            idxs: (0..size).collect(),
            idx: 0,
            rng: SmallRng::seed_from_u64(0),
        };
        t.reset();
        t
    }
    pub fn reset(&mut self) {
        self.idxs.shuffle(&mut self.rng);
        self.idx = 0;
    }
}
impl Iterator for IndexShuffler {
    type Item = usize;
    fn next(&mut self) -> Option<Self::Item> {
        let i = self.idxs.get(self.idx).map(|x| *x);
        self.idx += 1;
        i
    }
}

/// Overwrites the contents of a slice to zeros
pub fn zero(slice: &mut [f32]) {
    slice.into_iter().for_each(|f| *f = 0.);
}

pub fn zero_simd(slice: &mut [f32s]) {
    let zero = f32s::splat(0.);
    slice.into_iter().for_each(|f| *f = zero);
}

pub fn empty_slice() -> Box<[f32]> {
    vec![0f32; 0].into_boxed_slice()
}
pub fn empty_vec_simd() -> Vec<f32s> {
    Vec::new()
}
pub fn empty_vec() -> Vec<f32> {
    Vec::new()
}

pub fn splat_n(n: usize, val: f32) -> Vec<f32s> {
    std::iter::repeat(f32s::splat(val)).take(n).collect()
}

/// Compute the smallest amount of blocks of size `size` needed to fit n elements.
/// For example if we have 7 floats and we want to fit them into f32x4, we would need 2 f32x4s
pub fn least_size(n: usize, size: usize) -> usize {
    (n - 1) / size + 1
}

//should be safe because the simd is *packed*

#[inline(always)]
pub fn to_scalar(arr: &[f32s]) -> &[f32] {
    let ptr = arr.as_ptr() as *const f32;
    unsafe { std::slice::from_raw_parts(ptr, arr.len() * f32s::lanes()) }
}
#[inline(always)]
pub fn to_scalar_mut(arr: &mut [f32s]) -> &mut [f32] {
    let ptr = arr.as_mut_ptr() as *mut f32;
    unsafe { std::slice::from_raw_parts_mut(ptr, arr.len() * f32s::lanes()) }
}

pub fn simd_with<T>(mut iter: T) -> f32s
where
    T: ExactSizeIterator + Iterator<Item = f32>,
{
    let mut af_deriv: MaybeUninit<f32s> = MaybeUninit::uninit();
    unsafe {
        let ptr: *mut f32 = transmute(af_deriv.as_mut_ptr());
        for i in 0..f32s::lanes() {
            ptr.add(i).write(iter.next().unwrap());
        }
        return af_deriv.assume_init();
    }
}

pub fn simd_to_iter<'a>(simd: f32s) -> std::slice::Iter<'a, f32> {
    let slice = [simd];
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const f32, f32s::lanes()).into_iter() }
}

/// Returns a mask with the first n lanes set to true
pub fn mask(n: usize) -> mask_s {
    assert!(n <= mask_s::lanes());
    let mut m = MaybeUninit::zeroed();
    unsafe {
        let arr =
            std::slice::from_raw_parts_mut(transmute::<_, &mut usize_s>(&mut m), mask_s::lanes());
        for i in arr.iter_mut().take(n) {
            *i = usize_s::MAX;
        }
        return m.assume_init();
    }
}

pub fn sum(arr: &[f32s], len: usize) -> f32 {
    assert!(arr.len() > 0);
    let mask = mask(len % f32s::lanes());
    let mut iter = arr.iter().rev();
    let mut val = mask.select(*iter.next().unwrap(), f32s::splat(0.));

    for i in iter {
        val += *i
    }
    val.sum()
}
