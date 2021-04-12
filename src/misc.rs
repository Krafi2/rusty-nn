pub mod simd {
    use crate::{f32s, mask_s, usize_s};
    use std::mem::MaybeUninit;

    pub struct VectorAdapter<I> {
        iter: I,
    }

    impl<I> VectorAdapter<I>
    where
        I: Iterator<Item = f32>,
    {
        pub fn new(iter: I) -> Self {
            Self { iter }
        }
    }

    impl<I> Iterator for VectorAdapter<I>
    where
        I: Iterator<Item = f32>,
    {
        type Item = f32s;

        fn next(&mut self) -> Option<Self::Item> {
            let mut array = [0.; f32s::lanes()];
            for i in &mut array {
                if let Some(f) = self.iter.next() {
                    *i = f
                } else {
                    return None;
                }
            }
            Some(f32s::from_slice_unaligned(&array))
        }
    }

    pub fn splat_n(n: usize, val: f32) -> Vec<f32s> {
        std::iter::repeat(f32s::splat(val)).take(n).collect()
    }

    /// Compute the smallest amount of blocks of size `size` needed to fit n elements.
    /// For example if we have 7 floats and we want to fit them into f32x4, we would need 2 f32x4s
    pub fn to_blocks(n: usize, block: usize) -> usize {
        (n - 1) / block + 1
    }

    //should be safe because the simd is *packed*

    #[inline]
    pub fn as_scalar(arr: &[f32s]) -> &[f32] {
        let ptr = arr.as_ptr().cast::<f32>();
        unsafe { std::slice::from_raw_parts(ptr, arr.len() * f32s::lanes()) }
    }

    #[inline]
    pub fn as_scalar_mut(arr: &mut [f32s]) -> &mut [f32] {
        let ptr = arr.as_mut_ptr().cast::<f32>();
        unsafe { std::slice::from_raw_parts_mut(ptr, arr.len() * f32s::lanes()) }
    }

    #[inline]
    /// Converts a boxed slice of f32s into a boxed slice of f32s.
    pub fn into_scalar(arr: Box<[f32s]>) -> Box<[f32]> {
        // unsafe {
        //     let arr = ManuallyDrop::new(arr);
        //     // The alignment isn't the same, however we don't allocate or deallocate anything so we should be fine
        //     let arr = Vec::<f32>::from_raw_parts(arr.as_mut_ptr(), arr.len() * f32s::lanes(), arr.len() * f32s::lanes());
        //     arr.into_boxed_slice()
        // }
        as_scalar(&arr).to_owned().into_boxed_slice()
    }

    /// This trait provides the [as_scalar](self::as_scalar) and [as_scalar_mut](self::as_scalar_mut)
    /// in a more pleasant to use way.
    pub trait AsScalarExt<'a>
    where
        Self: 'a,
    {
        fn as_scalar(&'a self) -> &'a [f32]
        where
            Self: AsRef<[f32s]>;
        fn as_scalar_mut(&'a mut self) -> &'a mut [f32]
        where
            Self: AsMut<[f32s]>;
    }

    impl<'a, T: AsRef<[f32s]> + 'a> AsScalarExt<'a> for T {
        fn as_scalar(&'a self) -> &'a [f32]
        where
            Self: AsRef<[f32s]>,
        {
            as_scalar(self.as_ref())
        }

        fn as_scalar_mut(&'a mut self) -> &'a mut [f32]
        where
            Self: AsMut<[f32s]>,
        {
            as_scalar_mut(self.as_mut())
        }
    }

    pub fn simd_with<T>(mut iter: T) -> f32s
    where
        T: ExactSizeIterator + Iterator<Item = f32>,
    {
        let mut af_deriv: MaybeUninit<f32s> = MaybeUninit::uninit();
        unsafe {
            let ptr = af_deriv.as_mut_ptr() as *mut f32;
            for i in 0..f32s::lanes() {
                ptr.add(i).write(iter.next().unwrap());
            }
            af_deriv.assume_init()
        }
    }

    pub fn simd_to_iter<'a>(simd: f32s) -> std::slice::Iter<'a, f32> {
        let slice = [simd];
        unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const f32, f32s::lanes()).iter() }
    }

    /// Returns a mask with the first n lanes set to true
    pub fn mask(n: usize) -> mask_s {
        assert!(n <= mask_s::lanes());
        let mut m = MaybeUninit::zeroed();
        unsafe {
            let ptr = m.as_mut_ptr() as *mut usize_s;
            for i in 0..n {
                ptr.add(i).write(usize_s::MAX)
            }
            m.assume_init()
        }
    }

    pub fn sum(arr: &[f32s], len: usize) -> f32 {
        assert!(arr.len() * f32s::lanes() >= len);
        // the additional math is so that we get the lane size instead of zero when the len is divisible
        let mask = mask((len - 1) % f32s::lanes() + 1);
        let mut iter = arr.iter().rev();
        let mut val = mask.select(*iter.next().unwrap(), f32s::splat(0.));

        for i in iter {
            val += *i
        }
        val.sum()
    }
}

pub mod error {
    use std::{error::Error, fmt::Display};

    #[derive(Debug, Clone)]
    pub struct BuilderError {
        missing: &'static str,
    }

    impl BuilderError {
        pub fn new(missing: &'static str) -> Self {
            Self { missing }
        }
    }

    impl Display for BuilderError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "Missing field '{}'", self.missing)
        }
    }

    impl Error for BuilderError {}
}

// ///Shuffles an array of consequtive integers and allows yout to iterate through them
// pub struct IndexShuffler {
//     idxs: Box<[usize]>,
//     idx: usize,
//     rng: SmallRng,
// }
// impl IndexShuffler {
//     pub fn new(size: usize) -> Self {
//         let mut t = IndexShuffler {
//             idxs: (0..size).collect(),
//             idx: 0,
//             rng: SmallRng::seed_from_u64(0),
//         };
//         t.reset();
//         t
//     }
//     pub fn reset(&mut self) {
//         self.idxs.shuffle(&mut self.rng);
//         self.idx = 0;
//     }
// }

// impl Iterator for IndexShuffler {
//     type Item = usize;
//     fn next(&mut self) -> Option<Self::Item> {
//         let i = self.idxs.get(self.idx).copied();
//         self.idx += 1;
//         i
//     }
// }

pub struct IterMask<T> {
    iter: T,
    len: usize,
    total: usize,
    n: usize,
}

impl<T> IterMask<T> {
    pub fn new(iter: T, len: usize, total: usize) -> Self {
        Self {
            iter,
            len,
            total,
            n: 0,
        }
    }
}

impl<T> Iterator for IterMask<T>
where
    T: Iterator,
    T::Item: Default,
{
    type Item = T::Item;

    fn next(&mut self) -> Option<Self::Item> {
        if self.n >= self.total {
            self.n = 0;
        }

        let output = if self.n < self.len {
            self.iter.next()
        } else {
            Some(Default::default())
        };
        self.n += 1;
        output
    }
}
