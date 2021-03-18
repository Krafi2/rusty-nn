use crate::{
    f32s,
    helpers::{as_scalar, as_scalar_mut},
    serde::boxed_simd,
};

use serde::{Deserialize, Serialize};

use std::{fmt::Debug, slice::from_raw_parts_mut};

pub type WeightStorage = Storage;
pub type GradStorage = Storage;
pub type WeightHndl = Handle;
pub type GradHndl = Handle;
pub type WeightAllocator = Allocator;
pub type GradAllocator = Allocator;

pub use aligned_ref::{AlignedRef, AlignedRefMut};
mod aligned_ref {
    use super::*;

    #[derive(Debug, Clone, Copy)]
    pub struct AlignedRef<'a> {
        slice: &'a [f32s],
    }

    impl<'a> AlignedRef<'a> {
        pub fn new(slice: &'a [f32s]) -> Self {
            Self { slice }
        }

        pub fn as_scalar(&self) -> &[f32] {
            as_scalar(self.slice)
        }

        pub fn as_vector(&self) -> &[f32s] {
            self.slice
        }
    }

    impl<'a> AsRef<[f32]> for AlignedRef<'a> {
        fn as_ref(&self) -> &[f32] {
            self.as_scalar()
        }
    }

    impl<'a> AsRef<[f32s]> for AlignedRef<'a> {
        fn as_ref(&self) -> &[f32s] {
            self.as_vector()
        }
    }

    #[derive(Debug)]
    pub struct AlignedRefMut<'a> {
        slice: &'a mut [f32s],
    }

    impl<'a> AlignedRefMut<'a> {
        pub fn new(slice: &'a mut [f32s]) -> Self {
            Self { slice }
        }

        pub fn as_scalar(&self) -> &[f32] {
            as_scalar(self.slice)
        }

        pub fn as_vector(&self) -> &[f32s] {
            self.slice
        }

        pub fn as_scalar_mut(&mut self) -> &mut [f32] {
            as_scalar_mut(self.slice)
        }

        pub fn as_vector_mut(&mut self) -> &mut [f32s] {
            self.slice
        }
    }

    impl<'a> AsRef<[f32]> for AlignedRefMut<'a> {
        fn as_ref(&self) -> &[f32] {
            self.as_scalar()
        }
    }

    impl<'a> AsRef<[f32s]> for AlignedRefMut<'a> {
        fn as_ref(&self) -> &[f32s] {
            self.as_vector()
        }
    }

    impl<'a> AsMut<[f32]> for AlignedRefMut<'a> {
        fn as_mut(&mut self) -> &mut [f32] {
            self.as_scalar_mut()
        }
    }

    impl<'a> AsMut<[f32s]> for AlignedRefMut<'a> {
        fn as_mut(&mut self) -> &mut [f32s] {
            self.as_vector_mut()
        }
    }
}

pub use handle::Handle;
mod handle {
    use super::*;

    /// Generic handle for accesing blocks of memory stored within the matching Storage
    #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
    pub struct Handle {
        start: usize,
        end: usize,
    }

    impl Handle {
        pub(super) fn new(start: usize, end: usize) -> Self {
            Self { start, end }
        }

        pub(super) fn start(&self) -> usize {
            self.start
        }

        pub(super) fn end(&self) -> usize {
            self.end
        }
    }
}

pub use allocator::Allocator;
mod allocator {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct Allocator {
        mem: Vec<f32s>,
        buffered: usize,
    }

    impl Allocator {
        pub fn new() -> Self {
            Self {
                mem: Vec::new(),
                buffered: 0,
            }
        }

        fn new_handle(&self, len: usize) -> Handle {
            let start = self.mem.len() + self.buffered;
            Handle::new(start, start + len)
        }

        fn allocate_buffered(&mut self) {
            if self.buffered > 0 {
                self.mem
                    .extend(std::iter::repeat(f32s::splat(0.)).take(self.buffered));
                self.buffered = 0;
            }
        }

        pub fn allocate_zeroed(&mut self, len: usize) -> Handle {
            let handle = self.new_handle(len);
            self.buffered += len;
            handle
        }

        pub fn allocate<I>(&mut self, len: usize, iter: I) -> Handle
        where
            I: Iterator<Item = f32s>,
        {
            self.allocate_buffered();
            let len_before = self.mem.len();
            let handle = self.new_handle(len);
            self.mem.extend(iter.take(len));
            let received = self.mem.len() - len_before;
            assert_eq!(
                len, received,
                "Provided iterator did not yield enough elements. Expected: {}, Received: {}",
                len, received
            );
            handle
        }

        pub fn finish(mut self) -> Storage {
            self.allocate_buffered();
            let storage = self.mem.into_boxed_slice();
            Storage::new(storage)
        }
    }
}

pub use storage::Storage;
mod storage {
    use super::*;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Storage {
        #[serde(with = "boxed_simd")]
        storage: Box<[f32s]>,
    }

    impl Storage {
        pub(super) fn new(storage: Box<[f32s]>) -> Self {
            Self { storage }
        }

        pub fn get(&self, handle: Handle) -> AlignedRef<'_> {
            AlignedRef::new(&self.storage[handle.start()..handle.end()])
        }

        pub fn get_mut(&mut self, handle: Handle) -> AlignedRefMut<'_> {
            AlignedRefMut::new(&mut self.storage[handle.start()..handle.end()])
        }

        pub fn get_multiple_mut(
            &mut self,
            handles: &[Handle],
        ) -> Option<Vec<AlignedRefMut<'_>>> {
            let mut vec = Vec::with_capacity(handles.len() * 2);
            for (i, handle) in handles.iter().enumerate() {
                vec.push((handle.start(), i));
                vec.push((handle.end(), i));
            }
            vec.sort_by(|(a, _), (b, _)| a.cmp(b));

            for chunk in vec.chunks_exact(2) {
                let (_, i1) = chunk[0];
                let (_, i2) = chunk[1];
                if i1 != i2 {
                    return None;
                }
            }

            Some(
                handles
                    .iter()
                    .map(|handle| unsafe {
                        let ptr = self.storage.as_mut_ptr().add(handle.start());
                        let len = handle.end() - handle.start();
                        let slice = from_raw_parts_mut(ptr, len);
                        AlignedRefMut::new(slice)
                    })
                    .collect(),
            )
        }

        /// Get a reference to the raw contents of the storage
        pub fn raw(&self) -> &[f32s] {
            &self.storage
        }

        /// Get a mutable reference to the raw contents of the storage
        pub fn raw_mut(&mut self) -> &mut [f32s] {
            &mut self.storage
        }
    }
}

pub use weight_allocator::DualAllocator;
mod weight_allocator {
    use super::*;

    pub struct DualAllocator {
        weights: Allocator,
        grads: Allocator,
    }

    impl DualAllocator {
        pub fn new() -> Self {
            Self {
                weights: Allocator::new(),
                grads: Allocator::new(),
            }
        }

        pub fn allocate_zeroed(&mut self, len: usize) -> Handle {
            self.weights.allocate_zeroed(len);
            self.grads.allocate_zeroed(len)
        }

        pub fn allocate<I>(&mut self, len: usize, iter: I) -> Handle
        where
            I: Iterator<Item = f32s>,
        {
            self.weights.allocate(len, iter);
            self.grads.allocate_zeroed(len)
        }

        pub fn finish(self) -> (WeightStorage, GradAllocator) {
            (self.weights.finish(), self.grads)
        }
    }
}
