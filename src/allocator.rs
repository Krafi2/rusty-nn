use crate::f32s;

use serde::{Deserialize, Serialize};

use std::fmt::Debug;
use std::ops::{Deref, DerefMut, Index, IndexMut, Range};
use std::slice::{from_raw_parts, from_raw_parts_mut};

pub type WeightMediator<'a, T> = Mediator<'a, T, WeightHndl>;
pub type GradMediator<'a, T> = Mediator<'a, T, GradHdnl>;

/// This is the maximum extent of privacy Rust allows me to use so please just dont ever use this module
mod private {
    use super::*;
    pub trait SealedHandle {
        fn new(handle: Range<usize>) -> Self;
        fn range(&self) -> &Range<usize>;
    }
}
use private::SealedHandle;

/// A public marker trait for structs implementing SealedHandle
pub trait Handle: SealedHandle {}
impl<T: SealedHandle> Handle for T {}

/// Create a new  handle struct called $name
macro_rules! handle {
    ($name:ident) => {
        #[derive(Serialize, Deserialize)]
        pub struct $name {
            range: Range<usize>,
        }
        impl $name {
            /// Clones the handle.
            /// This function is unsafe because cloning the handle can allow you to obtain
            /// multiple references to the same data from the Mediator.
            /// As such it is the user's responsibility to make sure that a Mediator is accessed
            /// in accordance to the borrowing rules.
            pub unsafe fn clone(&self) -> Self {
                Self::new(self.range().clone())
            }
        }

        impl SealedHandle for $name {
            /// Create a new handle from a range in the data block
            fn new(range: Range<usize>) -> Self {
                Self { range }
            }

            /// Obtain the handle's internal range
            fn range(&self) -> &Range<usize> {
                &self.range
            }
        }
    };
}

handle!(WeightHndl);
handle!(GradHdnl);

/// This struct mediates access to its underlying data through the use of handles specified by the H parameter.
pub struct Mediator<'a, T: 'a, H: Handle> {
    arr: T,
    _marker: std::marker::PhantomData<&'a H>,
}

impl<'a, T: 'a, H: Handle> Mediator<'a, T, H> {
    /// Construct a new Mediator object from a data array
    pub fn new(arr: T) -> Mediator<'a, T, H> {
        Mediator {
            arr,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<'a, T: 'a, U: 'a, H> Mediator<'a, T, H>
where
    H: Handle,
    T: Deref<Target = U>,
    U: Index<Range<usize>>,
    U: ?Sized,
{
    #[inline]
    /// Get the concrete reference to the data held by the handle
    pub fn get<'b, E>(&self, handle: &'b H) -> Handled<&'b H, &'b [E]>
    where
        T: AsRef<[E]>,
    {
        let range = handle.range();
        unsafe {
            return Handled {
                handle,
                slice: from_raw_parts(
                    self.arr.as_ref().as_ptr().add(range.start),
                    range.end - range.start,
                ),
            };
        }
    }
}

impl<'a, T: 'a, H, U: 'a> Mediator<'a, T, H>
where
    H: Handle,
    T: DerefMut<Target = U>,
    U: IndexMut<Range<usize>>,
    U: ?Sized,
{
    #[inline]
    /// Get the concrete mutable reference to the data held by the handle
    pub fn get_mut<'b, E>(&mut self, handle: &'b mut H) -> Handled<&'b mut H, &'b mut [E]>
    where
        T: AsMut<[E]>,
    {
        let range = handle.range().clone();
        unsafe {
            return Handled {
                handle,
                slice: from_raw_parts_mut(
                    self.arr.as_mut().as_mut_ptr().add(range.start),
                    range.end - range.start,
                ),
            };
        }
    }
}

/// Struct for allocating space in vector
pub struct Allocator<'a>(&'a mut Vec<f32s>);
impl<'a> Allocator<'a> {
    /// Allocates n elements initialized to 0
    pub fn allocate(&mut self, n: usize) -> (WeightHndl, GradHdnl) {
        let range = self.0.len()..self.0.len() + n;
        self.0.extend(std::iter::repeat(f32s::splat(0.)).take(n));
        (WeightHndl::new(range.clone()), GradHdnl::new(range))
    }
    /// Allocates n elements by calling init to get their value
    pub fn allocate_with<F: FnMut() -> f32>(
        &mut self,
        n: usize,
        mut init: F,
    ) -> (WeightHndl, GradHdnl) {
        let range = self.0.len()..self.0.len() + n;
        self.0.reserve(n);

        unsafe {
            let ptr = self.0.as_mut_ptr().add(self.0.len()) as *mut f32;
            for i in 0..n * f32s::lanes() {
                let val = init();
                ptr.add(i).write(val)
            }
            self.0.set_len(self.0.len() + n);
        }
        (WeightHndl::new(range.clone()), GradHdnl::new(range))
    }
    pub fn new(vec: &'a mut Vec<f32s>) -> Self {
        Allocator(vec)
    }
}

/// Constructs a new handle which has no effect used and returns an empty slice.
pub fn invalid_handle<T: Handle>() -> T {
    T::new(0..0)
}

/// Struct containing a slice and a handle so that the compiler enforces borrowing rules for us and our code is sound.
pub struct Handled<H, T> {
    #[allow(dead_code)]
    handle: H, //we need to keep the handle in order for the compiler to enforce borrowing rules
    slice: T,
}

impl<H, T: Deref> Deref for Handled<H, T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.slice
    }
}

impl<H, T: DerefMut> DerefMut for Handled<H, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.slice
    }
}

impl<H, T: Debug> Debug for Handled<H, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.slice.fmt(f)
    }
}

impl<H, T: std::hash::Hash> std::hash::Hash for Handled<H, T> {
    fn hash<H_: std::hash::Hasher>(&self, state: &mut H_) {
        self.slice.hash(state);
    }
}

impl<H, T: Index<U>, U> Index<U> for Handled<H, T> {
    type Output = <T as Index<U>>::Output;
    fn index(&self, index: U) -> &Self::Output {
        self.slice.index(index)
    }
}

impl<H, T: IndexMut<U>, U> IndexMut<U> for Handled<H, T> {
    fn index_mut(&mut self, index: U) -> &mut Self::Output {
        self.slice.index_mut(index)
    }
}

impl<H, T: IntoIterator> IntoIterator for Handled<H, T> {
    type Item = <T as IntoIterator>::Item;
    type IntoIter = <T as IntoIterator>::IntoIter;
    fn into_iter(self) -> Self::IntoIter {
        self.slice.into_iter()
    }
}
