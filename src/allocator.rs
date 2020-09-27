use crate::f32s;

use serde::{Deserialize, Serialize};

use std::fmt::Debug;
use std::ops::{Deref, DerefMut, Index, IndexMut, Range};
use std::slice::{from_raw_parts, from_raw_parts_mut};

pub type WeightMediator<'a, T> = Mediator<'a, T, WeightHndl>;
pub type GradMediator<'a, T> = Mediator<'a, T, GradHdnl>;

/// Handle for accesing slices. We can't use simple references because rust doesn't allow self reference and it does make serialization easier.
#[derive(Serialize, Deserialize, Clone)]
pub struct VecHndl {
    pub(self) range: Range<usize>, //cannot be constructed outside of this module
}

/// A public version of handle so we gate access to the SealedHandle functions
pub trait Handle: private::SealedHandle {}

/// This is the maximum extent of privacy Rust allows me to use so please just dont ever use this module
mod private {
    use super::*;
    pub trait SealedHandle {
        fn handle(&self) -> &VecHndl;
        fn new(handle: VecHndl) -> Self;
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct WeightHndl(VecHndl);
impl private::SealedHandle for WeightHndl {
    fn handle(&self) -> &VecHndl {
        &self.0
    }
    fn new(handle: VecHndl) -> Self {
        Self(handle)
    }
}
impl Handle for WeightHndl {}

#[derive(Serialize, Deserialize, Clone)]
pub struct GradHdnl(VecHndl);
impl private::SealedHandle for GradHdnl {
    fn handle(&self) -> &VecHndl {
        &self.0
    }
    fn new(handle: VecHndl) -> Self {
        Self(handle)
    }
}
impl Handle for GradHdnl {}

pub struct Mediator<'a, T: 'a, H: Handle> {
    arr: T,
    _marker: std::marker::PhantomData<&'a H>,
}
impl<'a, T: 'a, H: Handle> Mediator<'a, T, H> {
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
    #[inline(always)]
    pub fn get<'b, E>(&self, handle: &'b H) -> Handled<&'b H, &'b [E]>
    where
        T: AsRef<[E]>,
    {
        let range = &handle.handle().range;
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
    #[inline(always)]
    pub fn get_mut<'b, E>(&mut self, handle: &'b mut H) -> Handled<&'b mut H, &'b mut [E]>
    where
        T: AsMut<[E]>,
    {
        let range = handle.handle().range.clone();
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
        let handle = VecHndl {
            range: self.0.len()..self.0.len() + n,
        };
        self.0.extend(std::iter::repeat(f32s::splat(0.)).take(n));
        (WeightHndl(handle.clone()), GradHdnl(handle))
    }
    /// Allocates n elements by calling init to get their value
    pub fn allocate_with<F: FnMut() -> f32>(
        &mut self,
        n: usize,
        mut init: F,
    ) -> (WeightHndl, GradHdnl) {
        let handle = VecHndl {
            range: self.0.len()..self.0.len() + n,
        };
        self.0.reserve(n);

        unsafe {
            let ptr: *mut f32 = std::mem::transmute(self.0.as_mut_ptr());
            for i in 0..n * f32s::lanes() {
                let val = init();
                ptr.add(i).write(val)
            }
            self.0.set_len(self.0.len() + n);
        }
        (WeightHndl(handle.clone()), GradHdnl(handle))
    }
    pub fn new(vec: &'a mut Vec<f32s>) -> Self {
        Allocator(vec)
    }
}

/// Constructs a new handle which has no effect and when used returns an empty slice.
pub fn invalid_handle<T: Handle>() -> T {
    T::new(VecHndl { range: 0..0 })
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
