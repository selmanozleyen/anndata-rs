use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Tracks a hard memory ceiling across the entire pipeline.
/// All allocations go through this so the engine never exceeds the user-defined
/// RAM limit regardless of how many concurrent tasks are in flight.
#[derive(Clone)]
pub struct MemoryBudget {
    limit_bytes: usize,
    used: Arc<AtomicUsize>,
}

impl MemoryBudget {
    pub fn new(limit_bytes: usize) -> Self {
        assert!(limit_bytes > 0, "memory budget must be > 0");
        Self {
            limit_bytes,
            used: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub fn limit(&self) -> usize {
        self.limit_bytes
    }

    pub fn used(&self) -> usize {
        self.used.load(Ordering::Relaxed)
    }

    pub fn available(&self) -> usize {
        self.limit_bytes.saturating_sub(self.used())
    }

    /// Try to reserve `bytes`. Returns false if it would exceed the budget.
    pub fn try_reserve(&self, bytes: usize) -> bool {
        let mut current = self.used.load(Ordering::Relaxed);
        loop {
            let new = current + bytes;
            if new > self.limit_bytes {
                return false;
            }
            match self.used.compare_exchange_weak(
                current,
                new,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => return true,
                Err(actual) => current = actual,
            }
        }
    }

    /// Release previously reserved bytes.
    pub fn release(&self, bytes: usize) {
        self.used.fetch_sub(bytes, Ordering::AcqRel);
    }
}

/// A pre-sized, reusable byte buffer that participates in the memory budget.
/// When dropped, its allocation is released back to the budget.
pub struct BufferPool {
    budget: MemoryBudget,
}

impl BufferPool {
    pub fn new(budget: MemoryBudget) -> Self {
        Self { budget }
    }

    pub fn budget(&self) -> &MemoryBudget {
        &self.budget
    }

    /// Create a new BufferPool sharing the same underlying MemoryBudget.
    /// Both pools will draw from and release to the same atomic counter.
    pub fn clone_with_same_budget(&self) -> Self {
        Self {
            budget: self.budget.clone(),
        }
    }

    /// Allocate a Vec<T> of `count` elements, reserving from the memory budget.
    /// Returns None if the budget cannot accommodate it.
    pub fn alloc_vec<T: Default + Clone>(&self, count: usize) -> Option<TrackedVec<T>> {
        let bytes = count * std::mem::size_of::<T>();
        if self.budget.try_reserve(bytes) {
            Some(TrackedVec {
                inner: vec![T::default(); count],
                tracked_bytes: bytes,
                budget: self.budget.clone(),
            })
        } else {
            None
        }
    }

    /// How many rows of a dense (n_rows x n_cols) matrix of element size
    /// `elem_bytes` fit in the available budget, reserving `headroom` bytes
    /// for sparse metadata / bookkeeping.
    pub fn max_rows_for_dense(
        &self,
        n_cols: usize,
        elem_bytes: usize,
        headroom: usize,
    ) -> usize {
        let avail = self.budget.available().saturating_sub(headroom);
        let row_bytes = n_cols * elem_bytes;
        if row_bytes == 0 {
            return usize::MAX;
        }
        avail / row_bytes
    }
}

/// A Vec whose byte footprint is tracked against a MemoryBudget.
/// Releasing happens automatically on drop.
pub struct TrackedVec<T> {
    inner: Vec<T>,
    tracked_bytes: usize,
    budget: MemoryBudget,
}

impl<T> TrackedVec<T> {
    pub fn as_slice(&self) -> &[T] {
        &self.inner
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.inner
    }

    pub fn into_inner(mut self) -> Vec<T> {
        self.budget.release(self.tracked_bytes);
        self.tracked_bytes = 0;
        std::mem::take(&mut self.inner)
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl<T> Drop for TrackedVec<T> {
    fn drop(&mut self) {
        if self.tracked_bytes > 0 {
            self.budget.release(self.tracked_bytes);
        }
    }
}

impl<T> std::ops::Deref for TrackedVec<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        &self.inner
    }
}

impl<T> std::ops::DerefMut for TrackedVec<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_tracks_usage() {
        let b = MemoryBudget::new(1024);
        assert!(b.try_reserve(512));
        assert_eq!(b.used(), 512);
        assert_eq!(b.available(), 512);
        assert!(!b.try_reserve(600));
        assert!(b.try_reserve(512));
        assert_eq!(b.available(), 0);
        b.release(1024);
        assert_eq!(b.available(), 1024);
    }

    #[test]
    fn pool_alloc_respects_budget() {
        let pool = BufferPool::new(MemoryBudget::new(100));
        let v = pool.alloc_vec::<u8>(50);
        assert!(v.is_some());
        let v2 = pool.alloc_vec::<u8>(60);
        assert!(v2.is_none());
        drop(v);
        let v3 = pool.alloc_vec::<u8>(60);
        assert!(v3.is_some());
    }
}
