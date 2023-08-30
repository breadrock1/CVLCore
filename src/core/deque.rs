use std::collections::VecDeque;

#[derive(Clone)]
pub struct CvlMatDeque<T> {
    pub inner: VecDeque<T>,
}

impl<T> CvlMatDeque<T> {
    pub fn new(size: usize) -> Self {
        CvlMatDeque {
            inner: VecDeque::with_capacity(size),
        }
    }

    pub fn length(&self) -> usize {
        self.inner.len()
    }

    pub fn set_max_size(&mut self, queue_size: usize) {
        self.inner = VecDeque::with_capacity(queue_size)
    }

    pub fn max_size(&self) -> usize {
        self.inner.capacity()
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        self.inner.get(index)
    }

    pub fn take_first(&mut self) -> Option<T> {
        self.inner.pop_front()
    }

    pub fn take_last(&mut self) -> Option<T> {
        self.inner.pop_back()
    }

    pub fn push(&mut self, value: T) {
        let curr_len = self.length();
        let curr_max = self.max_size();
        match curr_len >= curr_max {
            false => self.inner.push_back(value),
            true => {
                let _ = self.inner.remove(0);
                self.inner.push_back(value);
            }
        }
    }
}

impl<T> Default for CvlMatDeque<T> {
    fn default() -> Self {
        CvlMatDeque {
            inner: VecDeque::<T>::with_capacity(5),
        }
    }
}

impl<T> From<Vec<T>> for CvlMatDeque<T> {
    fn from(value: Vec<T>) -> Self {
        let vec_size = value.len();
        let mut deque: CvlMatDeque<T> = CvlMatDeque::new(vec_size);
        value.into_iter().for_each(|item| deque.push(item));

        deque
    }
}
