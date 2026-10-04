use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::hash::{Hash, Hasher};

use sidereon_core::astro::time::{ExactEpoch, ExactEpochQuery};

#[pyclass(module = "sidereon._sidereon", name = "ExactEpoch")]
#[derive(Clone, Copy)]
pub struct PyExactEpoch {
    pub(crate) inner: ExactEpoch,
}

#[pymethods]
impl PyExactEpoch {
    #[staticmethod]
    fn new(seconds: i64, attoseconds: u64) -> PyResult<Self> {
        let inner = ExactEpoch::new(seconds, attoseconds)
            .ok_or_else(|| PyValueError::new_err("attoseconds must be below one second"))?;
        Ok(Self { inner })
    }

    #[classattr]
    const ATTOSECONDS_PER_SECOND: u64 = ExactEpoch::ATTOSECONDS_PER_SECOND;

    #[classattr]
    const J2000: Self = Self {
        inner: ExactEpoch::J2000,
    };

    #[staticmethod]
    #[pyo3(signature = (year, month, day, hour=0, minute=0, second=0.0))]
    fn from_civil(
        year: i32,
        month: i32,
        day: i32,
        hour: i32,
        minute: i32,
        second: f64,
    ) -> PyResult<Self> {
        let inner =
            ExactEpoch::from_civil(year, month, day, hour, minute, second).ok_or_else(|| {
                PyValueError::new_err("civil fields do not name a representable epoch")
            })?;
        Ok(Self { inner })
    }

    #[staticmethod]
    fn from_j2000_seconds(seconds: f64) -> PyResult<Self> {
        let inner = ExactEpoch::from_j2000_seconds(seconds).ok_or_else(|| {
            PyValueError::new_err("seconds must name a representable finite epoch")
        })?;
        Ok(Self { inner })
    }

    #[staticmethod]
    fn from_binary_j2000_seconds(seconds: f64) -> PyResult<PyExactEpochQuery> {
        let inner = ExactEpoch::from_binary_j2000_seconds(seconds)
            .ok_or_else(|| PyValueError::new_err("seconds must be finite"))?;
        Ok(PyExactEpochQuery { inner })
    }

    #[getter]
    fn whole_seconds(&self) -> i64 {
        self.inner.whole_seconds()
    }

    #[getter]
    fn attoseconds(&self) -> u64 {
        self.inner.attoseconds()
    }

    #[getter]
    fn sub_attosecond(&self) -> (i64, u16) {
        self.inner.sub_attosecond()
    }

    fn j2000_seconds(&self) -> f64 {
        self.inner.j2000_seconds()
    }

    fn split_julian_date(&self) -> (f64, f64) {
        self.inner.split_julian_date()
    }

    fn seconds_since(&self, earlier: &PyExactEpoch) -> f64 {
        self.inner.seconds_since(earlier.inner)
    }

    fn checked_add_seconds(&self, seconds: f64) -> PyResult<Self> {
        self.inner
            .checked_add_seconds(seconds)
            .map(|inner| Self { inner })
            .ok_or_else(|| PyValueError::new_err("addition does not produce a representable epoch"))
    }

    fn checked_sub_seconds(&self, seconds: f64) -> PyResult<Self> {
        self.inner
            .checked_sub_seconds(seconds)
            .map(|inner| Self { inner })
            .ok_or_else(|| {
                PyValueError::new_err("subtraction does not produce a representable epoch")
            })
    }

    fn query(&self) -> PyExactEpochQuery {
        PyExactEpochQuery {
            inner: self.inner.query(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ExactEpoch(whole_seconds={}, attoseconds={}, sub_attosecond={:?})",
            self.inner.whole_seconds(),
            self.inner.attoseconds(),
            self.inner.sub_attosecond()
        )
    }

    fn __eq__(&self, other: &PyExactEpoch) -> bool {
        self.inner == other.inner
    }

    fn __lt__(&self, other: &PyExactEpoch) -> bool {
        self.inner < other.inner
    }

    fn __le__(&self, other: &PyExactEpoch) -> bool {
        self.inner <= other.inner
    }

    fn __gt__(&self, other: &PyExactEpoch) -> bool {
        self.inner > other.inner
    }

    fn __ge__(&self, other: &PyExactEpoch) -> bool {
        self.inner >= other.inner
    }

    fn __hash__(&self) -> isize {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.inner.hash(&mut hasher);
        let hash = hasher.finish() as isize;
        if hash == -1 {
            -2
        } else {
            hash
        }
    }
}

#[pyclass(module = "sidereon._sidereon", name = "ExactEpochQuery")]
#[derive(Clone)]
pub struct PyExactEpochQuery {
    pub(crate) inner: ExactEpochQuery,
}

#[pymethods]
impl PyExactEpochQuery {
    #[staticmethod]
    fn from_binary_j2000_seconds(seconds: f64) -> PyResult<Self> {
        let inner = ExactEpoch::from_binary_j2000_seconds(seconds)
            .ok_or_else(|| PyValueError::new_err("seconds must be finite"))?;
        Ok(Self { inner })
    }

    #[staticmethod]
    fn at_epoch(epoch: &PyExactEpoch) -> Self {
        Self {
            inner: epoch.inner.query(),
        }
    }

    fn checked_add_binary_seconds(&self, seconds: f64) -> PyResult<Self> {
        self.inner
            .clone()
            .checked_add_binary_seconds(seconds)
            .map(|inner| Self { inner })
            .ok_or_else(|| PyValueError::new_err("offset must be finite"))
    }

    fn checked_sub_binary_seconds(&self, seconds: f64) -> PyResult<Self> {
        self.inner
            .clone()
            .checked_sub_binary_seconds(seconds)
            .map(|inner| Self { inner })
            .ok_or_else(|| PyValueError::new_err("offset must be finite"))
    }

    fn seconds_since_epoch(&self, earlier: &PyExactEpoch) -> f64 {
        self.inner.seconds_since(earlier.inner)
    }

    fn seconds_since_query(&self, earlier: &PyExactEpochQuery) -> f64 {
        self.inner.seconds_since_query(&earlier.inner)
    }

    fn j2000_seconds(&self) -> f64 {
        self.inner.seconds_since(ExactEpoch::J2000)
    }

    fn epoch(&self) -> PyExactEpoch {
        PyExactEpoch {
            inner: self.inner.epoch(),
        }
    }

    fn __repr__(&self) -> String {
        format!("ExactEpochQuery(j2000_seconds={})", self.j2000_seconds())
    }

    fn __eq__(&self, other: &PyExactEpochQuery) -> bool {
        self.inner == other.inner
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyExactEpoch>()?;
    module.add_class::<PyExactEpochQuery>()?;
    Ok(())
}
