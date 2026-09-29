//! Preserve typed detail for refusals from public core time-model constructors.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use sidereon_core::astro::time::model::TimeModelError;

pub(crate) fn py_error(error: TimeModelError, message: String) -> PyErr {
    match error {
        TimeModelError::InvalidInput { field, reason } => Python::with_gil(|py| {
            let value_error = PyValueError::new_err(message.clone());
            let detail = PyDict::new(py);
            let result = (|| -> PyResult<()> {
                detail.set_item("family", "TimeModelError")?;
                detail.set_item("kind", "invalid_input")?;
                detail.set_item("message", error.to_string())?;
                detail.set_item("field", field)?;
                detail.set_item("reason", reason)?;
                value_error.value(py).setattr("detail", detail)?;
                Ok(())
            })();
            match result {
                Ok(()) => value_error,
                Err(error) => error,
            }
        }),
    }
}
