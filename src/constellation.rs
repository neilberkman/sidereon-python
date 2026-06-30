//! GNSS constellation identity catalog binding.
//!
//! Thin marshaling over [`sidereon_core::constellation`]: it turns already
//! fetched CelesTrak `gps-ops` OMM JSON and NAVCEN status HTML into normalized
//! satellite identity records, merges the two sources, and exposes the core
//! validation and diff helpers. The binding fetches no bytes of its own and
//! performs no catalog logic; every record, CSV byte, validation report, and
//! diff is exactly what `sidereon-core` produces.

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::IntoPyObjectExt;

use sidereon_core::astro::omm::parse_json_array;
use sidereon_core::constellation::{
    self, BoolStyle, CelestrakSource, Diff, FieldChange, NavcenSource, NavcenStatus, Record,
    RecordSource, Validation,
};
use sidereon_core::GnssSystem;

use crate::marshal::PyGnssSystem;
use crate::{ConstellationError, OmmParseError};

fn to_constellation_err(err: constellation::ConstellationError) -> PyErr {
    ConstellationError::new_err(err.to_string())
}

fn to_omm_err<E: std::fmt::Display>(err: E) -> PyErr {
    OmmParseError::new_err(err.to_string())
}

/// CelesTrak `gps-ops` provenance preserved on a record.
#[pyclass(module = "sidereon._sidereon", name = "CelestrakSource")]
#[derive(Clone, PartialEq, Eq)]
pub struct PyCelestrakSource {
    inner: CelestrakSource,
}

impl PyCelestrakSource {
    fn from_inner(inner: CelestrakSource) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyCelestrakSource {
    /// CelesTrak GP group the record came from (`gps-ops`).
    #[getter]
    fn group(&self) -> String {
        self.inner.group.clone()
    }

    /// The OMM `OBJECT_NAME`.
    #[getter]
    fn object_name(&self) -> Option<String> {
        self.inner.object_name.clone()
    }

    /// The OMM `OBJECT_ID` (international designator).
    #[getter]
    fn object_id(&self) -> Option<String> {
        self.inner.object_id.clone()
    }

    /// The OMM `EPOCH`, ISO-8601.
    #[getter]
    fn epoch(&self) -> Option<String> {
        self.inner.epoch.clone()
    }

    /// Block type parsed from the object name (`IIF`, `IIR`, `IIR-M`, `III`).
    #[getter]
    fn block_type(&self) -> Option<String> {
        self.inner.block_type.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "CelestrakSource(group={:?}, object_name={:?}, block_type={:?})",
            self.inner.group, self.inner.object_name, self.inner.block_type
        )
    }

    fn __eq__(&self, other: &PyCelestrakSource) -> bool {
        self == other
    }
}

/// NAVCEN status provenance preserved on a record or recorded as a conflict.
#[pyclass(module = "sidereon._sidereon", name = "NavcenSource")]
#[derive(Clone, PartialEq, Eq)]
pub struct PyNavcenSource {
    inner: NavcenSource,
}

impl PyNavcenSource {
    fn from_inner(inner: NavcenSource) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyNavcenSource {
    /// Space Vehicle Number.
    #[getter]
    fn svn(&self) -> Option<u16> {
        self.inner.svn
    }

    /// Block type as reported by NAVCEN.
    #[getter]
    fn block_type(&self) -> Option<String> {
        self.inner.block_type.clone()
    }

    /// Orbital plane letter.
    #[getter]
    fn plane(&self) -> Option<String> {
        self.inner.plane.clone()
    }

    /// Slot within the plane.
    #[getter]
    fn slot(&self) -> Option<String> {
        self.inner.slot.clone()
    }

    /// Clock type.
    #[getter]
    fn clock(&self) -> Option<String> {
        self.inner.clock.clone()
    }

    /// NANU type code (for example `FCSTSUMM`, `UNUSABLE`, `DECOM`).
    #[getter]
    fn nanu_type(&self) -> Option<String> {
        self.inner.nanu_type.clone()
    }

    /// NANU subject line.
    #[getter]
    fn nanu_subject(&self) -> Option<String> {
        self.inner.nanu_subject.clone()
    }

    /// Whether the row carried an active NANU.
    #[getter]
    fn active_nanu(&self) -> bool {
        self.inner.active_nanu
    }

    fn __repr__(&self) -> String {
        format!(
            "NavcenSource(svn={:?}, nanu_type={:?}, active_nanu={})",
            self.inner.svn, self.inner.nanu_type, self.inner.active_nanu
        )
    }

    fn __eq__(&self, other: &PyNavcenSource) -> bool {
        self == other
    }
}

/// Per-source provenance kept on a [`PyRecord`].
#[pyclass(module = "sidereon._sidereon", name = "RecordSource")]
#[derive(Clone, PartialEq, Eq)]
pub struct PyRecordSource {
    inner: RecordSource,
}

impl PyRecordSource {
    fn from_inner(inner: RecordSource) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyRecordSource {
    /// CelesTrak `gps-ops` identity provenance, when present.
    #[getter]
    fn celestrak(&self) -> Option<PyCelestrakSource> {
        self.inner
            .celestrak
            .clone()
            .map(PyCelestrakSource::from_inner)
    }

    /// NAVCEN overlay merged into this record, when present.
    #[getter]
    fn navcen(&self) -> Option<PyNavcenSource> {
        self.inner.navcen.clone().map(PyNavcenSource::from_inner)
    }

    /// NAVCEN row that matched the PRN but was not merged because its block type
    /// was incompatible with the CelesTrak identity (a PRN transition).
    #[getter]
    fn navcen_conflict(&self) -> Option<PyNavcenSource> {
        self.inner
            .navcen_conflict
            .clone()
            .map(PyNavcenSource::from_inner)
    }

    fn __repr__(&self) -> String {
        format!(
            "RecordSource(celestrak={}, navcen={}, navcen_conflict={})",
            self.inner.celestrak.is_some(),
            self.inner.navcen.is_some(),
            self.inner.navcen_conflict.is_some()
        )
    }

    fn __eq__(&self, other: &PyRecordSource) -> bool {
        self == other
    }
}

/// A normalized GNSS satellite identity record.
#[pyclass(module = "sidereon._sidereon", name = "Record")]
#[derive(Clone, PartialEq, Eq)]
pub struct PyRecord {
    inner: Record,
}

impl PyRecord {
    fn from_inner(inner: Record) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyRecord {
    /// The constellation. GPS today; the type is system-tagged for extension.
    #[getter]
    fn system(&self) -> PyGnssSystem {
        self.inner.system.into()
    }

    /// The within-constellation PRN.
    #[getter]
    fn prn(&self) -> u16 {
        self.inner.prn
    }

    /// Space Vehicle Number, when known (CelesTrak alone leaves this `None`).
    #[getter]
    fn svn(&self) -> Option<u16> {
        self.inner.svn
    }

    /// NORAD catalog id.
    #[getter]
    fn norad_id(&self) -> u32 {
        self.inner.norad_id
    }

    /// Canonical SP3/RINEX satellite token (`G03`).
    #[getter]
    fn sp3_id(&self) -> String {
        self.inner.sp3_id.clone()
    }

    /// GLONASS FDMA L1/L2 frequency-channel number (`k`, in `-7..=6`), `None`
    /// for the CDMA constellations.
    #[getter]
    fn fdma_channel(&self) -> Option<i8> {
        self.inner.fdma_channel
    }

    /// Present in the base identity source.
    #[getter]
    fn active(&self) -> bool {
        self.inner.active
    }

    /// Advisory usability flag.
    #[getter]
    fn usable(&self) -> bool {
        self.inner.usable
    }

    /// Source provenance.
    #[getter]
    fn source(&self) -> PyRecordSource {
        PyRecordSource::from_inner(self.inner.source.clone())
    }

    fn __repr__(&self) -> String {
        format!(
            "Record(prn={}, svn={:?}, norad_id={}, sp3_id={:?}, fdma_channel={:?}, \
             active={}, usable={})",
            self.inner.prn,
            self.inner.svn,
            self.inner.norad_id,
            self.inner.sp3_id,
            self.inner.fdma_channel,
            self.inner.active,
            self.inner.usable
        )
    }

    fn __eq__(&self, other: &PyRecord) -> bool {
        self == other
    }
}

/// A parsed row from NAVCEN's GPS constellation status table.
#[pyclass(module = "sidereon._sidereon", name = "NavcenStatus")]
#[derive(Clone, PartialEq, Eq)]
pub struct PyNavcenStatus {
    inner: NavcenStatus,
}

impl PyNavcenStatus {
    fn from_inner(inner: NavcenStatus) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyNavcenStatus {
    /// The constellation (GPS).
    #[getter]
    fn system(&self) -> PyGnssSystem {
        self.inner.system.into()
    }

    /// The within-constellation PRN.
    #[getter]
    fn prn(&self) -> u16 {
        self.inner.prn
    }

    /// Space Vehicle Number, when present.
    #[getter]
    fn svn(&self) -> Option<u16> {
        self.inner.svn
    }

    /// Whether the satellite is usable per the active NANU (if any).
    #[getter]
    fn usable(&self) -> bool {
        self.inner.usable
    }

    /// Whether the row carried an active NANU.
    #[getter]
    fn active_nanu(&self) -> bool {
        self.inner.active_nanu
    }

    /// NANU type code.
    #[getter]
    fn nanu_type(&self) -> Option<String> {
        self.inner.nanu_type.clone()
    }

    /// NANU subject line.
    #[getter]
    fn nanu_subject(&self) -> Option<String> {
        self.inner.nanu_subject.clone()
    }

    /// Orbital plane letter.
    #[getter]
    fn plane(&self) -> Option<String> {
        self.inner.plane.clone()
    }

    /// Slot within the plane.
    #[getter]
    fn slot(&self) -> Option<String> {
        self.inner.slot.clone()
    }

    /// Block type.
    #[getter]
    fn block_type(&self) -> Option<String> {
        self.inner.block_type.clone()
    }

    /// Clock type.
    #[getter]
    fn clock(&self) -> Option<String> {
        self.inner.clock.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "NavcenStatus(prn={}, svn={:?}, usable={}, active_nanu={}, nanu_type={:?})",
            self.inner.prn,
            self.inner.svn,
            self.inner.usable,
            self.inner.active_nanu,
            self.inner.nanu_type
        )
    }

    fn __eq__(&self, other: &PyNavcenStatus) -> bool {
        self == other
    }
}

/// Validation report for a constellation catalog.
#[pyclass(module = "sidereon._sidereon", name = "Validation")]
#[derive(Clone, PartialEq, Eq)]
pub struct PyValidation {
    inner: Validation,
}

impl PyValidation {
    fn from_inner(inner: Validation) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyValidation {
    /// Active+usable catalog SP3 ids absent from the compared product.
    #[getter]
    fn missing_sp3_ids(&self) -> Vec<String> {
        self.inner.missing_sp3_ids.clone()
    }

    /// `(system, PRN)` pairs that appear in more than one record. Keyed by
    /// system so a legitimate multi-system catalog (GPS PRN 1 and Galileo PRN 1)
    /// is not a false duplicate.
    #[getter]
    fn duplicate_prns(&self) -> Vec<(PyGnssSystem, u16)> {
        self.inner
            .duplicate_prns
            .iter()
            .map(|&(sys, prn)| (sys.into(), prn))
            .collect()
    }

    /// NORAD ids that appear in more than one record.
    #[getter]
    fn duplicate_norad_ids(&self) -> Vec<u32> {
        self.inner.duplicate_norad_ids.clone()
    }

    /// `(system, PRN)` pairs that are inactive or unusable.
    #[getter]
    fn inactive_unusable_prns(&self) -> Vec<(PyGnssSystem, u16)> {
        self.inner
            .inactive_unusable_prns
            .iter()
            .map(|&(sys, prn)| (sys.into(), prn))
            .collect()
    }

    /// SP3 ids present in the product but absent from the active+usable catalog.
    #[getter]
    fn extra_sp3_ids(&self) -> Vec<String> {
        self.inner.extra_sp3_ids.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "Validation(missing_sp3_ids={:?}, duplicate_prns={:?}, duplicate_norad_ids={:?}, \
             inactive_unusable_prns={:?}, extra_sp3_ids={:?})",
            self.inner.missing_sp3_ids,
            self.inner.duplicate_prns,
            self.inner.duplicate_norad_ids,
            self.inner.inactive_unusable_prns,
            self.inner.extra_sp3_ids
        )
    }

    fn __eq__(&self, other: &PyValidation) -> bool {
        self == other
    }
}

/// A single field change on a PRN that exists in both diffed snapshots.
///
/// `from_` and `to` carry the previous and current value of the changed field;
/// their Python type depends on the field (`int`, `str`, `int | None`, `bool`).
#[pyclass(module = "sidereon._sidereon", name = "FieldChange")]
pub struct PyFieldChange {
    system: PyGnssSystem,
    prn: u16,
    from_obj: PyObject,
    to_obj: PyObject,
}

impl Clone for PyFieldChange {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self {
            system: self.system,
            prn: self.prn,
            from_obj: self.from_obj.clone_ref(py),
            to_obj: self.to_obj.clone_ref(py),
        })
    }
}

impl PyFieldChange {
    fn build<'py, T>(py: Python<'py>, change: &FieldChange<T>) -> PyResult<Self>
    where
        T: Clone + IntoPyObject<'py>,
    {
        Ok(Self {
            system: change.system.into(),
            prn: change.prn,
            from_obj: change.from.clone().into_py_any(py)?,
            to_obj: change.to.clone().into_py_any(py)?,
        })
    }

    fn build_list<'py, T>(py: Python<'py>, changes: &[FieldChange<T>]) -> PyResult<Vec<Self>>
    where
        T: Clone + IntoPyObject<'py>,
    {
        changes.iter().map(|c| Self::build(py, c)).collect()
    }
}

#[pymethods]
impl PyFieldChange {
    /// The constellation.
    #[getter]
    fn system(&self) -> PyGnssSystem {
        self.system
    }

    /// The PRN whose field changed.
    #[getter]
    fn prn(&self) -> u16 {
        self.prn
    }

    /// The value in the previous snapshot.
    #[getter]
    fn from_(&self, py: Python<'_>) -> PyObject {
        self.from_obj.clone_ref(py)
    }

    /// The value in the current snapshot.
    #[getter]
    fn to(&self, py: Python<'_>) -> PyObject {
        self.to_obj.clone_ref(py)
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "FieldChange(prn={}, from_={}, to={})",
            self.prn,
            self.from_obj.bind(py).repr()?,
            self.to_obj.bind(py).repr()?
        ))
    }

    fn __eq__(&self, py: Python<'_>, other: &PyFieldChange) -> PyResult<bool> {
        Ok(self.system == other.system
            && self.prn == other.prn
            && self.from_obj.bind(py).eq(other.from_obj.bind(py))?
            && self.to_obj.bind(py).eq(other.to_obj.bind(py))?)
    }
}

/// Change report between two catalog snapshots, keyed by `(system, prn)`.
#[pyclass(module = "sidereon._sidereon", name = "Diff")]
pub struct PyDiff {
    added: Vec<PyRecord>,
    removed: Vec<PyRecord>,
    norad_reassigned: Vec<PyFieldChange>,
    sp3_id_changed: Vec<PyFieldChange>,
    svn_changed: Vec<PyFieldChange>,
    fdma_channel_changed: Vec<PyFieldChange>,
    activity_changed: Vec<PyFieldChange>,
    usability_changed: Vec<PyFieldChange>,
}

impl PyDiff {
    fn build(py: Python<'_>, inner: &Diff) -> PyResult<Self> {
        Ok(Self {
            added: inner
                .added
                .iter()
                .cloned()
                .map(PyRecord::from_inner)
                .collect(),
            removed: inner
                .removed
                .iter()
                .cloned()
                .map(PyRecord::from_inner)
                .collect(),
            norad_reassigned: PyFieldChange::build_list(py, &inner.norad_reassigned)?,
            sp3_id_changed: PyFieldChange::build_list(py, &inner.sp3_id_changed)?,
            svn_changed: PyFieldChange::build_list(py, &inner.svn_changed)?,
            fdma_channel_changed: PyFieldChange::build_list(py, &inner.fdma_channel_changed)?,
            activity_changed: PyFieldChange::build_list(py, &inner.activity_changed)?,
            usability_changed: PyFieldChange::build_list(py, &inner.usability_changed)?,
        })
    }
}

#[pymethods]
impl PyDiff {
    /// PRNs present only in the current snapshot.
    #[getter]
    fn added(&self) -> Vec<PyRecord> {
        self.added.clone()
    }

    /// PRNs present only in the previous snapshot.
    #[getter]
    fn removed(&self) -> Vec<PyRecord> {
        self.removed.clone()
    }

    /// NORAD id reassignments on a held PRN.
    #[getter]
    fn norad_reassigned(&self) -> Vec<PyFieldChange> {
        self.norad_reassigned.clone()
    }

    /// SP3 id changes on a held PRN.
    #[getter]
    fn sp3_id_changed(&self) -> Vec<PyFieldChange> {
        self.sp3_id_changed.clone()
    }

    /// SVN changes on a held PRN.
    #[getter]
    fn svn_changed(&self) -> Vec<PyFieldChange> {
        self.svn_changed.clone()
    }

    /// GLONASS FDMA channel changes on a held PRN.
    #[getter]
    fn fdma_channel_changed(&self) -> Vec<PyFieldChange> {
        self.fdma_channel_changed.clone()
    }

    /// Activity flips on a held PRN.
    #[getter]
    fn activity_changed(&self) -> Vec<PyFieldChange> {
        self.activity_changed.clone()
    }

    /// Usability flips on a held PRN.
    #[getter]
    fn usability_changed(&self) -> Vec<PyFieldChange> {
        self.usability_changed.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "Diff(added={}, removed={}, norad_reassigned={}, sp3_id_changed={}, svn_changed={}, \
             fdma_channel_changed={}, activity_changed={}, usability_changed={})",
            self.added.len(),
            self.removed.len(),
            self.norad_reassigned.len(),
            self.sp3_id_changed.len(),
            self.svn_changed.len(),
            self.fdma_channel_changed.len(),
            self.activity_changed.len(),
            self.usability_changed.len()
        )
    }
}

fn records_inner(records: &[PyRecord]) -> Vec<Record> {
    records.iter().map(|r| r.inner.clone()).collect()
}

fn bool_style(booleans: &str) -> PyResult<BoolStyle> {
    match booleans {
        "lower" => Ok(BoolStyle::Lower),
        "title" => Ok(BoolStyle::Title),
        other => Err(PyTypeError::new_err(format!(
            "booleans must be \"lower\" or \"title\", got {other:?}"
        ))),
    }
}

/// Build identity records for a constellation from CelesTrak OMM JSON array text.
///
/// Parses the JSON array with the core OMM parser (array elements that are not a
/// valid OMM object are skipped, not fatal), then derives a record per satellite
/// for `system` (PRN/slot, NORAD id, SP3 id, GLONASS FDMA channel, block type).
/// `system` defaults to GPS for backward compatibility. Records are returned
/// sorted by `(system, prn)`. Raises `OmmParseError` if the document is
/// malformed and `ConstellationError` if an `OBJECT_NAME` cannot be resolved to
/// a PRN/slot for `system`.
#[pyfunction]
#[pyo3(signature = (json, system=PyGnssSystem::GPS))]
fn from_celestrak_json(json: &str, system: PyGnssSystem) -> PyResult<Vec<PyRecord>> {
    let parsed = parse_json_array(json).map_err(to_omm_err)?;
    constellation::from_celestrak_omm(system.into(), &parsed.omms)
        .map(|records| records.into_iter().map(PyRecord::from_inner).collect())
        .map_err(to_constellation_err)
}

/// An OMM entry that the lenient catalog build could not resolve to a record.
///
/// Carries the entry's identity (not just a count) so the caller can triage why
/// it was skipped: a satellite of another constellation in a combined feed, or
/// a satellite of the requested system whose name does not yet resolve to a
/// published slot/SVID.
#[pyclass(module = "sidereon._sidereon", name = "SkippedOmm")]
#[derive(Clone, PartialEq, Eq)]
pub struct PySkippedOmm {
    inner: constellation::SkippedOmm,
}

#[pymethods]
impl PySkippedOmm {
    /// The OMM `OBJECT_NAME`, when present.
    #[getter]
    fn object_name(&self) -> Option<String> {
        self.inner.object_name.clone()
    }

    /// The OMM `NORAD_CAT_ID`.
    #[getter]
    fn norad_id(&self) -> u32 {
        self.inner.norad_id
    }

    fn __repr__(&self) -> String {
        format!(
            "SkippedOmm(object_name={:?}, norad_id={})",
            self.inner.object_name, self.inner.norad_id
        )
    }

    fn __eq__(&self, other: &PySkippedOmm) -> bool {
        self == other
    }
}

/// The result of a lenient constellation catalog build: the records that
/// resolved, plus the OMM entries that did not.
#[pyclass(module = "sidereon._sidereon", name = "Catalog")]
pub struct PyCatalog {
    records: Vec<PyRecord>,
    skipped: Vec<PySkippedOmm>,
}

#[pymethods]
impl PyCatalog {
    /// Records built from resolvable OMM entries, sorted by `(system, prn)`.
    #[getter]
    fn records(&self) -> Vec<PyRecord> {
        self.records.clone()
    }

    /// Entries whose `OBJECT_NAME` did not resolve to a PRN for the requested
    /// system, in input order.
    #[getter]
    fn skipped(&self) -> Vec<PySkippedOmm> {
        self.skipped.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "Catalog(records={}, skipped={})",
            self.records.len(),
            self.skipped.len()
        )
    }
}

/// Build a GNSS constellation catalog from a CelesTrak OMM JSON feed, skipping
/// (rather than aborting on) entries that do not resolve to `system`.
///
/// The lenient sibling of [`from_celestrak_json`]: feed it a raw combined
/// CelesTrak `gnss` feed and keep `catalog.records` (sorted by `(system, prn)`)
/// for the requested system, while `catalog.skipped` lists the entries that did
/// not resolve - satellites of other constellations, or freshly launched
/// satellites of `system` not yet in the published slot/SVID table. Raises
/// `OmmParseError` only when the document itself is malformed; an unresolvable
/// `OBJECT_NAME` is collected into `skipped` rather than raised.
#[pyfunction]
#[pyo3(signature = (json, system=PyGnssSystem::GPS))]
fn from_celestrak_omm_lenient(json: &str, system: PyGnssSystem) -> PyResult<PyCatalog> {
    let parsed = parse_json_array(json).map_err(to_omm_err)?;
    let catalog = constellation::from_celestrak_omm_lenient(system.into(), &parsed.omms);
    Ok(PyCatalog {
        records: catalog
            .records
            .into_iter()
            .map(PyRecord::from_inner)
            .collect(),
        skipped: catalog
            .skipped
            .into_iter()
            .map(|inner| PySkippedOmm { inner })
            .collect(),
    })
}

/// Parse NAVCEN GPS constellation status HTML into status rows.
///
/// Accepts the raw HTML as `str` or `bytes`. Returns rows sorted by PRN. Raises
/// `ConstellationError` if the bytes are not UTF-8, carry no GPS rows, or a
/// required integer cell fails to parse.
#[pyfunction]
fn parse_navcen(html: &Bound<'_, PyAny>) -> PyResult<Vec<PyNavcenStatus>> {
    let bytes: Vec<u8> = if let Ok(text) = html.extract::<String>() {
        text.into_bytes()
    } else if let Ok(raw) = html.extract::<Vec<u8>>() {
        raw
    } else {
        return Err(PyTypeError::new_err("parse_navcen expects str or bytes"));
    };
    constellation::parse_navcen(&bytes)
        .map(|statuses| {
            statuses
                .into_iter()
                .map(PyNavcenStatus::from_inner)
                .collect()
        })
        .map_err(to_constellation_err)
}

/// Merge NAVCEN status rows into CelesTrak records by PRN.
///
/// CelesTrak stays the identity base; a compatible NAVCEN row fills `svn`,
/// updates `usable`, and records provenance, while an incompatible row (a PRN
/// transition) is recorded under `source.navcen_conflict`. Returns records
/// sorted by PRN.
#[pyfunction]
fn merge_navcen(records: Vec<PyRecord>, statuses: Vec<PyNavcenStatus>) -> Vec<PyRecord> {
    let statuses: Vec<NavcenStatus> = statuses.into_iter().map(|s| s.inner).collect();
    constellation::merge_navcen(&records_inner(&records), &statuses)
        .into_iter()
        .map(PyRecord::from_inner)
        .collect()
}

/// Export records as the compact mapping CSV.
///
/// Header `prn,norad_cat_id,active,sp3_id`; the `active` column is `true` only
/// when a record is both active and usable. `booleans` selects how that column
/// renders: `"lower"` (`true`/`false`, the default) or `"title"` (`True`/`False`).
#[pyfunction]
#[pyo3(signature = (records, booleans="lower"))]
fn to_csv(records: Vec<PyRecord>, booleans: &str) -> PyResult<String> {
    let style = bool_style(booleans)?;
    Ok(constellation::to_csv(&records_inner(&records), style))
}

/// Validate catalog identity without an SP3 product.
#[pyfunction]
fn validate(records: Vec<PyRecord>) -> PyValidation {
    PyValidation::from_inner(constellation::validate(&records_inner(&records)))
}

/// Validate catalog identity against a list of SP3/RINEX satellite tokens.
#[pyfunction]
fn validate_against_sp3_ids(records: Vec<PyRecord>, ids: Vec<String>) -> PyValidation {
    let id_refs: Vec<&str> = ids.iter().map(String::as_str).collect();
    PyValidation::from_inner(constellation::validate_against_sp3_ids(
        &records_inner(&records),
        &id_refs,
    ))
}

#[pyfunction]
fn validate_against_sp3_ids_strict(records: Vec<PyRecord>, ids: Vec<String>) -> PyResult<()> {
    let id_refs: Vec<&str> = ids.iter().map(String::as_str).collect();
    constellation::validate_against_sp3_ids_strict(&records_inner(&records), &id_refs)
        .map_err(to_constellation_err)
}

/// Returns `True` when a validation report has no findings.
#[pyfunction]
fn is_valid(validation: &PyValidation) -> bool {
    constellation::is_valid(&validation.inner)
}

/// Compare two catalog snapshots by `(system, prn)` identity.
#[pyfunction]
fn diff(py: Python<'_>, previous: Vec<PyRecord>, current: Vec<PyRecord>) -> PyResult<PyDiff> {
    let report = constellation::diff(&records_inner(&previous), &records_inner(&current));
    PyDiff::build(py, &report)
}

/// Returns `True` when a diff has any findings.
#[pyfunction]
fn changed(diff: &PyDiff) -> bool {
    !diff.added.is_empty()
        || !diff.removed.is_empty()
        || !diff.norad_reassigned.is_empty()
        || !diff.sp3_id_changed.is_empty()
        || !diff.svn_changed.is_empty()
        || !diff.fdma_channel_changed.is_empty()
        || !diff.activity_changed.is_empty()
        || !diff.usability_changed.is_empty()
}

/// Render the canonical SP3/RINEX satellite token for a constellation + PRN
/// (`(GnssSystem.GPS, 7)` -> `G07`, `(GnssSystem.GLONASS, 13)` -> `R13`).
#[pyfunction]
fn gnss_sp3_id(system: PyGnssSystem, prn: u16) -> String {
    constellation::gnss_sp3_id(system.into(), prn)
}

/// Render the canonical SP3/RINEX satellite token for a GPS PRN (`7` -> `G07`).
///
/// Convenience wrapper over [`gnss_sp3_id`] for the GPS-only callers.
#[pyfunction]
fn gps_sp3_id(prn: u16) -> String {
    constellation::gnss_sp3_id(GnssSystem::Gps, prn)
}

/// The GLONASS FDMA L1/L2 frequency-channel number (`k`, in `-7..=6`) for an
/// orbital slot, or `None` if the slot has no published channel assignment.
#[pyfunction]
fn glonass_fdma_channel(slot: u16) -> Option<i8> {
    constellation::glonass_fdma_channel(slot)
}

#[pyfunction]
fn galileo_prn_for_gsat(gsat: u16) -> Option<u16> {
    constellation::galileo_prn_for_gsat(gsat)
}

#[pyfunction]
fn glonass_slot_for_number(number: u16) -> Option<u16> {
    constellation::glonass_slot_for_number(number)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCelestrakSource>()?;
    m.add_class::<PyNavcenSource>()?;
    m.add_class::<PyRecordSource>()?;
    m.add_class::<PyRecord>()?;
    m.add_class::<PyNavcenStatus>()?;
    m.add_class::<PyValidation>()?;
    m.add_class::<PyFieldChange>()?;
    m.add_class::<PyDiff>()?;
    m.add_class::<PySkippedOmm>()?;
    m.add_class::<PyCatalog>()?;
    m.add_function(wrap_pyfunction!(from_celestrak_json, m)?)?;
    m.add_function(wrap_pyfunction!(from_celestrak_omm_lenient, m)?)?;
    m.add_function(wrap_pyfunction!(parse_navcen, m)?)?;
    m.add_function(wrap_pyfunction!(merge_navcen, m)?)?;
    m.add_function(wrap_pyfunction!(to_csv, m)?)?;
    m.add_function(wrap_pyfunction!(validate, m)?)?;
    m.add_function(wrap_pyfunction!(validate_against_sp3_ids, m)?)?;
    m.add_function(wrap_pyfunction!(validate_against_sp3_ids_strict, m)?)?;
    m.add_function(wrap_pyfunction!(is_valid, m)?)?;
    m.add_function(wrap_pyfunction!(diff, m)?)?;
    m.add_function(wrap_pyfunction!(changed, m)?)?;
    m.add_function(wrap_pyfunction!(gnss_sp3_id, m)?)?;
    m.add_function(wrap_pyfunction!(gps_sp3_id, m)?)?;
    m.add_function(wrap_pyfunction!(glonass_fdma_channel, m)?)?;
    m.add_function(wrap_pyfunction!(galileo_prn_for_gsat, m)?)?;
    m.add_function(wrap_pyfunction!(glonass_slot_for_number, m)?)?;
    Ok(())
}
