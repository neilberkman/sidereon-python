//! Structural validation for legacy Unix `compress` (`.Z`) archives.
//!
//! Actual decompression remains in Python's `ncompress` dependency, where its
//! output is written through a size-bounded sink. This scanner emits no output
//! and builds no LZW dictionary: it only follows code widths, CLEAR resets, and
//! ncompress code-group alignment so an incomplete terminal code cannot be
//! mistaken for a successful end of stream.
//!
//! The code-width and group-alignment algorithm is adapted from
//! `newtua-lzw-z` 0.1.0's `src/decode.rs`, Copyright (c) 2026 Aleksei Trankov,
//! used under its MIT license option. See `THIRD-PARTY-NOTICES.md`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyModule};

const MAGIC: [u8; 2] = [0x1f, 0x9d];
const BLOCK_MODE_FLAG: u8 = 0x80;
const MAXBITS_MASK: u8 = 0x1f;
const INIT_BITS: u32 = 9;
const MAX_MAXBITS: u32 = 16;
const CLEAR: u32 = 256;

#[derive(Debug, PartialEq, Eq)]
enum ValidationError {
    BadMagic,
    BadMaxBits,
    InvalidCode,
    Truncated,
}

fn read_code(data: &[u8], total_bits: u64, bitpos: u64, width: u32) -> Option<u32> {
    let end = bitpos.checked_add(u64::from(width))?;
    if end > total_bits {
        return None;
    }

    // A code is at most 16 bits and can straddle at most three bytes. Assemble
    // that fixed-size little-endian window instead of paying a loop per bit.
    let byte_index = (bitpos >> 3) as usize;
    let shift = (bitpos & 7) as u32;
    let mut window = u32::from(data[byte_index]);
    if byte_index + 1 < data.len() {
        window |= u32::from(data[byte_index + 1]) << 8;
    }
    if byte_index + 2 < data.len() {
        window |= u32::from(data[byte_index + 2]) << 16;
    }
    Some((window >> shift) & ((1u32 << width) - 1))
}

fn align_to_group(bitpos: u64, width: u32, boundary_offset: u64) -> u64 {
    debug_assert!(bitpos > 0, "alignment requires a prior code read");
    let group_bits = u64::from(width) * 8;
    let previous = bitpos - 1;
    let relative = previous - boundary_offset;
    let padding = group_bits - (relative + group_bits) % group_bits;
    previous + padding
}

fn validate_terminal_padding(
    data: &[u8],
    total_bits: u64,
    bitpos: u64,
) -> Result<(), ValidationError> {
    let remaining = total_bits
        .checked_sub(bitpos)
        .ok_or(ValidationError::Truncated)?;
    if remaining > 7 {
        return Err(ValidationError::Truncated);
    }

    for position in bitpos..total_bits {
        if ((data[(position >> 3) as usize] >> (position & 7)) & 1) != 0 {
            return Err(ValidationError::Truncated);
        }
    }
    Ok(())
}

fn validate_codes(data: &[u8], block_mode: bool, maxbits: u32) -> Result<(), ValidationError> {
    let total_bits = u64::try_from(data.len())
        .ok()
        .and_then(|len| len.checked_mul(8))
        .ok_or(ValidationError::Truncated)?;
    if total_bits == 0 {
        // ncompress 1.0.2 emits a header-only archive for empty input.
        return Ok(());
    }

    let max_code_count = 1u32 << maxbits;
    let first_free = if block_mode { CLEAR + 1 } else { 256 };
    let mut free_entry = first_free;
    let mut width = INIT_BITS;
    let mut max_code = (1u32 << width) - 1;
    let mut bitpos = 0u64;
    let mut boundary_offset = 0u64;

    let first = read_code(data, total_bits, bitpos, width).ok_or(ValidationError::Truncated)?;
    bitpos += u64::from(width);
    if first >= 256 {
        return Err(ValidationError::InvalidCode);
    }

    while let Some(code) = read_code(data, total_bits, bitpos, width) {
        bitpos += u64::from(width);

        if block_mode && code == CLEAR {
            let next = align_to_group(bitpos, width, boundary_offset);
            boundary_offset = next;
            bitpos = next;
            free_entry = CLEAR + 1;
            width = INIT_BITS;
            max_code = (1u32 << width) - 1;

            let literal =
                read_code(data, total_bits, bitpos, width).ok_or(ValidationError::Truncated)?;
            bitpos += u64::from(width);
            if literal >= 256 {
                return Err(ValidationError::InvalidCode);
            }
            continue;
        }

        if code > free_entry {
            return Err(ValidationError::InvalidCode);
        }

        if free_entry < max_code_count {
            free_entry += 1;
            if free_entry > max_code && width < maxbits {
                let next = align_to_group(bitpos, width, boundary_offset);
                boundary_offset = next;
                bitpos = next;
                width += 1;
                max_code = if width == maxbits {
                    max_code_count
                } else {
                    (1u32 << width) - 1
                };
            }
        }
    }

    validate_terminal_padding(data, total_bits, bitpos)
}

fn validate_archive(archive: &[u8]) -> Result<(), ValidationError> {
    // Some Unix compress implementations represent empty input as an empty
    // stream, while ncompress emits the three-byte header form. Accept both;
    // product validation remains responsible for rejecting an empty product.
    if archive.is_empty() {
        return Ok(());
    }
    if archive.len() < 2 {
        return Err(ValidationError::Truncated);
    }
    if archive[..2] != MAGIC {
        return Err(ValidationError::BadMagic);
    }
    if archive.len() < 3 {
        return Err(ValidationError::Truncated);
    }

    let flags = archive[2];
    let maxbits = u32::from(flags & MAXBITS_MASK);
    if !(INIT_BITS..=MAX_MAXBITS).contains(&maxbits) {
        return Err(ValidationError::BadMaxBits);
    }

    validate_codes(&archive[3..], (flags & BLOCK_MODE_FLAG) != 0, maxbits)
}

#[pyfunction]
fn _validate_unix_compress(archive: &Bound<'_, PyBytes>) -> PyResult<()> {
    validate_archive(archive.as_bytes())
        .map_err(|_| PyValueError::new_err("invalid or truncated Unix-compress product"))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_validate_unix_compress, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn literal_archive(bytes: &[u8]) -> Vec<u8> {
        let mut archive = vec![MAGIC[0], MAGIC[1], MAX_MAXBITS as u8];
        let mut codes = vec![0u8; (bytes.len() * INIT_BITS as usize).div_ceil(8)];

        for (code_index, byte) in bytes.iter().enumerate() {
            for bit_index in 0..INIT_BITS as usize {
                let bit = (usize::from(*byte) >> bit_index) & 1;
                let position = code_index * INIT_BITS as usize + bit_index;
                codes[position / 8] |= (bit as u8) << (position % 8);
            }
        }

        archive.extend(codes);
        archive
    }

    #[test]
    fn complete_literal_archives_and_empty_archive_are_valid() {
        assert_eq!(validate_archive(&[]), Ok(()));
        assert_eq!(validate_archive(&literal_archive(b"")), Ok(()));
        assert_eq!(validate_archive(&literal_archive(b"A")), Ok(()));
        assert_eq!(validate_archive(&literal_archive(b"AB")), Ok(()));
    }

    #[test]
    fn incomplete_codes_and_nonzero_terminal_padding_are_invalid() {
        assert_eq!(
            validate_archive(&[MAGIC[0], MAGIC[1], MAX_MAXBITS as u8, b'A']),
            Err(ValidationError::Truncated)
        );

        let two_literals = literal_archive(b"AB");
        assert_eq!(
            validate_archive(&two_literals[..two_literals.len() - 1]),
            Err(ValidationError::Truncated)
        );

        let mut nonzero_padding = literal_archive(b"A");
        *nonzero_padding.last_mut().unwrap() |= 0x80;
        assert_eq!(
            validate_archive(&nonzero_padding),
            Err(ValidationError::Truncated)
        );
    }

    #[test]
    fn malformed_header_and_first_code_are_invalid() {
        assert_eq!(validate_archive(&[0x1f]), Err(ValidationError::Truncated));
        assert_eq!(
            validate_archive(&[0x00, 0x01, 0x10]),
            Err(ValidationError::BadMagic)
        );
        assert_eq!(
            validate_archive(&[MAGIC[0], MAGIC[1], 0x08]),
            Err(ValidationError::BadMaxBits)
        );
        assert_eq!(
            validate_archive(&literal_archive(&[0xff, 0x01])[..4]),
            Err(ValidationError::Truncated)
        );
    }
}
