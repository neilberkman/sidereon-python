use flate2::read::GzDecoder;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs;
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};

const REGISTRY_SOURCE: &str = "registry+https://github.com/rust-lang/crates.io-index";
const GIT_SOURCE_PREFIX: &str = "git+https://github.com/neilberkman/sidereon?rev=";
const RELEASE_VERSION: &str = "3.0.0";

#[derive(Debug)]
struct LockedPackage {
    name: String,
    version: String,
    source: Option<String>,
    checksum: Option<String>,
}

/// Resolve the paired engine revision from this generator's locked source.
/// Registry packages are tied to Cargo.lock by hashing the actual cached
/// `.crate` archive before its published manifest and VCS metadata are read.
pub fn revision_from_lock(lock: &str) -> Result<String, String> {
    let packages = lock
        .split("[[package]]")
        .skip(1)
        .map(parse_package)
        .collect::<Vec<_>>();
    let facade = unique_package(&packages, "sidereon")?;
    let core = unique_package(&packages, "sidereon-core")?;
    if facade.version != RELEASE_VERSION || core.version != RELEASE_VERSION {
        return Err(format!(
            "sidereon and sidereon-core must both be {RELEASE_VERSION}; locked {} and {}",
            facade.version, core.version
        ));
    }
    if facade.source != core.source {
        return Err("sidereon and sidereon-core have different locked sources".to_owned());
    }
    let source = facade
        .source
        .as_deref()
        .ok_or_else(|| "sidereon source is missing from Cargo.lock".to_owned())?;
    if source.starts_with("git+") {
        let facade_revision = git_revision(source)?;
        let core_revision = git_revision(
            core.source
                .as_deref()
                .ok_or_else(|| "sidereon-core source is missing from Cargo.lock".to_owned())?,
        )?;
        if facade_revision != core_revision {
            return Err("sidereon and sidereon-core resolve to different Git commits".to_owned());
        }
        return Ok(facade_revision);
    }
    if source != REGISTRY_SOURCE {
        return Err(format!("unsupported sidereon source {source:?}"));
    }
    let facade_checksum = checked_checksum(&facade)?;
    let core_checksum = checked_checksum(&core)?;
    let cargo_home = cargo_home()?;
    let facade_revision =
        registry_revision(&cargo_home, &facade.name, &facade.version, &facade_checksum)?;
    let core_revision = registry_revision(&cargo_home, &core.name, &core.version, &core_checksum)?;
    if facade_revision != core_revision {
        return Err("published sidereon and sidereon-core VCS commits differ".to_owned());
    }
    Ok(facade_revision)
}

fn parse_package(block: &str) -> LockedPackage {
    LockedPackage {
        name: field(block, "name").unwrap_or_default(),
        version: field(block, "version").unwrap_or_default(),
        source: field(block, "source"),
        checksum: field(block, "checksum"),
    }
}

fn field(block: &str, name: &str) -> Option<String> {
    block.lines().find_map(|line| {
        let (key, value) = line.split_once('=')?;
        if key.trim() != name {
            return None;
        }
        let value = value.trim();
        value
            .strip_prefix('"')
            .and_then(|value| value.strip_suffix('"'))
            .map(str::to_owned)
    })
}

fn unique_package<'a>(
    packages: &'a [LockedPackage],
    name: &str,
) -> Result<&'a LockedPackage, String> {
    let matches = packages
        .iter()
        .filter(|package| package.name == name)
        .collect::<Vec<_>>();
    if matches.len() != 1 {
        return Err(format!(
            "expected exactly one {name} package in Cargo.lock, found {}",
            matches.len()
        ));
    }
    Ok(matches[0])
}

fn git_revision(source: &str) -> Result<String, String> {
    let revision = source
        .strip_prefix(GIT_SOURCE_PREFIX)
        .ok_or_else(|| format!("noncanonical Git source {source:?}"))?;
    let (query, fragment) = revision
        .split_once('#')
        .ok_or_else(|| format!("Git source has no resolved commit: {source:?}"))?;
    if !is_hex(query, 40) || query != fragment {
        return Err(format!(
            "Git source does not lock one full commit: {source:?}"
        ));
    }
    Ok(query.to_owned())
}

fn checked_checksum(package: &LockedPackage) -> Result<String, String> {
    let checksum = package
        .checksum
        .as_deref()
        .ok_or_else(|| format!("{} registry lock has no checksum", package.name))?;
    if !is_hex(checksum, 64) {
        return Err(format!("{} registry checksum is malformed", package.name));
    }
    Ok(checksum.to_owned())
}

fn registry_revision(
    cargo_home: &Path,
    name: &str,
    version: &str,
    checksum: &str,
) -> Result<String, String> {
    let cache_root = cargo_home.join("registry").join("cache");
    let registries = fs::read_dir(&cache_root).map_err(|error| {
        format!(
            "read Cargo registry archives {}: {error}",
            cache_root.display()
        )
    })?;
    let archive_name = format!("{name}-{version}.crate");
    let mut revisions = BTreeSet::new();
    for registry in registries {
        let registry = registry.map_err(|error| format!("read Cargo registry source: {error}"))?;
        let archive_path = registry.path().join(&archive_name);
        if !archive_path.is_file() {
            continue;
        }
        let archive = fs::read(&archive_path)
            .map_err(|error| format!("read {}: {error}", archive_path.display()))?;
        let actual_checksum = format!("{:x}", Sha256::digest(&archive));
        if actual_checksum != checksum {
            continue;
        }
        let members = verified_archive_members(&archive, name, version)?;
        if !manifest_matches(&members.manifest, name, version)? {
            return Err(format!("verified registry archive is not {name} {version}"));
        }
        let vcs_doc: Value = serde_json::from_slice(&members.vcs)
            .map_err(|error| format!("parse published {name} {version} VCS metadata: {error}"))?;
        let revision = vcs_doc
            .get("git")
            .and_then(|git| git.get("sha1"))
            .and_then(Value::as_str)
            .ok_or_else(|| format!("published {name} {version} has no Git commit"))?;
        if !is_hex(revision, 40) {
            return Err(format!(
                "published {name} {version} has a malformed Git commit"
            ));
        }
        if vcs_doc
            .get("git")
            .and_then(|git| git.get("dirty"))
            .and_then(Value::as_bool)
            == Some(true)
        {
            return Err(format!("published {name} {version} has a dirty Git source"));
        }
        revisions.insert(revision.to_owned());
    }
    match revisions.len() {
        1 => Ok(revisions.into_iter().next().expect("one revision")),
        0 => Err(format!(
            "no cached .crate archive for {name} {version} matches lock checksum {checksum}"
        )),
        _ => Err(format!(
            "Cargo caches disagree on the published Git commit for {name} {version}"
        )),
    }
}

fn manifest_matches(
    bytes: &[u8],
    expected_name: &str,
    expected_version: &str,
) -> Result<bool, String> {
    let manifest = std::str::from_utf8(bytes)
        .map_err(|error| format!("published Cargo.toml is not UTF-8: {error}"))?;
    let mut in_package = false;
    let mut name = None;
    let mut version = None;
    for line in manifest.lines() {
        let line = line.trim();
        if line.starts_with('[') && line.ends_with(']') {
            in_package = line == "[package]";
        } else if in_package {
            if let Some(value) = line.strip_prefix("name = ") {
                name = Some(value.trim_matches('"').to_owned());
            } else if let Some(value) = line.strip_prefix("version = ") {
                version = Some(value.trim_matches('"').to_owned());
            }
        }
    }
    Ok(name.as_deref() == Some(expected_name) && version.as_deref() == Some(expected_version))
}

struct ArchiveMembers {
    manifest: Vec<u8>,
    vcs: Vec<u8>,
}

fn verified_archive_members(
    archive: &[u8],
    name: &str,
    version: &str,
) -> Result<ArchiveMembers, String> {
    let mut decoder = GzDecoder::new(Cursor::new(archive));
    let mut tar = Vec::new();
    decoder
        .read_to_end(&mut tar)
        .map_err(|error| format!("decompress verified {name} {version} archive: {error}"))?;
    let prefix = format!("{name}-{version}/");
    let mut manifest = None;
    let mut vcs = None;
    let mut offset = 0usize;
    while offset + 512 <= tar.len() {
        let header = &tar[offset..offset + 512];
        if header.iter().all(|byte| *byte == 0) {
            break;
        }
        let member_path = tar_path(header)?;
        let size = tar_octal(&header[124..136])?;
        let data_start = offset + 512;
        let data_end = data_start
            .checked_add(size)
            .filter(|end| *end <= tar.len())
            .ok_or_else(|| "truncated member in verified Cargo archive".to_owned())?;
        if let Some(relative) = member_path.strip_prefix(&prefix) {
            match relative {
                "Cargo.toml" => {
                    set_once(&mut manifest, tar[data_start..data_end].to_vec(), relative)?
                }
                ".cargo_vcs_info.json" => {
                    set_once(&mut vcs, tar[data_start..data_end].to_vec(), relative)?
                }
                _ => {}
            }
        }
        offset = data_start + size.div_ceil(512) * 512;
    }
    Ok(ArchiveMembers {
        manifest: manifest.ok_or_else(|| "verified archive has no Cargo.toml".to_owned())?,
        vcs: vcs.ok_or_else(|| "verified archive has no .cargo_vcs_info.json".to_owned())?,
    })
}

fn tar_path(header: &[u8]) -> Result<String, String> {
    let name = tar_text(&header[..100])?;
    let prefix = tar_text(&header[345..500])?;
    Ok(if prefix.is_empty() {
        name
    } else {
        format!("{prefix}/{name}")
    })
}

fn tar_text(bytes: &[u8]) -> Result<String, String> {
    let bytes = bytes.split(|byte| *byte == 0).next().unwrap_or_default();
    String::from_utf8(bytes.to_vec()).map_err(|error| format!("invalid UTF-8 in tar path: {error}"))
}

fn tar_octal(bytes: &[u8]) -> Result<usize, String> {
    let text = bytes
        .split(|byte| *byte == 0 || *byte == b' ')
        .next()
        .unwrap_or_default();
    let text = std::str::from_utf8(text).map_err(|error| format!("invalid tar size: {error}"))?;
    usize::from_str_radix(text.trim(), 8).map_err(|error| format!("invalid tar size: {error}"))
}

fn set_once(slot: &mut Option<Vec<u8>>, value: Vec<u8>, member: &str) -> Result<(), String> {
    if slot.replace(value).is_some() {
        return Err(format!("verified archive contains duplicate {member}"));
    }
    Ok(())
}

fn cargo_home() -> Result<PathBuf, String> {
    if let Some(path) = std::env::var_os("CARGO_HOME") {
        return Ok(PathBuf::from(path));
    }
    if let Some(path) = std::env::var_os("HOME") {
        return Ok(PathBuf::from(path).join(".cargo"));
    }
    if let Some(path) = std::env::var_os("USERPROFILE") {
        return Ok(PathBuf::from(path).join(".cargo"));
    }
    Err("CARGO_HOME, HOME, and USERPROFILE are all unset".to_owned())
}

fn is_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

    #[test]
    fn registry_revision_reads_vcs_from_checksum_verified_archive() {
        let temp = std::env::temp_dir().join(format!(
            "sidereon-source-identity-{}-{}",
            std::process::id(),
            NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        ));
        let cache = temp.join("registry/cache/test-index");
        let extracted = temp.join("registry/src/test-index/sidereon-3.0.0");
        fs::create_dir_all(&cache).expect("create registry cache");
        fs::create_dir_all(&extracted).expect("create extracted source");
        fs::write(
            extracted.join(".cargo_vcs_info.json"),
            r#"{"git":{"sha1":"ffffffffffffffffffffffffffffffffffffffff"}}"#,
        )
        .expect("write deliberately altered extracted VCS file");

        let revision = "0123456789abcdef0123456789abcdef01234567";
        let archive = crate_archive(
            "sidereon",
            "3.0.0",
            "[package]\nname = \"sidereon\"\nversion = \"3.0.0\"\n",
            &format!(r#"{{"git":{{"sha1":"{revision}","dirty":false}}}}"#),
        );
        let checksum = format!("{:x}", Sha256::digest(&archive));
        fs::write(cache.join("sidereon-3.0.0.crate"), archive).expect("write crate archive");
        assert_eq!(
            registry_revision(&temp, "sidereon", "3.0.0", &checksum).unwrap(),
            revision
        );
        assert!(registry_revision(&temp, "sidereon", "3.0.0", &"f".repeat(64)).is_err());
        fs::remove_dir_all(temp).expect("remove registry test directory");
    }

    fn crate_archive(name: &str, version: &str, manifest: &str, vcs: &str) -> Vec<u8> {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        append_tar_member(
            &mut encoder,
            &format!("{name}-{version}/Cargo.toml"),
            manifest.as_bytes(),
        );
        append_tar_member(
            &mut encoder,
            &format!("{name}-{version}/.cargo_vcs_info.json"),
            vcs.as_bytes(),
        );
        encoder.write_all(&[0; 1024]).expect("write tar terminator");
        encoder.finish().expect("finish gzip archive")
    }

    fn append_tar_member(encoder: &mut GzEncoder<Vec<u8>>, path: &str, contents: &[u8]) {
        let mut header = [0u8; 512];
        header[..path.len()].copy_from_slice(path.as_bytes());
        write_octal(&mut header[100..108], 0o644);
        write_octal(&mut header[108..116], 0);
        write_octal(&mut header[116..124], 0);
        write_octal(&mut header[124..136], contents.len());
        write_octal(&mut header[136..148], 0);
        header[148..156].fill(b' ');
        header[156] = b'0';
        header[257..263].copy_from_slice(b"ustar\0");
        header[263..265].copy_from_slice(b"00");
        let checksum = header.iter().map(|byte| usize::from(*byte)).sum();
        write_octal(&mut header[148..156], checksum);
        encoder.write_all(&header).expect("write tar header");
        encoder.write_all(contents).expect("write tar data");
        let padding = (512 - contents.len() % 512) % 512;
        encoder
            .write_all(&vec![0; padding])
            .expect("write tar padding");
    }

    fn write_octal(field: &mut [u8], value: usize) {
        let digits = format!("{value:o}");
        let start = field.len() - digits.len() - 1;
        field.fill(b'0');
        field[start..start + digits.len()].copy_from_slice(digits.as_bytes());
        field[field.len() - 1] = 0;
    }
}
