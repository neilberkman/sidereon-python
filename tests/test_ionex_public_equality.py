"""Direct equality coverage for public IONEX values and query results."""

import sidereon


def _record(data, label):
    return f"{data:<60}{label:<20}\n"


def _band(latitude, values):
    prefix = _record(
        f"  {latitude:6.1f}{0.0:6.1f}{1.0:6.1f}{1.0:6.1f}{450.0:6.1f}",
        "LAT/LON1/LON2/DLON/H",
    )
    return prefix + "".join(f"{value:5d}" for value in values) + "\n"


def _ionex(north=(10, 9999), south=(30, 40), mapping="COSZ"):
    epoch = "  2020     1     1     0     0     0"
    lines = [
        _record("     1.0            IONOSPHERE MAPS     GPS", "IONEX VERSION / TYPE"),
        _record("public equality test", "PGM / RUN BY / DATE"),
        _record(epoch, "EPOCH OF FIRST MAP"),
        _record(epoch, "EPOCH OF LAST MAP"),
        _record("  3600", "INTERVAL"),
        _record("     1", "# OF MAPS IN FILE"),
        _record(f"  {mapping}", "MAPPING FUNCTION"),
        _record("     0.0", "ELEVATION CUTOFF"),
        _record("  6371.0", "BASE RADIUS"),
        _record("     2", "MAP DIMENSION"),
        _record("   450.0 450.0   0.0", "HGT1 / HGT2 / DHGT"),
        _record("     1.0   0.0  -1.0", "LAT1 / LAT2 / DLAT"),
        _record("     0.0   1.0   1.0", "LON1 / LON2 / DLON"),
        _record("     0", "EXPONENT"),
        _record("", "END OF HEADER"),
        _record("     1", "START OF TEC MAP"),
        _record(epoch, "EPOCH OF CURRENT MAP"),
        _band(1.0, north),
        _band(0.0, south),
        _record("     1", "END OF TEC MAP"),
        _record("", "END OF FILE"),
    ]
    return "".join(lines).encode("ascii")


def _grid(north=(10, 9999), south=(30, 40), mapping="COSZ"):
    return sidereon.load_ionex(_ionex(north, south, mapping))


def _renormalized_evaluation(grid, policy):
    epoch = int(grid.map_epochs_j2000_s[0])
    return grid.slant_delay_with_policy(
        0.5,
        0.5,
        0.0,
        90.0,
        epoch,
        1575420000.0,
        policy=policy,
    )


def _assert_eq_and_ne(equal_left, equal_again, different):
    # These wrappers define __eq__ and rely on Python's normal __ne__ fallback.
    assert equal_left == equal_again
    assert not (equal_left != equal_again)
    assert equal_left != different
    assert not (equal_left == different)


def test_public_ionex_value_and_query_result_equality_protocol():
    source = _ionex()
    grid = sidereon.load_ionex(source)
    repeated_grid = sidereon.load_ionex(source)
    other_grid = _grid(mapping="QFAC")
    different_missing_grid = _grid(north=(9999, 20))

    _assert_eq_and_ne(grid.header, repeated_grid.header, other_grid.header)
    _assert_eq_and_ne(
        grid.header.mapping_function,
        repeated_grid.header.mapping_function,
        other_grid.header.mapping_function,
    )
    _assert_eq_and_ne(
        grid.header.mapping_declaration,
        repeated_grid.header.mapping_declaration,
        other_grid.header.mapping_declaration,
    )

    strict_policy = sidereon.IonexSlantPolicy()
    renormalize_policy = strict_policy.with_missing_nodes(
        sidereon.IonexMissingNodePolicy.RENORMALIZE
    )
    repeated_renormalize_policy = sidereon.IonexSlantPolicy().with_missing_nodes(
        sidereon.IonexMissingNodePolicy.RENORMALIZE
    )
    _assert_eq_and_ne(
        renormalize_policy,
        repeated_renormalize_policy,
        strict_policy,
    )

    evaluation = _renormalized_evaluation(grid, renormalize_policy)
    repeated_evaluation = _renormalized_evaluation(
        repeated_grid, repeated_renormalize_policy
    )
    different_evaluation = _renormalized_evaluation(
        different_missing_grid, renormalize_policy
    )
    _assert_eq_and_ne(evaluation, repeated_evaluation, different_evaluation)
    _assert_eq_and_ne(
        evaluation.status,
        repeated_evaluation.status,
        different_evaluation.status,
    )

    gap = evaluation.status.degraded
    repeated_gap = repeated_evaluation.status.degraded
    different_gap = different_evaluation.status.degraded
    assert gap is not None and repeated_gap is not None and different_gap is not None
    _assert_eq_and_ne(gap, repeated_gap, different_gap)

    missing_nodes = gap.earlier
    repeated_missing_nodes = repeated_gap.earlier
    different_missing_nodes = different_gap.earlier
    assert (
        missing_nodes is not None
        and repeated_missing_nodes is not None
        and different_missing_nodes is not None
    )
    _assert_eq_and_ne(
        missing_nodes,
        repeated_missing_nodes,
        different_missing_nodes,
    )
