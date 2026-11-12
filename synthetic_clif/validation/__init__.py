"""Integrity checks for a generated CLIF dataset (2.1 or 3.0).

``check_dataset(tables, clif_version)`` returns a list of issues; an empty list
means the dataset passed. The checks go beyond column names and vocabularies
(see ``synthetic_clif.clif3.validate``) to the rules a CLIF site's own data must
meet, taken from the consortium validator clifpy and vendored in
``clif_rules.json`` (rebuild with ``tools/build_clif_rules.py``):

* ``schema``          3.0 only: columns, dtypes, vocabularies, category groups
* ``primary_key``     duplicate rows on a table's composite key
* ``foreign_key``     ids that point at no row of the referenced table, and
                      patient_id / hospitalization_id pairs that disagree with
                      the hospitalization table
* ``chronology``      an end before its start (admission/discharge, in/out,
                      order/collect/result, placed/removed, ...)
* ``overlap``         overlapping ADT stays, or overlapping hospitalizations of
                      one patient
* ``stay_window``     a timestamp outside its hospitalization, or after death
* ``value_range``     a number outside clifpy's outlier range for its column
                      and category (an impossible value, not just an abnormal one)
* ``units``           a lab unit that is not the CLIF unit for its lab category,
                      or a medication dose unit of the wrong kind (rate vs dose)
* ``death``           an expired discharge with no patient death time, or a
                      death time that does not match that discharge
* ``never_charted``   a setting CLIF requires for a device (say, flow for high
                      flow nasal cannula) that is missing from every such row

Adding a table or a column: the generic checks key on column names
(``*_id`` foreign keys, ``*_dttm`` timestamps, start/end name pairs), so most
new tables are covered with no change. Table-specific rules go in the
``EXTRA_*`` dicts below, and a new kind of check can be added with
``@register_check("name")``.
"""

from synthetic_clif.validation.integrity import (  # noqa: F401
    CHECKS,
    EXTRA_COMPOSITE_KEYS,
    EXTRA_FOREIGN_KEYS,
    EXTRA_ORDER,
    EXTRA_RANGES,
    STAY_WINDOW_ALLOWANCE,
    check_dataset,
    format_issues,
    range_coverage,
    register_check,
)
