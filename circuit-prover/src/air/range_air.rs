//! [`RangeAir`] defines the AIR for the range-check NPO table: it proves every looked-up value
//! lies in `[0, 256)` via a LogUp lookup against a fixed 256-row byte table.
//!
//! Each row is either a **lookup row** or a **table row**, selected by the committed preprocessed
//! columns:
//!
//! - Lookup rows (`is_lookup = 1`) come from `add_range` calls. The `WitnessChecks` bus binds the
//!   `value` main columns to the checked witness id (reader), and `RangeBus` gets one `+1` send of
//!   `value_0`.
//! - Table rows (`is_table = 1`, `table_const = 0..=255`) are appended once, globally, by
//!   [`super::super::batch_stark_prover::RangePreprocessor`] (preprocessed) and
//!   [`RangeAir::trace_to_matrix`] (main, carrying the per-byte multiplicity witness). Each table
//!   row gets one `-mult` receive of `table_const` on `RangeBus`.
//!
//! The framework's LogUp gadget asserts `RangeBus` balances to zero across all rows:
//! `Σ_lookups value_0 == Σ_table mult · table_const` as a rational identity. Because the 256
//! `table_const` values are distinct committed constants `0..=255`, this holds iff every looked-up
//! `value_0` equals one of them — i.e. lies in `[0, 256)` — with `mult` forced to the exact count.
//!
//! # Column layout (lane = 1)
//!
//! Main columns (`D + 1`): `[value_0, …, value_{D-1}, mult]`
//! - `value_j` — checked value's `j`-th base-field coefficient (lookup rows); `0` on table rows.
//! - `mult`    — per-byte lookup multiplicity (table rows); `0` on lookup/padding rows.
//!
//! Preprocessed columns (`4`):
//! - col 0 `witness_idx` — checked witness base-field index (lookup rows); `0` on table rows.
//! - col 1 `is_lookup`   — `1` on lookup rows, else `0`.
//! - col 2 `table_const` — byte value `0..=255` on table rows, else `0`.
//! - col 3 `is_table`    — `1` on table rows, else `0`.

use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
use p3_lookup::builder::InteractionBuilder;
use p3_matrix::dense::RowMajorMatrix;
use tracing::instrument;

/// Number of entries in the byte table.
pub(crate) const RANGE_TABLE_SIZE: usize = 256;

/// Per-row preprocessed width for [`RangeAir`].
pub(crate) const RANGE_PREP_LANE_WIDTH: usize = 4;
/// Position of `witness_idx` within a preprocessed row.
pub(crate) const RANGE_PREP_WITNESS_IDX: usize = 0;
/// Position of `is_lookup` within a preprocessed row.
pub(crate) const RANGE_PREP_IS_LOOKUP: usize = 1;
/// Position of `table_const` within a preprocessed row.
pub(crate) const RANGE_PREP_TABLE_CONST: usize = 2;
/// Position of `is_table` within a preprocessed row.
pub(crate) const RANGE_PREP_IS_TABLE: usize = 3;

/// AIR for the range-check (byte-table lookup) table.
#[derive(Debug, Clone)]
pub struct RangeAir<F, const D: usize> {
    pub(crate) lanes: usize,
    pub(crate) preprocessed: Vec<F>,
    pub(crate) min_height: usize,
    _phantom: PhantomData<F>,
}

impl<F: PrimeField64, const D: usize> RangeAir<F, D> {
    /// Main trace width per lane: `D` value coeffs + 1 multiplicity column.
    pub const fn lane_width() -> usize {
        D + 1
    }

    /// Preprocessed width per lane: 4 (witness_idx, is_lookup, table_const, is_table).
    pub const fn preprocessed_lane_width() -> usize {
        RANGE_PREP_LANE_WIDTH
    }

    pub fn new_with_preprocessed(lanes: usize, preprocessed: Vec<F>, min_height: usize) -> Self {
        Self {
            lanes: lanes.max(1),
            preprocessed,
            min_height,
            _phantom: PhantomData,
        }
    }

    /// Build the main trace matrix: all lookup entries (one per checked value, `mult = 0`)
    /// followed by 256 table entries (`value = 0`, `mult = count of that byte among lookups`).
    ///
    /// `op.input_values` holds the base-field coefficient layout: `D` consecutive base elements
    /// per logical checked value (`generate_range_trace::<BF, EF>` produces this). Only coefficient
    /// 0 is histogrammed; the higher coefficients are forced to zero on lookup rows by the AIR.
    ///
    /// Entry order MUST match the committed preprocessed built by `RangeAir`'s op `preprocess`
    /// (lookups, in op order) + `RangePreprocessor` (256 table rows) so main/preprocessed rows
    /// align.
    #[instrument(skip_all, name = "RangeAir::build_trace")]
    pub fn trace_to_matrix(
        ops: &[p3_circuit::ops::range::RangeCircuitRow<F>],
        lanes: usize,
    ) -> RowMajorMatrix<F> {
        let lanes = lanes.max(1);
        let lane_w = Self::lane_width();
        let row_width = lanes * lane_w;

        // Collect lookup entries and histogram their coefficient-0 value.
        let mut counts = [0u64; RANGE_TABLE_SIZE];
        let mut lookup_coeffs: Vec<[F; D]> = Vec::new();
        for op in ops {
            let n = op.input_wids.len();
            debug_assert_eq!(
                op.input_values.len(),
                n * D,
                "RangeCircuitRow.input_values must hold D base-field coefficients per checked value"
            );
            for k in 0..n {
                let mut coeffs = [F::ZERO; D];
                for j in 0..D {
                    coeffs[j] = op.input_values[k * D + j];
                }
                let v0 = coeffs[0].as_canonical_u64();
                if v0 < RANGE_TABLE_SIZE as u64 {
                    counts[v0 as usize] += 1;
                }
                lookup_coeffs.push(coeffs);
            }
        }

        let num_entries = lookup_coeffs.len() + RANGE_TABLE_SIZE;
        let num_rows = num_entries.div_ceil(lanes).max(1);
        let mut values = F::zero_vec(num_rows * row_width);

        // Lookup entries: [value_0..value_{D-1}, mult = 0].
        for (idx, coeffs) in lookup_coeffs.iter().enumerate() {
            let r = idx / lanes;
            let l = idx % lanes;
            let base = r * row_width + l * lane_w;
            for j in 0..D {
                values[base + j] = coeffs[j];
            }
            // mult column (base + D) stays zero.
        }

        // Table entries: [0.., mult = count(byte)].
        for b in 0..RANGE_TABLE_SIZE {
            let idx = lookup_coeffs.len() + b;
            let r = idx / lanes;
            let l = idx % lanes;
            let base = r * row_width + l * lane_w;
            // value columns stay zero.
            values[base + D] = F::from_u64(counts[b]);
        }

        let mut mat = RowMajorMatrix::new(values, row_width);
        mat.pad_to_power_of_two_height(F::ZERO);
        mat
    }
}

impl<F: Field, const D: usize> BaseAir<F> for RangeAir<F, D> {
    fn width(&self) -> usize {
        self.lanes * (D + 1)
    }

    fn preprocessed_width(&self) -> usize {
        self.lanes * RANGE_PREP_LANE_WIDTH
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let width = self.lanes * RANGE_PREP_LANE_WIDTH;
        let mut mat = RowMajorMatrix::from_flat_padded(self.preprocessed.to_vec(), width, F::ZERO);
        mat.pad_to_min_power_of_two_height(self.min_height, F::ZERO);
        Some(mat)
    }

    fn main_next_row_columns(&self) -> Vec<usize> {
        vec![]
    }

    fn preprocessed_next_row_columns(&self) -> Vec<usize> {
        vec![]
    }
}

impl<AB: AirBuilder + InteractionBuilder, const D: usize> Air<AB> for RangeAir<AB::F, D>
where
    AB::F: Field,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let main_local = main.current_slice();
        let prep = builder.preprocessed().clone();
        let prep_local = prep.current_slice();

        let lane_w = D + 1;
        let prep_lane_w = RANGE_PREP_LANE_WIDTH;

        for lane in 0..self.lanes {
            let main_off = lane * lane_w;
            let prep_off = lane * prep_lane_w;

            let witness_idx: AB::Expr = prep_local[prep_off + RANGE_PREP_WITNESS_IDX].into();
            let is_lookup: AB::Expr = prep_local[prep_off + RANGE_PREP_IS_LOOKUP].into();
            let table_const: AB::Expr = prep_local[prep_off + RANGE_PREP_TABLE_CONST].into();
            let is_table: AB::Expr = prep_local[prep_off + RANGE_PREP_IS_TABLE].into();
            let mult: AB::Expr = main_local[main_off + D].into();

            // ---- Local: higher coefficients of the checked value are zero on lookup rows ----
            for j in 1..D {
                builder.assert_zero(is_lookup.clone() * main_local[main_off + j].into());
            }

            // ---- Local: multiplicity is zero on non-table rows (lookups + padding) ----
            // Prevents laundering negative RangeBus mass on non-table rows.
            builder.assert_zero((AB::Expr::ONE - is_table.clone()) * mult.clone());

            // ---- WitnessChecks: bind value to the checked witness (reader) on lookup rows ----
            let mut wc_fields: Vec<AB::Expr> = Vec::with_capacity(1 + D);
            wc_fields.push(witness_idx);
            for j in 0..D {
                wc_fields.push(main_local[main_off + j].into());
            }
            builder.push_interaction("WitnessChecks", wc_fields, -is_lookup.clone(), 1);

            // ---- RangeBus: lookup sends +is_lookup for value_0 ----
            builder.push_interaction(
                "RangeBus",
                vec![main_local[main_off].into()],
                is_lookup,
                1,
            );

            // ---- RangeBus: table receives -mult for table_const ----
            builder.push_interaction("RangeBus", vec![table_const], -mult, 1);
        }
    }
}
