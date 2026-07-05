//! [`ExposeAir`] surfaces K witness values as the table instance's
//! `public_values`, so they propagate as recursion `air_public_targets`.
//!
//! This is the only AIR in the framework that declares `num_public_values() >
//! 0`. Each row carries one exposed witness value (`D` coefficient columns).
//! Soundness is enforced by two bindings:
//!
//! 1. `WitnessChecks` bus: the per-row value columns are bound to the input
//!    witness (a reader, multiplicity `is_real`), so the value equals what the
//!    rest of the circuit computed.
//! 2. Public-value equality: on each real row, `is_real * (value - pv) = 0`
//!    binds the declared public value to the value column.
//!
//! Composed, this gives `public_value == value_column == witness_value`. The
//! exposed witnesses are packed one-per-lane in a single real row (the table
//! prover sets `lanes` to the total exposed count); padding rows have
//! `is_real = 0` so they contribute nothing.
//!
//! # Column layout (per lane)
//!
//! Main columns (`D`): `[v_0, ..., v_{D-1}]` — the exposed value's coefficients.
//! Preprocessed columns (`2`):
//! - col 0 `input_idx` — input witness base-field index
//! - col 1 `is_real`   — `F::ONE` for real rows, `F::ZERO` for padding

use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_lookup::builder::InteractionBuilder;
use p3_matrix::dense::RowMajorMatrix;
use tracing::instrument;

/// Per-row preprocessed width for [`ExposeAir`].
pub(crate) const EXPOSE_PREP_LANE_WIDTH: usize = 2;
/// Position of `input_idx` within a preprocessed row.
pub(crate) const EXPOSE_PREP_IN_IDX: usize = 0;
/// Position of `is_real` within a preprocessed row.
pub(crate) const EXPOSE_PREP_IS_REAL: usize = 1;

/// AIR for the expose (public-value surfacing) table.
#[derive(Debug, Clone)]
pub struct ExposeAir<F, const D: usize> {
    pub(crate) lanes: usize,
    pub(crate) preprocessed: Vec<F>,
    pub(crate) min_height: usize,
    _phantom: PhantomData<F>,
}

impl<F: Field + PrimeCharacteristicRing, const D: usize> ExposeAir<F, D> {
    /// Main trace width per lane: `D` value coefficients.
    pub const fn lane_width() -> usize {
        D
    }

    /// Preprocessed width per lane: 2 (input_idx, is_real).
    pub const fn preprocessed_lane_width() -> usize {
        EXPOSE_PREP_LANE_WIDTH
    }

    pub fn new_with_preprocessed(lanes: usize, preprocessed: Vec<F>, min_height: usize) -> Self {
        // Derive lanes from the preprocessed length so the AIR is always
        // self-consistent (loop count, width, and `num_public_values` all
        // match the preprocessed), independent of the `lanes` hint — which the
        // degree/symbolic pass does not always set to the true value. Falls
        // back to the hint only when preprocessed is empty (verifier's
        // pre-injection AIR).
        let lanes = if preprocessed.is_empty() {
            lanes.max(1)
        } else {
            (preprocessed.len() / EXPOSE_PREP_LANE_WIDTH).max(1)
        };
        Self { lanes, preprocessed, min_height, _phantom: PhantomData }
    }

    /// Build the main trace by concatenating all exposed values across all
    /// ExposeCircuitRow ops, `lanes` per row. Padded to power-of-two height.
    /// `op.input_values` holds the base-field coefficient layout (`D` per slot).
    #[instrument(skip_all, name = "ExposeAir::build_trace")]
    pub fn trace_to_matrix(
        ops: &[p3_circuit::ops::expose::ExposeCircuitRow<F>],
        lanes: usize,
    ) -> RowMajorMatrix<F> {
        let lane_w = Self::lane_width();
        let row_width = lanes * lane_w;
        let total: usize = ops.iter().map(|r| r.input_wids.len()).sum();
        let num_rows = total.div_ceil(lanes).max(1);

        let mut values = F::zero_vec(num_rows * row_width);
        let mut idx = 0usize;
        for op in ops {
            let n = op.input_wids.len();
            debug_assert_eq!(
                op.input_values.len(),
                n * D,
                "ExposeCircuitRow.input_values must hold D base-field coefficients per slot"
            );
            for slot in 0..n {
                let r = idx / lanes;
                let l = idx % lanes;
                let base = r * row_width + l * lane_w;
                for j in 0..D {
                    values[base + j] = op.input_values[slot * D + j];
                }
                idx += 1;
            }
        }

        let mut mat = RowMajorMatrix::new(values, row_width);
        mat.pad_to_power_of_two_height(F::ZERO);
        mat
    }
}

impl<F: Field, const D: usize> BaseAir<F> for ExposeAir<F, D> {
    fn width(&self) -> usize {
        self.lanes * Self::lane_width()
    }

    fn num_public_values(&self) -> usize {
        // Expose every lane's D coefficients as public values.
        self.lanes * Self::lane_width()
    }

    fn preprocessed_width(&self) -> usize {
        self.lanes * Self::preprocessed_lane_width()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let width = self.lanes * Self::preprocessed_lane_width();
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

impl<AB, const D: usize> Air<AB> for ExposeAir<AB::F, D>
where
    AB: AirBuilder + InteractionBuilder,
    AB::F: Field,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let main_local = main.current_slice();
        let prep = builder.preprocessed().clone();
        let prep_local = prep.current_slice();
        let pis = builder.public_values().to_vec();

        let lane_w = Self::lane_width();
        let prep_lane_w = Self::preprocessed_lane_width();

        for lane in 0..self.lanes {
            let main_off = lane * lane_w;
            let prep_off = lane * prep_lane_w;

            let in_idx: AB::Expr = prep_local[prep_off + EXPOSE_PREP_IN_IDX].into();
            let is_real: AB::Expr = prep_local[prep_off + EXPOSE_PREP_IS_REAL].into();

            // ---- WitnessChecks: value is a READER (multiplicity is_real) ----
            let mut fields: Vec<AB::Expr> = Vec::with_capacity(1 + D);
            fields.push(in_idx);
            for j in 0..D {
                fields.push(main_local[main_off + j].into());
            }
            builder.push_interaction("WitnessChecks", fields, -is_real.clone(), 1);

            // ---- Public-value binding: is_real * (value - pv) == 0 ----
            for j in 0..D {
                let v: AB::Expr = main_local[main_off + j].into();
                let pv: AB::Expr = pis[main_off + j].into();
                builder.assert_zero(is_real.clone() * (v - pv));
            }
        }
    }
}
