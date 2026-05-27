//! [`PermuteAir`] defines the AIR for the permute NPO table.
//!
//! Each row carries one (input_witness, output_witness) pair plus their values. The AIR
//! proves multiset equality between the input and output value vectors via two interactions:
//!
//! 1. The shared `WitnessChecks` bus binds the per-row main columns to the corresponding
//!    witness IDs (input as a reader, output as a creator). This is the same bus every
//!    other AIR uses.
//! 2. A dedicated `PermuteBus` carries one +1 send per real input value and one -1 send
//!    per real output value. The framework's LogUp gadget asserts the running sum balances
//!    to zero across all rows of all PermuteAir instances — which is exactly the statement
//!    `multiset(input_values) == multiset(output_values)`.
//!
//! Lane packing is not yet used (single-lane). The smoke test exercises a 4-element
//! permutation; production callers will register multiple Permute ops which concatenate
//! rows in one PermuteAir trace.
//!
//! # Column layout (lane=1)
//!
//! Main columns (`2 * D`): `[in_v_0, ..., in_v_{D-1}, out_v_0, ..., out_v_{D-1}]`
//!
//! Preprocessed columns (`5`):
//! - col 0 `input_idx`  — input witness base-field index
//! - col 1 `input_mult` — `F::ONE` for real rows, `F::ZERO` for padding
//! - col 2 `output_idx` — output witness base-field index
//! - col 3 `output_mult` — `ext_reads[output_wid]` patched by [`super::super::batch_stark_prover::PermutePreprocessor`]
//! - col 4 `is_real`    — `F::ONE` for real rows, `F::ZERO` for padding (used as the PermuteBus mult)

use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_lookup::builder::InteractionBuilder;
use p3_matrix::dense::RowMajorMatrix;
use tracing::instrument;

/// Per-row preprocessed width for [`PermuteAir`].
pub(crate) const PERMUTE_PREP_LANE_WIDTH: usize = 5;
/// Position of `input_idx` within a preprocessed row.
pub(crate) const PERMUTE_PREP_IN_IDX: usize = 0;
/// Position of `input_mult` within a preprocessed row.
pub(crate) const PERMUTE_PREP_IN_MULT: usize = 1;
/// Position of `output_idx` within a preprocessed row.
pub(crate) const PERMUTE_PREP_OUT_IDX: usize = 2;
/// Position of `output_mult` within a preprocessed row.
pub(crate) const PERMUTE_PREP_OUT_MULT: usize = 3;
/// Position of `is_real` within a preprocessed row.
pub(crate) const PERMUTE_PREP_IS_REAL: usize = 4;

/// AIR for the permute (multiset-equality) table.
///
/// Local constraints: none. Soundness is enforced via two LogUp buses (`WitnessChecks`
/// for witness binding, `PermuteBus` for multiset equality).
#[derive(Debug, Clone)]
pub struct PermuteAir<F, const D: usize> {
    pub(crate) lanes: usize,
    pub(crate) preprocessed: Vec<F>,
    pub(crate) min_height: usize,
    _phantom: PhantomData<F>,
}

impl<F: Field + PrimeCharacteristicRing, const D: usize> PermuteAir<F, D> {
    /// Main trace width per lane: `2 * D` (D input value coeffs + D output value coeffs).
    pub const fn lane_width() -> usize {
        2 * D
    }

    /// Preprocessed width per lane: 5 (input_idx, input_mult, output_idx, output_mult, is_real).
    pub const fn preprocessed_lane_width() -> usize {
        PERMUTE_PREP_LANE_WIDTH
    }

    pub fn new_with_preprocessed(lanes: usize, preprocessed: Vec<F>, min_height: usize) -> Self {
        Self {
            lanes: lanes.max(1),
            preprocessed,
            min_height,
            _phantom: PhantomData,
        }
    }

    /// Build the main trace matrix by concatenating all (input, output) pairs across all
    /// PermuteCircuitRow ops. Padded to power-of-two height with zeros.
    ///
    /// `op.input_values` and `op.output_values` are expected to hold the base-field
    /// coefficient layout: `D` consecutive base elements per logical permute slot.
    /// `generate_permute_trace::<BF, EF>` in `p3_circuit::ops::permute` produces this
    /// layout — for `D > 1` it unpacks each extension-field value into its `D`
    /// coefficients; for `D == 1` (smoke-test path) the layout is a single base
    /// element per slot, identical to the original design.
    #[instrument(skip_all, name = "PermuteAir::build_trace")]
    pub fn trace_to_matrix(
        ops: &[p3_circuit::ops::permute::PermuteCircuitRow<F>],
        lanes: usize,
    ) -> RowMajorMatrix<F> {
        let lane_w = Self::lane_width();
        let row_width = lanes * lane_w;
        let total_pairs: usize = ops.iter().map(|r| r.input_wids.len()).sum();
        let num_rows = total_pairs.div_ceil(lanes).max(1);

        let mut values = F::zero_vec(num_rows * row_width);
        let mut pair_idx = 0usize;
        for op in ops {
            let n_pairs = op.input_wids.len();
            debug_assert_eq!(
                op.input_values.len(),
                n_pairs * D,
                "PermuteCircuitRow.input_values must hold D base-field coefficients per logical slot"
            );
            debug_assert_eq!(
                op.output_values.len(),
                n_pairs * D,
                "PermuteCircuitRow.output_values must hold D base-field coefficients per logical slot"
            );
            for pair in 0..n_pairs {
                let r = pair_idx / lanes;
                let l = pair_idx % lanes;
                let base = r * row_width + l * lane_w;
                // Input value's D coefficients into [base .. base + D];
                // output value's D coefficients into [base + D .. base + 2*D].
                for j in 0..D {
                    values[base + j] = op.input_values[pair * D + j];
                    values[base + D + j] = op.output_values[pair * D + j];
                }
                pair_idx += 1;
            }
        }

        let mut mat = RowMajorMatrix::new(values, row_width);
        mat.pad_to_power_of_two_height(F::ZERO);
        mat
    }
}

impl<F: Field, const D: usize> BaseAir<F> for PermuteAir<F, D> {
    fn width(&self) -> usize {
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

impl<AB: AirBuilder + InteractionBuilder, const D: usize> Air<AB> for PermuteAir<AB::F, D>
where
    AB::F: Field,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let main_local = main.current_slice();
        let prep = builder.preprocessed().clone();
        let prep_local = prep.current_slice();

        let lane_w = Self::lane_width();
        let prep_lane_w = Self::preprocessed_lane_width();

        for lane in 0..self.lanes {
            let main_off = lane * lane_w;
            let prep_off = lane * prep_lane_w;

            let in_idx: AB::Expr = prep_local[prep_off + PERMUTE_PREP_IN_IDX].into();
            let in_mult: AB::Expr = prep_local[prep_off + PERMUTE_PREP_IN_MULT].into();
            let out_idx: AB::Expr = prep_local[prep_off + PERMUTE_PREP_OUT_IDX].into();
            let out_mult: AB::Expr = prep_local[prep_off + PERMUTE_PREP_OUT_MULT].into();
            let is_real: AB::Expr = prep_local[prep_off + PERMUTE_PREP_IS_REAL].into();

            // ---- WitnessChecks: input is a READER (negative multiplicity) ----
            let mut in_fields: Vec<AB::Expr> = Vec::with_capacity(1 + D);
            in_fields.push(in_idx);
            for j in 0..D {
                in_fields.push(main_local[main_off + j].into());
            }
            builder.push_interaction("WitnessChecks", in_fields, -in_mult, 1);

            // ---- WitnessChecks: output is a CREATOR (positive multiplicity, patched to ext_reads) ----
            let mut out_fields: Vec<AB::Expr> = Vec::with_capacity(1 + D);
            out_fields.push(out_idx);
            for j in 0..D {
                out_fields.push(main_local[main_off + D + j].into());
            }
            builder.push_interaction("WitnessChecks", out_fields, out_mult, 1);

            // ---- PermuteBus: input contributes +is_real ----
            let mut perm_in: Vec<AB::Expr> = Vec::with_capacity(D);
            for j in 0..D {
                perm_in.push(main_local[main_off + j].into());
            }
            builder.push_interaction("PermuteBus", perm_in, is_real.clone(), 1);

            // ---- PermuteBus: output contributes -is_real ----
            let mut perm_out: Vec<AB::Expr> = Vec::with_capacity(D);
            for j in 0..D {
                perm_out.push(main_local[main_off + D + j].into());
            }
            builder.push_interaction("PermuteBus", perm_out, -is_real, 1);
        }
    }
}
