use alloc::vec::Vec;

use p3_circuit::ops::NpoTypeId;
use serde::{Deserialize, Serialize};

use crate::ProofMetadataError;

/// Configuration for packing multiple primitive operations into a single AIR row.
///
/// Equality ignores the zero-knowledge hiding-budget settings ([`Self::zk_num_queries`] and the
/// floor set by [`Self::with_zk_min_trace_height`]), which are prover-side policy and are not
/// serialized (so the proof metadata, and the bytes of serialized proofs, are unchanged by them).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TablePacking {
    /// Number of public-input operations packed per AIR row.
    public_lanes: usize,
    /// Number of ALU operations packed per AIR row.
    alu_lanes: usize,
    /// Per-NPO lane counts: `(op_type, lanes)`. Defaults to 1 for any op not listed.
    #[serde(default)]
    npo_lanes: Vec<(NpoTypeId, usize)>,
    /// Minimum trace height for all tables (must be power of two).
    /// This is required for FRI with higher `log_final_poly_len`.
    /// FRI requires: `log_trace_height > log_final_poly_len + log_blowup`
    /// So min_trace_height should be >= `2^(log_final_poly_len + log_blowup + 1)`
    min_trace_height: usize,
    /// Pack this many consecutive `HornerAcc` ops (same `b` witness) per ALU row on lane 0.
    /// Must be at least 2. Default 2 matches the previous double-step Horner layout.
    #[serde(default = "default_horner_pack_k")]
    horner_packed_steps: usize,
    /// FRI query count the zero-knowledge hiding budget is computed for (see
    /// [`Self::zk_min_trace_height`]). Prover-side only: never serialized, ignored by equality.
    #[serde(skip, default = "default_zk_num_queries")]
    zk_num_queries: usize,
    /// Optional extra lower bound on `min_trace_height` under ZK, on top of the hiding budget
    /// (0 = none). It can only raise the requirement. Prover-side only, like `zk_num_queries`.
    #[serde(skip)]
    zk_min_trace_height_floor: usize,
}

const fn default_horner_pack_k() -> usize {
    2
}

/// FRI query count assumed by the zero-knowledge hiding budget unless the caller declares the
/// real one with [`TablePacking::with_zk_num_queries`].
///
/// 54 is the query count of the configuration this fork is used with (log blowup 2); a caller
/// whose FRI parameters use more queries (for example log blowup 1 at the same security level)
/// must declare it, or the guard in [`super::BatchStarkProver`] under-estimates the budget.
pub const DEFAULT_ZK_NUM_QUERIES: usize = 54;

/// Out-of-domain opening points of every committed main column (`zeta` and `g * zeta`).
pub const ZK_OPENING_POINTS: usize = 2;

/// Hiding budget for the default query count and a degree-4 challenge field:
/// `zk_hiding_budget(54, 4, 2) = 128` (see [`TablePacking::zk_hiding_budget`]).
///
/// Under ZK each committed column of a table of base height `n` is a polynomial of degree below
/// `2n` with `n` random interleaved rows. The verifier learns `D` base-field values per
/// out-of-domain opening point (`zeta`, `g * zeta`) and one value per FRI query; the openings are
/// jointly uniform, so the column stays hidden, when `n >= 2 * (num_queries + D * points)`. For
/// `num_queries = 54`, `D = 4` and 2 points this is 124, rounded up to the next power of two:
/// 128. This covers the LogUp masks and the auxiliary (permutation) columns.
pub const DEFAULT_ZK_MIN_TRACE_HEIGHT: usize =
    TablePacking::zk_hiding_budget(DEFAULT_ZK_NUM_QUERIES, 4, ZK_OPENING_POINTS);

const _: () = assert!(DEFAULT_ZK_MIN_TRACE_HEIGHT == 128);

const fn default_zk_num_queries() -> usize {
    DEFAULT_ZK_NUM_QUERIES
}

impl PartialEq for TablePacking {
    fn eq(&self, other: &Self) -> bool {
        // Destructure so that a new layout field cannot be silently left out of the comparison.
        let Self {
            public_lanes,
            alu_lanes,
            npo_lanes,
            min_trace_height,
            horner_packed_steps,
            zk_num_queries: _,
            zk_min_trace_height_floor: _,
        } = self;
        *public_lanes == other.public_lanes
            && *alu_lanes == other.alu_lanes
            && *npo_lanes == other.npo_lanes
            && *min_trace_height == other.min_trace_height
            && *horner_packed_steps == other.horner_packed_steps
    }
}

impl Eq for TablePacking {}

/// Sanity ceiling for lane counts and `horner_packed_steps` in [`TablePacking::validate`].
///
/// Far beyond any real packing (which uses single- to low-double-digit values), but small
/// enough that every width computation derived from these fields (`lanes * lane_width`, and
/// similar) stays comfortably within `usize` range instead of overflowing on proof-controlled
/// metadata.
pub(crate) const MAX_SANE_LANES: usize = 1 << 16;

impl TablePacking {
    /// Create a new [`TablePacking`] with the given primitive lane counts (clamped to at least 1).
    ///
    /// NPO lanes default to 1. Use [`with_npo_lanes`](Self::with_npo_lanes) to override per op type.
    pub fn new(public_lanes: usize, alu_lanes: usize) -> Self {
        Self {
            public_lanes: public_lanes.max(1),
            alu_lanes: alu_lanes.max(1),
            npo_lanes: Vec::new(),
            min_trace_height: 1,
            horner_packed_steps: 2,
            zk_num_queries: DEFAULT_ZK_NUM_QUERIES,
            zk_min_trace_height_floor: 0,
        }
    }

    /// Override packed Horner chain length (must be >= 2).
    #[must_use]
    pub fn with_horner_pack_k(mut self, k: usize) -> Self {
        assert!(k >= 2, "horner_packed_steps must be at least 2");
        self.horner_packed_steps = k;
        self
    }

    /// Override public and ALU lane counts after trace-driven clamping (e.g. dummy-only traces).
    ///
    /// Used when embedding the effective packing in [`super::BatchStarkProof`] so metadata matches
    /// proving while preserving [`Self::horner_packed_steps`] and NPO lane overrides.
    #[must_use]
    pub fn with_public_alu_lanes(mut self, public_lanes: usize, alu_lanes: usize) -> Self {
        self.public_lanes = public_lanes.max(1);
        self.alu_lanes = alu_lanes.max(1);
        self
    }

    /// Override the lane count for a specific non-primitive op type (builder-style).
    ///
    /// Any NPO not listed falls back to the lane count returned by its [`TableProver`].
    #[must_use]
    pub fn with_npo_lanes(mut self, op_type: impl Into<NpoTypeId>, lanes: usize) -> Self {
        let op_type = op_type.into();
        let lanes = lanes.max(1);
        if let Some(entry) = self.npo_lanes.iter_mut().find(|(k, _)| *k == op_type) {
            entry.1 = lanes;
        } else {
            self.npo_lanes.push((op_type, lanes));
        }
        self
    }

    /// Update the current [`TablePacking`] with a minimum trace height requirement.
    ///
    /// FRI requires: `log_trace_height > log_final_poly_len + log_blowup`
    /// So `min_trace_height` should be >= `2^(log_final_poly_len + log_blowup + 1)`
    ///
    /// For example, with `log_final_poly_len = 3` and `log_blowup = 1`:
    /// - Required: `min_trace_height > 2^(3+1) = 16`
    /// - So use `min_trace_height = 32` (next power of two)
    #[must_use]
    pub fn with_min_trace_height(mut self, min_trace_height: usize) -> Self {
        // Ensure min_trace_height is a power of two and at least 1
        self.min_trace_height = min_trace_height.next_power_of_two().max(1);
        self
    }

    /// Update the current [`TablePacking`] with minimum height derived from FRI parameters.
    ///
    /// This automatically calculates the minimum trace height from `log_final_poly_len` and `log_blowup`.
    #[must_use]
    pub const fn with_fri_params(mut self, log_final_poly_len: usize, log_blowup: usize) -> Self {
        // FRI requires: log_min_height > log_final_poly_len + log_blowup
        // So min_height must be >= 2^(log_final_poly_len + log_blowup + 1)
        let min_log_height = log_final_poly_len + log_blowup + 1;
        self.min_trace_height = 1usize << min_log_height;
        self
    }

    /// Return the number of public-input operations packed per AIR row.
    pub const fn public_lanes(&self) -> usize {
        self.public_lanes
    }

    /// Return the number of ALU operations packed per AIR row.
    pub const fn alu_lanes(&self) -> usize {
        self.alu_lanes
    }

    /// Return the lane count for a specific NPO type.
    ///
    /// Returns the overridden value if one was set via [`with_npo_lanes`](Self::with_npo_lanes),
    /// otherwise returns `None` (the caller should fall back to the prover's own default).
    pub fn npo_lanes(&self, op_type: &NpoTypeId) -> Option<usize> {
        self.npo_lanes
            .iter()
            .find(|(k, _)| k == op_type)
            .map(|(_, v)| *v)
    }

    /// Return the minimum trace height (always a power of two, at least 1).
    pub const fn min_trace_height(&self) -> usize {
        self.min_trace_height
    }

    /// Number of consecutive HornerAcc steps packed into one scheduled ALU row (lane 0).
    pub const fn horner_packed_steps(&self) -> usize {
        self.horner_packed_steps
    }

    /// Smallest [`Self::min_trace_height`] that [`super::BatchStarkProver`] accepts when the
    /// PCS is zero-knowledge and the challenge field has degree `challenge_degree` over the base
    /// field: the hiding budget `zk_hiding_budget(zk_num_queries, challenge_degree, 2)`, or the
    /// floor set by [`Self::with_zk_min_trace_height`] if that is larger.
    ///
    /// Every table is padded to at least `min_trace_height` rows, so this bounds the base
    /// height of every committed table from below. A table is hidden only if its base height
    /// exceeds the number of linear functionals the verifier learns about each column; see
    /// [`Self::zk_hiding_budget`]. The pinned `HidingFriPcs` does not check this itself, and
    /// it does not expose its FRI query count, so the count is a declared parameter
    /// ([`Self::with_zk_num_queries`], default [`DEFAULT_ZK_NUM_QUERIES`]). For the default
    /// query count and a degree-4 challenge field this is [`DEFAULT_ZK_MIN_TRACE_HEIGHT`] (128).
    pub const fn zk_min_trace_height(&self, challenge_degree: usize) -> usize {
        let budget =
            Self::zk_hiding_budget(self.zk_num_queries, challenge_degree, ZK_OPENING_POINTS);
        if self.zk_min_trace_height_floor > budget {
            self.zk_min_trace_height_floor
        } else {
            budget
        }
    }

    /// FRI query count the ZK hiding budget is computed for (default [`DEFAULT_ZK_NUM_QUERIES`]).
    pub const fn zk_num_queries(&self) -> usize {
        self.zk_num_queries
    }

    /// Declare the FRI query count of the PCS this packing is proved with, so that the ZK hiding
    /// budget ([`Self::zk_min_trace_height`]) matches it.
    ///
    /// A caller whose FRI parameters use more than [`DEFAULT_ZK_NUM_QUERIES`] queries must call
    /// this (with the real `num_queries`), and then raise [`Self::min_trace_height`] to the
    /// budget if needed; otherwise ZK proving under-estimates the budget. Declaring fewer queries
    /// than the PCS really uses weakens hiding. Has no effect without ZK.
    #[must_use]
    pub const fn with_zk_num_queries(mut self, num_queries: usize) -> Self {
        self.zk_num_queries = num_queries;
        self
    }

    /// Require `min_trace_height >= zk_min_trace_height` under ZK in addition to the hiding
    /// budget (a stricter policy). This can only raise the requirement: values at or below the
    /// budget derived from [`Self::zk_num_queries`] have no effect, so it cannot switch the
    /// guard off. To lower the budget, declare the real (smaller) query count with
    /// [`Self::with_zk_num_queries`].
    #[must_use]
    pub const fn with_zk_min_trace_height(mut self, zk_min_trace_height: usize) -> Self {
        self.zk_min_trace_height_floor = zk_min_trace_height;
        self
    }

    /// Hiding budget for the given FRI query count, extension degree and number of out-of-domain
    /// opening points: `2 * (num_queries + ext_degree * opening_points)`, rounded up to a power of
    /// two. For 54 queries, `D = 4` and 2 points (`zeta`, `g * zeta`) this is 128.
    ///
    /// The factor 2 is the design's sufficient margin over the tight count
    /// `num_queries + ext_degree * opening_points`; it is not independently proven tight.
    pub const fn zk_hiding_budget(
        num_queries: usize,
        ext_degree: usize,
        opening_points: usize,
    ) -> usize {
        (2 * (num_queries + ext_degree * opening_points)).next_power_of_two()
    }

    /// Re-check the invariants the builder methods enforce, after deserialization.
    pub fn validate(&self) -> Result<(), ProofMetadataError> {
        if self.public_lanes == 0 {
            return Err(ProofMetadataError::ZeroLanes("public_lanes"));
        }
        if self.public_lanes > MAX_SANE_LANES {
            return Err(ProofMetadataError::LanesTooLarge {
                field: "public_lanes",
                got: self.public_lanes,
                max: MAX_SANE_LANES,
            });
        }
        if self.alu_lanes == 0 {
            return Err(ProofMetadataError::ZeroLanes("alu_lanes"));
        }
        if self.alu_lanes > MAX_SANE_LANES {
            return Err(ProofMetadataError::LanesTooLarge {
                field: "alu_lanes",
                got: self.alu_lanes,
                max: MAX_SANE_LANES,
            });
        }
        for (op_type, lanes) in &self.npo_lanes {
            if *lanes == 0 {
                return Err(ProofMetadataError::ZeroNpoLanes(op_type.clone()));
            }
            if *lanes > MAX_SANE_LANES {
                return Err(ProofMetadataError::NpoLanesTooLarge {
                    op_type: op_type.clone(),
                    got: *lanes,
                    max: MAX_SANE_LANES,
                });
            }
        }
        if self.min_trace_height == 0 || !self.min_trace_height.is_power_of_two() {
            return Err(ProofMetadataError::BadMinTraceHeight(self.min_trace_height));
        }
        if self.horner_packed_steps < 2 {
            return Err(ProofMetadataError::BadHornerPackedSteps(
                self.horner_packed_steps,
            ));
        }
        if self.horner_packed_steps > MAX_SANE_LANES {
            return Err(ProofMetadataError::HornerPackedStepsTooLarge(
                self.horner_packed_steps,
                MAX_SANE_LANES,
            ));
        }
        Ok(())
    }
}

impl Default for TablePacking {
    fn default() -> Self {
        Self::new(1, 1)
    }
}

/// Main trace width, preprocessed row width, logical AIR row count, and lane packing for one table.
#[derive(Clone, Copy, Debug)]
pub(crate) struct AirTableShape {
    pub main_cols: usize,
    pub prep_cols: usize,
    pub rows: usize,
    pub lanes: usize,
}

/// Layout of every table in a batch (for prover logging).
#[derive(Debug)]
pub(crate) struct TraceTablesLayout {
    pub const_: AirTableShape,
    pub public: AirTableShape,
    pub alu: AirTableShape,
    pub non_primitives: Vec<(NpoTypeId, AirTableShape)>,
}

impl TraceTablesLayout {
    /// Log each AIR’s main width, preprocessed width, row count, and lanes at info level.
    pub fn log(&self) {
        tracing::info!(
            table = "CONST",
            main_cols = self.const_.main_cols,
            prep_cols = self.const_.prep_cols,
            rows = self.const_.rows,
            "AIR shape"
        );
        tracing::info!(
            table = "PUBLIC",
            main_cols = self.public.main_cols,
            prep_cols = self.public.prep_cols,
            rows = self.public.rows,
            "AIR shape"
        );
        tracing::info!(
            table = "ALU",
            main_cols = self.alu.main_cols,
            prep_cols = self.alu.prep_cols,
            rows = self.alu.rows,
            lanes = self.alu.lanes,
            "AIR shape"
        );
        for (op, shape) in &self.non_primitives {
            tracing::info!(
                table = ?op,
                main_cols = shape.main_cols,
                prep_cols = shape.prep_cols,
                rows = shape.rows,
                lanes = shape.lanes,
                "AIR shape"
            );
        }
    }
}
