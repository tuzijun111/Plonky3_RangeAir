//! Circuit-based challenger implementation matching native DuplexChallenger exactly.
//!
//! This module provides [`CircuitChallenger`], which maintains state as coefficient-level
//! targets to ensure exact transcript compatibility with the native `DuplexChallenger`.
//!
//! # Soundness
//!
//! All Poseidon2 permutations in the challenger are CTL-verified against the Poseidon2 AIR table.
//! The circuit builder's `add_poseidon2_perm_for_challenger_chained` /
//! `add_poseidon2_perm_for_challenger_base` (in `p3_circuit`) delegate to the standard Poseidon2
//! non-primitive op with rate-input and rate-output CTL exposure, and the executor runs the real
//! permutation so the lookup argument enforces correctness.
//!
//! The capacity is never published on the bus, so it is carried by the AIR: the first
//! permutation starts a sponge chain whose capacity is CTL-fed from the zero initial state
//! (constants), and every later permutation continues it (`new_start = false`, capacity slots
//! left off the bus), so the Poseidon AIR's sponge chain constraint binds its capacity input to
//! the previous row's capacity output. The builder refuses a circuit in which the previous row
//! of the table is not the previous duplex step (`SpongeChainNotAdjacent`). (Previously every
//! `D >= 2` permutation started a fresh chain with CTL-fed capacity read from witnesses no
//! table created with the previous permutation's capacity output, so it was prover-chosen.)
//!
//! No packing or unpacking goes through the plain recompose table: its row holds the
//! coefficients in free main-trace columns and publishes only its packed output on the
//! `WitnessChecks` bus, which would leave every state coefficient — and so every absorbed value,
//! every squeezed challenge and the limbs each permutation reads — a value the prover picks
//! independently of the permutation that produced it. Packing (state coefficients into limbs,
//! squeezed coefficients into `sample_ext` challenges) uses the ALU `mul_add` chain, which reads
//! each coefficient as a bus-bound operand. Unpacking (a permutation's rate outputs, and
//! `observe_ext` values) uses the `recompose/coeff` table whenever the recompose NPO is enabled:
//! it publishes each coefficient as `[idx, v_i, 0, .., 0]` next to the limb, so every coefficient
//! the transcript hands out is a base-field element tied to the limb it came from. The ALU chain
//! alone constrains only `sum(c_i · basis_i)`, which over an extension field leaves each
//! coefficient `D - 1` free base dimensions — enough to move a squeezed challenge or a query
//! index while every permutation limb stays put — so a circuit without `enable_recompose` still
//! has that gap (upstream f89d776 closes it by requiring the table).

use alloc::vec;
use alloc::vec::Vec;

use p3_circuit::ops::{Poseidon1Config, Poseidon2Config};
use p3_circuit::{CircuitBuilder, CircuitBuilderError, NonPrimitiveOpId};
use p3_field::{ExtensionField, PrimeField64};

use crate::Target;
use crate::challenger_perm::ChallengerPermConfig;
use crate::traits::RecursiveChallenger;

/// Circuit challenger with coefficient-level state management.
///
/// Maintains state as WIDTH base field coefficient targets to exactly match
/// the native `DuplexChallenger<F, P, WIDTH, RATE>` behavior.
///
/// # Type Parameters
/// - `WIDTH`: Sponge state width (16 for Poseidon2)
/// - `RATE`: Sponge rate (8 for typical configuration)
/// - `C`: Challenger permutation config (e.g. [`Poseidon2Config`])
pub struct CircuitChallenger<const WIDTH: usize, const RATE: usize, C: ChallengerPermConfig> {
    /// Permutation config for the challenger (e.g. Poseidon2).
    config: C,

    /// Sponge state: WIDTH base field coefficient targets.
    /// Each target represents a base field element embedded in EF.
    state: Vec<Target>,

    /// Buffered inputs not yet absorbed into state.
    input_buffer: Vec<Target>,

    /// Buffered outputs from last duplexing.
    output_buffer: Vec<Target>,

    /// Whether the challenger has been initialized with zero state.
    initialized: bool,

    /// Whether `duplexing_base` has been called at least once since the last init/clear.
    ///
    /// The first D=1 permutation uses `new_start=true` with rate slots CTL-verified and
    /// capacity `None` (zeros enforced by the compact D=1 AIR). Later permutations use
    /// `new_start=false` with the same capacity pattern and chaining.
    duplexed_once: bool,

    /// The `D >= 2` path's previous permutation, if any. Every permutation after the first
    /// continues the sponge chain from it: its capacity limbs are left off the bus and the
    /// Poseidon AIR's sponge chain constraint binds them to this row's capacity output.
    last_ext_perm: Option<NonPrimitiveOpId>,
}

impl<const WIDTH: usize, const RATE: usize, C: ChallengerPermConfig>
    CircuitChallenger<WIDTH, RATE, C>
{
    /// Create a new uninitialized circuit challenger.
    ///
    /// # Parameters
    /// - `config`: The permutation configuration (e.g. Poseidon2) for the challenger.
    ///
    /// Call `init()` to initialize the state with zeros before use.
    pub const fn new(config: C) -> Self {
        Self {
            config,
            state: Vec::new(),
            input_buffer: Vec::new(),
            output_buffer: Vec::new(),
            initialized: false,
            duplexed_once: false,
            last_ext_perm: None,
        }
    }

    /// Initialize the challenger state with zeros.
    ///
    /// This must be called before any observe/sample operations.
    pub fn init<BF, EF>(&mut self, circuit: &mut CircuitBuilder<EF>)
    where
        BF: PrimeField64,
        EF: ExtensionField<BF>,
    {
        if self.initialized {
            return;
        }
        let zero = circuit.define_const(EF::ZERO);
        self.state = vec![zero; WIDTH];
        self.initialized = true;
    }

    /// Perform duplexing: absorb inputs, permute, fill output buffer.
    ///
    /// Matches native `DuplexChallenger::duplexing()` exactly.
    fn duplexing<BF, EF>(&mut self, circuit: &mut CircuitBuilder<EF>)
    where
        BF: PrimeField64,
        EF: ExtensionField<BF>,
    {
        debug_assert!(self.initialized, "Challenger must be initialized");
        debug_assert!(self.input_buffer.len() <= RATE, "Input buffer exceeds RATE");

        let p2_config = self.config.as_poseidon2().copied();
        let p1_config = self.config.as_poseidon1().copied();

        // 1. Overwrite state[0..n] with inputs (NOT XOR, matches native)
        for (i, val) in self.input_buffer.drain(..).enumerate() {
            self.state[i] = val;
        }

        // Branch by NPO packing (`config.d()`), not `EF::DIMENSION`, so a quintic
        // (or other) challenge field can still use a base width-16 permutation.
        if let Some(cfg) = p2_config {
            if cfg.d() == 1 {
                self.duplexing_base(circuit, cfg);
            } else {
                self.duplexing_ext::<BF, EF>(circuit, cfg);
            }
        } else if let Some(cfg) = p1_config {
            if cfg.d() == 1 {
                self.duplexing_base_p1(circuit, cfg);
            } else {
                self.duplexing_ext_p1::<BF, EF>(circuit, cfg);
            }
        } else {
            panic!("unsupported challenger permutation");
        }

        // 5. Fill output buffer from state[0..RATE]
        self.output_buffer.clear();
        self.output_buffer.extend_from_slice(&self.state[..RATE]);
    }

    /// Duplexing for D=1 (base field): permutation operates directly on 16 elements.
    ///
    /// The first call uses `new_start=true`: rate slots are CTL-verified; capacity slots use
    /// `None` (initial state is zero and the compact D=1 AIR asserts zero capacity on sponge
    /// chain starts). Subsequent calls use `new_start=false` with `None` for capacity so the
    /// AIR enforces continuity via the chain constraint.
    fn duplexing_base<EF>(
        &mut self,
        circuit: &mut CircuitBuilder<EF>,
        poseidon2_config: Poseidon2Config,
    ) where
        EF: p3_field::Field,
    {
        let (new_start, inputs) = if !self.duplexed_once {
            // First permutation: CTL-verify rate only; capacity is zero without witness CTL.
            let inputs: [Option<Target>; 16] =
                core::array::from_fn(|i| if i < RATE { Some(self.state[i]) } else { None });
            (true, inputs)
        } else {
            // Subsequent permutations: CTL-verify rate inputs only; capacity via chain.
            let inputs: [Option<Target>; 16] =
                core::array::from_fn(|i| if i < RATE { Some(self.state[i]) } else { None });
            (false, inputs)
        };
        self.duplexed_once = true;

        let outputs = circuit
            .add_poseidon2_perm_for_challenger_base(poseidon2_config, new_start, inputs)
            .expect("poseidon2 base permutation should succeed");

        self.state = outputs.to_vec();
    }

    /// Unpack one extension limb into base coefficients: through the `recompose/coeff` table
    /// when the recompose NPO is enabled (each coefficient base-field and bound to the limb),
    /// else through the ALU chain (weighted sum only).
    fn unpack_limb<BF, EF>(circuit: &mut CircuitBuilder<EF>, limb: Target) -> Vec<Target>
    where
        BF: PrimeField64,
        EF: ExtensionField<BF>,
    {
        if circuit.is_recompose_enabled() {
            circuit.decompose_ext_to_base_coeffs_with_coeff_lookups::<BF>(limb)
        } else {
            circuit.decompose_ext_to_base_coeffs_via_alu::<BF>(limb)
        }
        .expect("decomposition should succeed")
    }

    /// Pack the sponge state into the extension limbs the next `D >= 2` permutation reads:
    /// every limb for the chain start, only the rate limbs for a continuation (whose capacity
    /// is chained in the AIR).
    fn pack_ext_inputs<BF, EF>(&self, circuit: &mut CircuitBuilder<EF>) -> Vec<Target>
    where
        BF: PrimeField64,
        EF: ExtensionField<BF>,
    {
        let num_limbs = if self.last_ext_perm.is_some() {
            RATE / EF::DIMENSION
        } else {
            WIDTH / EF::DIMENSION
        };
        (0..num_limbs)
            .map(|i| {
                let start = i * EF::DIMENSION;
                circuit
                    .recompose_base_coeffs_to_ext_via_alu::<BF>(
                        &self.state[start..start + EF::DIMENSION],
                    )
                    .expect("recomposition should succeed")
            })
            .collect()
    }

    fn duplexing_ext<BF, EF>(
        &mut self,
        circuit: &mut CircuitBuilder<EF>,
        poseidon2_config: Poseidon2Config,
    ) where
        BF: PrimeField64,
        EF: ExtensionField<BF>,
    {
        let ext_inputs = self.pack_ext_inputs::<BF, EF>(circuit);
        let (op_id, rate_outputs) = circuit
            .add_poseidon2_perm_for_challenger_chained(
                poseidon2_config,
                self.last_ext_perm,
                &ext_inputs,
            )
            .expect("poseidon2 permutation should succeed");
        self.last_ext_perm = Some(op_id);

        // Only the rate limbs are unpacked: the capacity never leaves the permutation table
        // after the first step (continuations chain it in the AIR), so the capacity entries of
        // `self.state` keep the zero initial state and are never read again.
        for (limb, &ext_out) in rate_outputs.iter().enumerate() {
            let coeffs = Self::unpack_limb::<BF, EF>(circuit, ext_out);
            let start = limb * EF::DIMENSION;
            for (i, coeff) in coeffs.into_iter().enumerate() {
                self.state[start + i] = coeff;
            }
        }
    }

    /// Poseidon1 D=1 duplexing.
    fn duplexing_base_p1<EF>(
        &mut self,
        circuit: &mut CircuitBuilder<EF>,
        poseidon1_config: Poseidon1Config,
    ) where
        EF: p3_field::Field,
    {
        let new_start = !self.duplexed_once;
        let inputs: [Option<Target>; 16] =
            core::array::from_fn(|i| if i < RATE { Some(self.state[i]) } else { None });
        self.duplexed_once = true;

        let outputs = circuit
            .add_poseidon1_perm_for_challenger_base(poseidon1_config, new_start, inputs)
            .expect("poseidon1 base permutation should succeed");

        self.state = outputs.to_vec();
    }

    /// Poseidon1 D>=2 duplexing.
    fn duplexing_ext_p1<BF, EF>(
        &mut self,
        circuit: &mut CircuitBuilder<EF>,
        poseidon1_config: Poseidon1Config,
    ) where
        BF: PrimeField64,
        EF: ExtensionField<BF>,
    {
        let ext_inputs = self.pack_ext_inputs::<BF, EF>(circuit);
        let (op_id, rate_outputs) = circuit
            .add_poseidon1_perm_for_challenger_chained(
                poseidon1_config,
                self.last_ext_perm,
                &ext_inputs,
            )
            .expect("poseidon1 permutation should succeed");
        self.last_ext_perm = Some(op_id);

        // Only the rate limbs are unpacked; see `duplexing_ext`.
        for (limb, &ext_out) in rate_outputs.iter().enumerate() {
            let coeffs = Self::unpack_limb::<BF, EF>(circuit, ext_out);
            let start = limb * EF::DIMENSION;
            for (i, coeff) in coeffs.into_iter().enumerate() {
                self.state[start + i] = coeff;
            }
        }
    }
}

impl<const WIDTH: usize, const RATE: usize> CircuitChallenger<WIDTH, RATE, Poseidon2Config> {
    /// Create a challenger with BabyBear D4 Width16 configuration (default).
    pub const fn new_babybear() -> Self {
        Self::new(Poseidon2Config::BABY_BEAR_D4_W16)
    }

    /// Create a challenger with BabyBear D1 Width16 configuration (base field challenges).
    pub const fn new_babybear_base() -> Self {
        Self::new(Poseidon2Config::BABY_BEAR_D1_W16)
    }

    /// Create a challenger with KoalaBear D4 Width16 configuration.
    pub const fn new_koalabear() -> Self {
        Self::new(Poseidon2Config::KOALA_BEAR_D4_W16)
    }

    /// Create a challenger with KoalaBear D1 Width16 configuration (base field challenges).
    pub const fn new_koalabear_base() -> Self {
        Self::new(Poseidon2Config::KOALA_BEAR_D1_W16)
    }
}

impl CircuitChallenger<8, 4, Poseidon2Config> {
    /// Create a challenger with Goldilocks D2 Width8 configuration.
    pub const fn new_goldilocks() -> Self {
        Self::new(Poseidon2Config::GOLDILOCKS_D2_W8)
    }
}

impl<const WIDTH: usize, const RATE: usize> CircuitChallenger<WIDTH, RATE, Poseidon1Config> {
    /// Create a Poseidon1 challenger with BabyBear D1 Width16 (base field challenges).
    pub const fn new_babybear_poseidon1_base() -> Self {
        Self::new(Poseidon1Config::BABY_BEAR_D1_W16)
    }

    /// Create a Poseidon1 challenger with KoalaBear D1 Width16 (base field challenges).
    pub const fn new_koalabear_poseidon1_base() -> Self {
        Self::new(Poseidon1Config::KOALA_BEAR_D1_W16)
    }
}

impl CircuitChallenger<8, 4, Poseidon1Config> {
    /// Create a Poseidon1 challenger with Goldilocks D2 Width8 configuration.
    pub const fn new_goldilocks_poseidon1() -> Self {
        Self::new(Poseidon1Config::GOLDILOCKS_D2_W8)
    }
}

impl<BF, EF, const WIDTH: usize, const RATE: usize, C: ChallengerPermConfig>
    RecursiveChallenger<BF, EF> for CircuitChallenger<WIDTH, RATE, C>
where
    BF: PrimeField64,
    EF: ExtensionField<BF>,
{
    fn observe(&mut self, circuit: &mut CircuitBuilder<EF>, value: Target) {
        // Ensure initialized
        self.init::<BF, EF>(circuit);

        // Any buffered output is now invalid (matches native behavior)
        self.output_buffer.clear();

        self.input_buffer.push(value);

        if self.input_buffer.len() == RATE {
            self.duplexing::<BF, EF>(circuit);
        }
    }

    fn sample(&mut self, circuit: &mut CircuitBuilder<EF>) -> Target {
        // Ensure initialized
        self.init::<BF, EF>(circuit);

        // If we have buffered inputs or ran out of outputs, duplex
        // (matches native DuplexChallenger::sample behavior)
        if !self.input_buffer.is_empty() || self.output_buffer.is_empty() {
            self.duplexing::<BF, EF>(circuit);
        }

        self.output_buffer
            .pop()
            .expect("Output buffer should be non-empty after duplexing")
    }

    fn observe_ext(&mut self, circuit: &mut CircuitBuilder<EF>, value: Target) {
        // Decompose extension element to D base coefficients
        let coeffs = Self::unpack_limb::<BF, EF>(circuit, value);

        // Observe each coefficient (matches native observe_algebra_element)
        for coeff in coeffs {
            self.observe(circuit, coeff);
        }
    }

    fn sample_ext(&mut self, circuit: &mut CircuitBuilder<EF>) -> Target {
        // Sample D base elements (matches native sample_algebra_element)
        let coeffs: Vec<_> = (0..EF::DIMENSION).map(|_| self.sample(circuit)).collect();

        // Recompose into extension element
        circuit
            .recompose_base_coeffs_to_ext_via_alu::<BF>(&coeffs)
            .expect("recomposition should succeed")
    }

    fn sample_bits(
        &mut self,
        circuit: &mut CircuitBuilder<EF>,
        num_bits: usize,
    ) -> Result<Vec<Target>, CircuitBuilderError> {
        // A verifier parameter with `num_bits > BF::bits()` would panic on the
        // `bits[..num_bits]` slice below, so we check and abort early if needed.
        let bf_bits = BF::bits();
        if num_bits > bf_bits {
            return Err(CircuitBuilderError::BinaryDecompositionTooManyBits {
                expected: bf_bits,
                n_bits: num_bits,
            });
        }
        let base_sample = self.sample(circuit);
        // Decompose base field element to bits
        // We decompose the full base field bit width to ensure correct reconstruction
        let bits = circuit.decompose_to_bits::<BF>(base_sample, bf_bits)?;
        Ok(bits[..num_bits].to_vec())
    }

    fn check_pow_witness(
        &mut self,
        circuit: &mut CircuitBuilder<EF>,
        witness_bits: usize,
        witness: Target,
    ) -> Result<(), CircuitBuilderError> {
        // When no PoW is required, keep challenger state unchanged
        if witness_bits == 0 {
            return Ok(());
        }

        // Observe witness as base field element
        self.observe(circuit, witness);

        // Sample and check leading bits are zero
        let bits = self.sample_bits(circuit, witness_bits)?;
        for bit in bits {
            circuit.assert_zero(bit);
        }

        Ok(())
    }

    fn clear(&mut self, circuit: &mut CircuitBuilder<EF>) {
        let zero = circuit.define_const(EF::ZERO);
        self.state = vec![zero; WIDTH];
        self.input_buffer.clear();
        self.output_buffer.clear();
        self.initialized = true;
        self.duplexed_once = false;
        // The next `D >= 2` permutation must open a fresh sponge chain on the zero state above.
        // Keeping the pre-clear permutation here would make it a continuation of that
        // permutation, chaining the old transcript's capacity into the new one.
        self.last_ext_perm = None;
    }
}
