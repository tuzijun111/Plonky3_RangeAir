use alloc::vec;
use alloc::vec::Vec;

use p3_circuit::CircuitBuilder;
use p3_field::Field;
use p3_lookup::LookupProtocol;
use p3_lookup::logup::LogUpGadget;

use crate::Target;

/// Trait for converting a non-recursive type into its circuit representation.
///
/// Types implementing this trait can be used in recursive verification circuits.
/// The trait handles allocation of circuit targets and extraction of field element values.
pub trait Recursive<F: Field> {
    /// The non-recursive type associated with this recursive type.
    type Input;

    /// Creates a new instance of the recursive type by allocating targets in the circuit.
    ///
    /// This method should allocate all necessary public/private inputs and return
    /// a structure containing the allocated targets.
    ///
    /// # Parameters
    /// - `circuit`: Circuit builder to allocate targets in
    /// - `input`: The non-recursive input (used only for structure, not values)
    fn new(circuit: &mut CircuitBuilder<F>, input: &Self::Input) -> Self;

    /// Like [`new`](Self::new), but bakes the input's concrete field values in as
    /// circuit **constants** rather than allocating free public/private inputs.
    ///
    /// The default forwards to [`new`](Self::new) (i.e. free inputs), so behavior
    /// is unchanged unless a type overrides it. Override this for a commitment
    /// that must be pinned into a recursive verifier's *preprocessed* trace: a
    /// child's preprocessed commitment baked as a constant makes the parent's own
    /// verification key depend on it, so a single trusted root-VK check
    /// transitively anchors the whole recursion tree (closing the gap where the
    /// verifier trusts a proof-carried child VK). A type overriding this must not
    /// emit the corresponding values from [`get_values`](Self::get_values), since
    /// no public inputs are allocated for them.
    fn new_const(circuit: &mut CircuitBuilder<F>, input: &Self::Input) -> Self
    where
        Self: Sized,
    {
        Self::new(circuit, input)
    }

    /// Extracts private field element values from the input.
    ///
    /// Values returned here will be used to populate private inputs during execution.
    /// Default implementation returns an empty vector (no private inputs).
    ///
    /// # Parameters
    /// - `input`: The non-recursive input to extract private values from
    fn get_private_values(_input: &Self::Input) -> Vec<F> {
        vec![]
    }

    /// Extracts public field element values from the input.
    ///
    /// Values returned here will be used to populate public inputs during execution.
    /// The order must match the order in which targets were allocated in `new()`.
    ///
    /// # Parameters
    /// - `input`: The non-recursive input to extract public values from
    fn get_values(input: &Self::Input) -> Vec<F>;
}

pub trait RecursiveLookupGadget<F: Field>: LookupProtocol {
    fn verify_global_final_value_circuit(
        &self,
        circuit: &mut CircuitBuilder<F>,
        all_expected_cumulative: &[Target],
    );
}

impl<F: Field> RecursiveLookupGadget<F> for LogUpGadget {
    fn verify_global_final_value_circuit(
        &self,
        circuit: &mut CircuitBuilder<F>,
        all_expected_cumulative: &[Target],
    ) {
        let mut final_cumulative = circuit.define_const(F::ZERO);
        for a_e_c in all_expected_cumulative {
            final_cumulative = circuit.add(final_cumulative, *a_e_c);
        }

        circuit.assert_zero(final_cumulative);
    }
}
