use std::fmt::Debug;

pub trait LossFunc {
    /// Calculate the loss on a single target-value pair.
    fn eval(val: f32, target: f32) -> f32;
    /// Calculate the loss derivative on a single target-value pair.
    fn deriv(val: f32, target: f32) -> f32;
    /// Calculate the loss of all of the values.
    fn loss(val: &[f32], target: &[f32]) -> f32 {
        assert_eq!(val.len(), target.len());
        val.iter()
            .zip(target)
            .map(|(a, b)| Self::eval(*a, *b))
            .sum::<f32>()
            / val.len() as f32
    }
    /// Calculate the gradients and write them into `deriv`.
    fn gradients(val: &[f32], target: &[f32], deriv: &mut [f32]) {
        assert_eq!(val.len(), target.len());
        assert_eq!(val.len(), deriv.len());

        let len = val.len() as f32;
        for ((val, target), deriv) in val.iter().zip(target).zip(deriv) {
            *deriv = Self::deriv(*val, *target) / len;
        }
    }
}

#[derive(Clone)]
pub struct SquaredError;

impl Debug for SquaredError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SquaredError")
    }
}

impl LossFunc for SquaredError {
    fn eval(val: f32, target: f32) -> f32 {
        let diff = target - val;
        diff * diff
    }
    fn deriv(val: f32, target: f32) -> f32 {
        -2. * (target - val)
    }
}

#[derive(Clone)]
pub struct PolarError;

impl Debug for PolarError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("PolarError")
    }
}

impl LossFunc for PolarError {
    fn eval(val: f32, target: f32) -> f32 {
        let diff = target - val;
        diff * diff + 1. / (val * val + 1.)
    }
    
    fn deriv(val: f32, target: f32) -> f32 {
        let x = val * val + 1.;
        2. * (val * (1. - 1. / (x * x) - target))
    }
}
