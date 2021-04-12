use std::fmt::Debug;

pub struct Loss {
    pub loss: f32,
    pub grads: Vec<f32>,
}

pub trait LossFunc {
    fn eval(&self, val: &[f32], target: &[f32]) -> Loss;
}

#[derive(Clone, Debug, Default)]
pub struct MeanSquared;

impl LossFunc for MeanSquared {
    fn eval(&self, val: &[f32], target: &[f32]) -> Loss {
        assert_eq!(
            val.len(),
            target.len(),
            "Value vector must be the same length as target vector. val: {}, target: {}",
            val.len(),
            target.len()
        );

        let recip = 1. / val.len() as f32;
        let mut grads = Vec::with_capacity(val.len());
        let mut loss = 0.;
        for (val, target) in val.iter().copied().zip(target.iter().copied()) {
            let diff = val - target;
            loss += diff * diff;
            let deriv = 2. * (val - target) * recip;
            grads.push(deriv);
        }
        grads.shrink_to_fit();

        Loss {
            loss: loss * recip,
            grads,
        }
    }
}
