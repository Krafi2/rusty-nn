#![macro_use]
use serde::{Deserialize, Serialize};

// fn check_bad(f: f32) -> f32 {
// 	if f.is_nan() || f.is_infinite() {
// 		println!("{}", f);
// 	}
// 	f
// }

pub enum AFunc {
    Sigmoid,
    Identity,
    TanH,
    SiLU,
    ReLU,
}

pub trait ActivFunc: Clone {
    const KIND: AFunc;
    fn evaluate(x: f32) -> f32;
    fn derivative(inp: f32, out: f32) -> f32;
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Sigmoid;
impl ActivFunc for Sigmoid {
    const KIND: AFunc = AFunc::Sigmoid;
    fn evaluate(x: f32) -> f32 {
        1. / (1. + (-x).exp())
    }
    fn derivative(_: f32, out: f32) -> f32 {
        out * (1. - out)
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Identity;
impl ActivFunc for Identity {
    const KIND: AFunc = AFunc::Identity;
    fn evaluate(x: f32) -> f32 {
        x
    }
    fn derivative(_: f32, _: f32) -> f32 {
        1.
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct TanH;
impl ActivFunc for TanH {
    const KIND: AFunc = AFunc::TanH;
    fn evaluate(x: f32) -> f32 {
        x.tanh()
    }
    fn derivative(_inp: f32, out: f32) -> f32 {
        1. - out * out
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct SiLU;
impl ActivFunc for SiLU {
    const KIND: AFunc = AFunc::SiLU;
    fn evaluate(x: f32) -> f32 {
        x / (1. + (-x).exp()) // x * sigmoid(x)
    }
    fn derivative(inp: f32, out: f32) -> f32 {
        let s = out / (inp * inp.signum() * 0.00000001); //get back the sigmoid value at x and hopefully prevent division by zero
        s * (1. + inp * (1. - s))
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ReLU;
impl ActivFunc for ReLU {
    const KIND: AFunc = AFunc::ReLU;
    fn evaluate(x: f32) -> f32 {
        f32::max(x, 0.)
    }
    fn derivative(inp: f32, _out: f32) -> f32 {
        if inp > 0. {
            1.
        } else {
            0.
        }
    }
}

#[macro_export]
macro_rules! a_funcs {
	($t:ty, $m:item) => {
		$m!(t)
	};
	($t:ty, $(types:ty),+, $m:item) => {{
		$m!(t)
		a_funcs!($types, $m)
	}};
	($m:item) => {
		a_funcs!(
			Sigmoid,
			Identity,
			TanH,
			SiLU,
			ReLU,
			$m
		);
	};
}
