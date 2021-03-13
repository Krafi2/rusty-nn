use crate::f32s;

pub mod boxed_simd {
    use crate::helpers::as_scalar;

    use super::*;
    use serde::{
        de::{self, SeqAccess, Visitor},
        ser::{SerializeSeq, SerializeStruct},
        Deserializer, Serializer,
    };

    pub fn serialize<S>(aligned: &Box<[f32s]>, ser: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = ser.serialize_seq(Some(aligned.len() * f32s::lanes()))?;
        for i in as_scalar(aligned) {
            seq.serialize_element(i)?;
        }
        seq.end()
    }

    struct SimdVisitor;

    impl<'de> Visitor<'de> for SimdVisitor {
        type Value = Box<[f32s]>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a sequence of floats")
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            let mut vec = if let Some(len) = seq.size_hint() {
                if len % f32s::lanes() != 0 {
                    return Err(<A::Error as de::Error>::custom(format!(
                        "Number of elements must be a multiple of {}, received: {}",
                        f32s::lanes(),
                        len,
                    )));
                }
                Vec::with_capacity(len / f32s::lanes())
            } else {
                Vec::new()
            };

            let mut buffer = [0.; f32s::lanes()];
            'end: loop {
                for i in 0..f32s::lanes() {
                    match seq.next_element()? {
                        Some(e) => {
                            buffer[i] = e;
                        }
                        None => match i {
                            0 => break 'end,
                            _ => {
                                return Err(<A::Error as de::Error>::custom(format!(
                                    "Number of elements must be a multiple of {}, received: {}",
                                    f32s::lanes(),
                                    vec.len() * f32s::lanes() + i
                                )))
                            }
                        },
                    }
                }
                vec.push(f32s::from_slice_unaligned(&buffer))
            }
            Ok(vec.into_boxed_slice())
        }
    }

    pub fn deserialize<'de, D>(de: D) -> Result<Box<[f32s]>, D::Error>
    where
        D: Deserializer<'de>,
    {
        de.deserialize_seq(SimdVisitor)
    }
}
