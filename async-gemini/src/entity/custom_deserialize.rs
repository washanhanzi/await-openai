use serde::{
    de::{self, Deserializer, MapAccess, SeqAccess, Visitor},
    Deserialize,
};
use std::fmt;

pub fn deserialize_obj_or_arr<'de, T, D>(__deserializer: D) -> Result<Vec<T>, D::Error>
where
    T: Deserialize<'de>,
    D: Deserializer<'de>,
{
    struct VecVisitor<T> {
        marker: std::marker::PhantomData<T>,
    }

    impl<'de, T> Visitor<'de> for VecVisitor<T>
    where
        T: Deserialize<'de>,
    {
        type Value = Vec<T>;
        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("an object or array of objects")
        }

        fn visit_seq<S>(self, mut seq: S) -> Result<Self::Value, S::Error>
        where
            S: SeqAccess<'de>,
        {
            let size_hint = seq.size_hint().unwrap_or(5); // Use the hint if available, otherwise default to 0
            let mut v = Vec::with_capacity(size_hint);

            while let Some(bar) = seq.next_element()? {
                v.push(bar);
            }

            Ok(v)
        }

        fn visit_map<M>(self, map: M) -> Result<Self::Value, M::Error>
        where
            M: MapAccess<'de>,
        {
            let bar: T = Deserialize::deserialize(de::value::MapAccessDeserializer::new(map))?;
            Ok(vec![bar])
        }
    }
    Deserializer::deserialize_any(
        __deserializer,
        VecVisitor {
            marker: std::marker::PhantomData,
        },
    )
}

pub fn deserialize_option_obj_or_arr<'de, T, D>(
    __deserializer: D,
) -> Result<Option<Vec<T>>, D::Error>
where
    T: Deserialize<'de>,
    D: Deserializer<'de>,
{
    struct VecVisitor<T> {
        marker: std::marker::PhantomData<T>,
    }

    impl<'de, T> Visitor<'de> for VecVisitor<T>
    where
        T: Deserialize<'de>,
    {
        type Value = Option<Vec<T>>;
        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("an object or array of objects")
        }

        fn visit_seq<S>(self, mut seq: S) -> Result<Self::Value, S::Error>
        where
            S: SeqAccess<'de>,
        {
            let size_hint = seq.size_hint().unwrap_or(5); // Use the hint if available, otherwise default to 0
            let mut v = Vec::with_capacity(size_hint);

            while let Some(bar) = seq.next_element()? {
                v.push(bar);
            }

            Ok(Some(v))
        }

        fn visit_map<M>(self, map: M) -> Result<Self::Value, M::Error>
        where
            M: MapAccess<'de>,
        {
            let bar: Option<T> =
                Deserialize::deserialize(de::value::MapAccessDeserializer::new(map))?;
            match bar {
                None => Ok(None),
                Some(bar) => Ok(Some(vec![bar])),
            }
        }
    }
    Deserializer::deserialize_any(
        __deserializer,
        VecVisitor {
            marker: std::marker::PhantomData,
        },
    )
}
