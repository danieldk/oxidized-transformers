use candle_core::{CpuStorage, CustomOp1, Layout, Result, Shape, WithDType};

use crate::ops::strided_index::StridedIndex;

pub struct NonZero;

impl NonZero {
    fn cpu_n_nonzero<T: WithDType>(data: &[T], layout: &Layout) -> usize {
        match layout.contiguous_offsets() {
            Some((start_offset, end_offset)) => {
                let data = &data[start_offset..end_offset];
                data.iter().filter(|x| !x.is_zero()).count()
            }

            None => StridedIndex::from_layout(layout)
                .filter(|&x| !data[x].is_zero())
                .count(),
        }
    }

    fn cpu_fwd_impl<T: WithDType>(
        &self,
        data: &[T],
        layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let n_nonzero = Self::cpu_n_nonzero(data, layout);
        let index_len = layout.dims().len();

        let mut dst = Vec::with_capacity(n_nonzero * index_len);
        let dst_to_set = dst.spare_capacity_mut();
        let dst_to_set = unsafe { std::mem::transmute::<_, &mut [i64]>(dst_to_set) };

        let dst = match layout.contiguous_offsets() {
            // TODO: parallelize larger working sets.
            Some((start_offset, end_offset)) => {
                let data = &data[start_offset..end_offset];
                let mut offset = 0;
                for (mut idx, val) in data.iter().enumerate() {
                    if !val.is_zero() {
                        for (i, s) in layout.dims().iter().enumerate() {
                            dst_to_set[offset + i] = (idx % s) as i64;
                            idx /= s;
                        }
                        offset += index_len;
                    }

                    if offset == n_nonzero * index_len {
                        break;
                    }
                }
                unsafe { dst.set_len(n_nonzero * index_len) }

                dst
            }
            None => {
                let mut offset = 0;
                for (mut idx, source_idx) in StridedIndex::from_layout(layout).enumerate() {
                    if !data[source_idx].is_zero() {
                        for (i, &s) in layout.dims().iter().enumerate().rev() {
                            dst_to_set[offset + i] = (idx % s) as i64;
                            idx /= s;
                        }
                        offset += index_len;
                    }

                    if offset == n_nonzero * index_len {
                        break;
                    }
                }

                unsafe { dst.set_len(n_nonzero * index_len) }

                dst
            }
        };

        Ok((
            CpuStorage::I64(dst),
            Shape::from_dims(&[n_nonzero, index_len]),
        ))
    }
}

impl CustomOp1 for NonZero {
    fn name(&self) -> &'static str {
        "nonzero"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        match storage {
            CpuStorage::U8(data) => self.cpu_fwd_impl(data, layout),
            CpuStorage::U32(data) => self.cpu_fwd_impl(data, layout),
            CpuStorage::I64(data) => self.cpu_fwd_impl(data, layout),
            CpuStorage::BF16(data) => self.cpu_fwd_impl(data, layout),
            CpuStorage::F16(data) => self.cpu_fwd_impl(data, layout),
            CpuStorage::F32(data) => self.cpu_fwd_impl(data, layout),
            CpuStorage::F64(data) => self.cpu_fwd_impl(data, layout),
        }
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Tensor};
    use ndarray::{array, ArrayD};

    use crate::util::tests::IntoArrayD;

    use super::NonZero;

    #[test]
    fn nonzero_works() {
        let t = Tensor::eye(3, DType::F32, &Device::Cpu).unwrap();
        let r: ArrayD<i64> = t.apply_op1(NonZero).unwrap().into_arrayd().unwrap();
        assert_eq!(r, array![[0i64, 0], [1, 1], [2, 2]].into_arrayd().unwrap());

        let t = t.t().unwrap();
        let r: ArrayD<i64> = t.apply_op1(NonZero).unwrap().into_arrayd().unwrap();
        assert_eq!(r, array![[0i64, 0], [1, 1], [2, 2]].into_arrayd().unwrap());
    }
}
