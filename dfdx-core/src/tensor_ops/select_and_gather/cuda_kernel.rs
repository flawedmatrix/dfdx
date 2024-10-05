#[allow(unused_imports)]
use crate::{
    dtypes::*,
    shapes::{RemoveDimTo, ReplaceDimTo, Shape},
    tensor::{launch_cfg, Cuda, Error, Storage, Tensor},
};
use cudarc::driver::{DeviceSlice, LaunchAsync};

const GATHER_PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/gather.ptx"));
const SELECT_PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/select.ptx"));

pub(crate) trait HasCudaKernel<E> {
    const MOD_GATHER: &'static str;
    const FNS_GATHER: &'static [&'static str];
    const MOD_SELECT: &'static str;
    const FNS_SELECT: &'static [&'static str];
}

macro_rules! has_kernels {
    ($($dtype:ty),*) => {
        $(
        impl HasCudaKernel<$dtype> for Cuda {
            const MOD_GATHER: &'static str = concat!("gather_", stringify!($dtype));
            const MOD_SELECT: &'static str = concat!("select_", stringify!($dtype));
            const FNS_GATHER: &'static [&'static str] = &[concat!("gather_fwd_", stringify!($dtype))];
            const FNS_SELECT: &'static [&'static str] = &[concat!("select_fwd_", stringify!($dtype))];
        }
        )*
    }
}

has_kernels!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize, bool);

#[cfg(feature = "f16")]
impl HasCudaKernel<f16> for Cuda {
    const MOD_GATHER: &'static str = "gather_f16";
    const MOD_SELECT: &'static str = "select_f16";
    const FNS_GATHER: &'static [&'static str] = &["gather_fwd_f16", "gather_bwd_f16"];
    const FNS_SELECT: &'static [&'static str] = &["select_fwd_f16", "select_bwd_f16"];
}

#[cfg(feature = "f16")]
impl HasCudaKernel<f16> for Cuda {
    const MOD_GATHER: &'static str = "gather_f16";
    const MOD_SELECT: &'static str = "select_f16";
    const FNS_GATHER: &'static [&'static str] = &["gather_fwd_f16", "gather_bwd_f16"];
    const FNS_SELECT: &'static [&'static str] = &["select_fwd_f16", "select_bwd_f16"];
}

impl HasCudaKernel<f32> for Cuda {
    const MOD_GATHER: &'static str = "gather_f32";
    const MOD_SELECT: &'static str = "select_f32";
    const FNS_GATHER: &'static [&'static str] = &["gather_fwd_f32", "gather_bwd_f32"];
    const FNS_SELECT: &'static [&'static str] = &["select_fwd_f32", "select_bwd_f32"];
}

impl HasCudaKernel<f64> for Cuda {
    const MOD_GATHER: &'static str = "gather_f64";
    const MOD_SELECT: &'static str = "select_f64";
    const FNS_GATHER: &'static [&'static str] = &["gather_fwd_f64", "gather_bwd_f64"];
    const FNS_SELECT: &'static [&'static str] = &["select_fwd_f64", "select_bwd_f64"];
}

impl<E: Dtype> super::ReplaceDimKernel<E> for Cuda
where
    Self: HasCudaKernel<E>,
{
    fn forward<Src: Shape, Dst: Shape, Idx: Shape>(
        &self,
        inp: &Tensor<Src, E, Self>,
        idx: &Tensor<Idx, usize, Self>,
    ) -> Result<Tensor<Dst, E, Self>, Error>
    where
        Src: ReplaceDimTo<Dst, Idx>,
    {
        if !self.dev.has_func(Self::MOD_GATHER, Self::FNS_GATHER[0]) {
            self.dev
                .load_ptx(GATHER_PTX_SRC.into(), Self::MOD_GATHER, Self::FNS_GATHER)?;
        }

        let dst = inp.shape.replace(idx.shape);
        let numel = dst.num_elements();
        let mut storage = unsafe { self.alloc_empty::<E>(numel) }?;
        self.dev.memset_zeros(&mut storage)?;

        let inp_dims = self.dev.htod_copy(inp.shape.concrete().into())?;
        let idx_dims = self.dev.htod_copy(idx.shape.concrete().into())?;
        let inp_strides = self.dev.htod_copy(inp.strides.into())?;
        let idx_strides = self.dev.htod_copy(idx.strides.into())?;

        let fwd_fn = self
            .dev
            .get_func(Self::MOD_GATHER, Self::FNS_GATHER[0])
            .unwrap();
        let cfg = launch_cfg::<128>(numel as u32);
        let params = (
            numel,             // const size_t numel,
            inp.data.as_ref(), // const T *inp,
            Src::NUM_DIMS,     // const size_t inp_num_dims,
            &inp_dims,         // const size_t *inp_dims,
            &inp_strides,      // const size_t *inp_strides,
            idx.data.as_ref(), // const size_t *idx,
            Idx::NUM_DIMS,     // const size_t idx_num_dims,
            &idx_dims,         // const size_t *idx_dims,
            &idx_strides,      // const size_t *idx_strides,
            &mut storage,      // T *out,
            Dst::NUM_DIMS,     // const size_t out_num_dims,
        );
        unsafe { fwd_fn.launch(cfg, params) }?;

        Ok(self.build_tensor(dst, dst.strides(), storage))
    }

    fn backward<Src: Shape, Dst: Shape, Idx: Shape>(
        &self,
        inp: &Tensor<Src, E, Self>,
        grad_inp: &mut <Self as Storage<E>>::Vec,
        idx: &Tensor<Idx, usize, Self>,
        _: &Tensor<Dst, E, Self>,
        grad_out: &<Self as Storage<E>>::Vec,
    ) -> Result<(), Error>
    where
        Src: ReplaceDimTo<Dst, Idx>,
    {
        if !self.dev.has_func(Self::MOD_GATHER, Self::FNS_GATHER[1]) {
            self.dev
                .load_ptx(GATHER_PTX_SRC.into(), Self::MOD_GATHER, Self::FNS_GATHER)?;
        }

        let bwd_fn = self
            .dev
            .get_func(Self::MOD_GATHER, Self::FNS_GATHER[1])
            .unwrap();
        let numel = grad_out.len();

        let inp_dims = self.dev.htod_copy(inp.shape.concrete().into())?;
        let idx_dims = self.dev.htod_copy(idx.shape.concrete().into())?;
        let inp_strides = self.dev.htod_copy(inp.strides.into())?;
        let idx_strides = self.dev.htod_copy(idx.strides.into())?;

        let cfg = launch_cfg::<128>(numel as u32);
        let params = (
            numel,             // const size_t numel,
            grad_inp,          // T *grad_inp,
            Src::NUM_DIMS,     // const size_t inp_num_dims,
            &inp_dims,         // const size_t *inp_dims,
            &inp_strides,      // const size_t *inp_strides,
            idx.data.as_ref(), // const size_t *idx,
            Idx::NUM_DIMS,     // const size_t idx_num_dims,
            &idx_dims,         // const size_t *idx_dims,
            &idx_strides,      // const size_t *idx_strides,
            grad_out,          // const T *grad_out,
            Dst::NUM_DIMS,     // const size_t out_num_dims,
        );
        unsafe { bwd_fn.launch(cfg, params) }?;
        Ok(())
    }
}

impl<E: Dtype> super::RemoveDimKernel<E> for Cuda
where
    Self: HasCudaKernel<E>,
{
    fn forward<Src: Shape, Dst: Shape, Idx: Shape>(
        &self,
        inp: &Tensor<Src, E, Self>,
        idx: &Tensor<Idx, usize, Self>,
    ) -> Result<Tensor<Dst, E, Self>, Error>
    where
        Src: RemoveDimTo<Dst, Idx>,
    {
        if !self.dev.has_func(Self::MOD_SELECT, Self::FNS_SELECT[0]) {
            self.dev
                .load_ptx(SELECT_PTX_SRC.into(), Self::MOD_SELECT, Self::FNS_SELECT)?;
        }

        let dst = inp.shape.remove(idx.shape);
        let numel = dst.num_elements();
        let mut storage = unsafe { self.alloc_empty::<E>(numel) }?;
        self.dev.memset_zeros(&mut storage)?;

        let inp_dims = self.dev.htod_copy(inp.shape.concrete().into())?;
        let idx_dims = self.dev.htod_copy(idx.shape.concrete().into())?;
        let dst_dims = self.dev.htod_copy(dst.concrete().into())?;
        let inp_strides = self.dev.htod_copy(inp.strides.into())?;
        let idx_strides = self.dev.htod_copy(idx.strides.into())?;
        let dst_strides = self.dev.htod_copy(dst.strides().into())?;

        let fwd_fn = self
            .dev
            .get_func(Self::MOD_SELECT, Self::FNS_SELECT[0])
            .unwrap();
        let cfg = launch_cfg::<128>(numel as u32);
        let params = (
            numel,             // const size_t numel,
            inp.data.as_ref(), // const T *inp,
            Src::NUM_DIMS,     // const size_t inp_num_dims,
            &inp_dims,         // const size_t *inp_dims,
            &inp_strides,      // const size_t *inp_strides,
            idx.data.as_ref(), // const size_t *idx,
            Idx::NUM_DIMS,     // const size_t idx_num_dims,
            &idx_dims,         // const size_t *idx_dims,
            &idx_strides,      // const size_t *idx_strides,
            &mut storage,      // T *out,
            &dst_dims,         // const size_t *out_dims,
            &dst_strides,      // const size_t *out_strides
        );
        unsafe { fwd_fn.launch(cfg, params) }?;

        Ok(self.build_tensor(dst, dst.strides(), storage))
    }

    fn backward<Src: Shape, Dst: Shape, Idx: Shape>(
        &self,
        inp: &Tensor<Src, E, Self>,
        grad_inp: &mut <Self as Storage<E>>::Vec,
        idx: &Tensor<Idx, usize, Self>,
        out: &Tensor<Dst, E, Self>,
        grad_out: &<Self as Storage<E>>::Vec,
    ) -> Result<(), Error>
    where
        Src: RemoveDimTo<Dst, Idx>,
    {
        if !self.dev.has_func(Self::MOD_SELECT, Self::FNS_SELECT[1]) {
            self.dev
                .load_ptx(SELECT_PTX_SRC.into(), Self::MOD_SELECT, Self::FNS_SELECT)?;
        }

        let bwd_fn = self
            .dev
            .get_func(Self::MOD_SELECT, Self::FNS_SELECT[1])
            .unwrap();
        let numel = grad_out.len();

        let inp_dims = self.dev.htod_copy(inp.shape.concrete().into())?;
        let idx_dims = self.dev.htod_copy(idx.shape.concrete().into())?;
        let out_dims = self.dev.htod_copy(out.shape.concrete().into())?;
        let inp_strides = self.dev.htod_copy(inp.strides.into())?;
        let idx_strides = self.dev.htod_copy(idx.strides.into())?;
        let out_strides = self.dev.htod_copy(out.strides.into())?;

        let cfg = launch_cfg::<128>(numel as u32);
        let params = (
            numel,             // const size_t numel,
            grad_inp,          // T *grad_inp,
            Src::NUM_DIMS,     // const size_t inp_num_dims,
            &inp_dims,         // const size_t *inp_dims,
            &inp_strides,      // const size_t *inp_strides,
            idx.data.as_ref(), // const size_t *idx,
            Idx::NUM_DIMS,     // const size_t idx_num_dims,
            &idx_dims,         // const size_t *idx_dims,
            &idx_strides,      // const size_t *idx_strides,
            grad_out,          // const T *grad_out,
            &out_dims,         // const size_t *out_dims,
            &out_strides,      // const size_t *out_strides
        );
        unsafe { bwd_fn.launch(cfg, params) }?;
        Ok(())
    }
}
