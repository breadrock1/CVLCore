use opencv::core::{Mat, Scalar, Vector};
use opencv::core::{MatTrait, MatTraitConst, MatTraitConstManual};
use std::ops::Deref;

#[derive(Default, Clone)]
pub struct CvlMat {
    frame: Mat,
}

impl CvlMat {
    pub fn new(frame: Mat) -> Self {
        CvlMat { frame }
    }

    pub fn typ(&self) -> i32 {
        self.frame().typ()
    }

    pub fn dims(&self) -> i32 {
        self.frame().dims()
    }

    pub fn rows(&self) -> i32 {
        self.frame().rows()
    }

    pub fn columns(&self) -> i32 {
        self.frame().cols()
    }

    pub fn channels(&self) -> i32 {
        self.frame().channels()
    }

    pub fn bytes_data(&self) -> *const u8 {
        self.frame.data()
    }

    pub fn frame(&self) -> &Mat {
        &self.frame
    }

    pub fn frame_mut(&mut self) -> &mut Mat {
        &mut self.frame
    }

    pub fn to_slice(&self) -> opencv::Result<&[u8]> {
        let frame = self.frame();
        frame.data_bytes()
    }

    pub fn to_f64_vec(&self) -> opencv::Result<Vec<f64>> {
        let frame = self.frame();
        let data = frame
            .data_bytes()
            .unwrap()
            .iter()
            .map(|f| f64::from(*f))
            .collect::<Vec<f64>>();

        Ok(data)
    }

    pub fn to_scalar_vec(&self) -> Vec<f64> {
        let cvl_scalars: Vec<Vec<Scalar>> =
            self.frame().to_vec_2d::<Scalar>().unwrap_or(Vec::default());

        let mut new_vec = Vec::new();
        for row_elem in cvl_scalars.into_iter() {
            for scal in row_elem.into_iter() {
                new_vec.extend_from_slice(scal.0.as_slice());
            }
        }

        new_vec
    }

    pub fn new_with_data(rows: i32, cols: i32, typ: i32, bytes: &[u8]) -> Self {
        let mat = unsafe {
            let sizes = Vector::from_slice(&[rows, cols]);
            let mut m = Mat::new_nd_vec(&sizes, typ).unwrap();
            m.set_data(bytes.as_ptr().cast_mut());
            m
        };
        CvlMat::from(mat)
    }
}

impl From<Mat> for CvlMat {
    fn from(value: Mat) -> Self {
        CvlMat::new(value)
    }
}

impl Deref for CvlMat {
    type Target = Mat;

    fn deref(&self) -> &Self::Target {
        self.frame()
    }
}
