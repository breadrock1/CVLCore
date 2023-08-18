use crate::core::bounds::ColorBounds;
use crate::core::colors::*;
use crate::errors::{ProcessingError, ProcessingResult};
use std::ops::Deref;
use std::rc::Rc;

use ndarray::Array;
use opencv::core::{absdiff, cart_to_polar, count_non_zero, find_non_zero};
use opencv::core::{Mat, MatExprTraitConst, MatTrait, MatTraitConst, MatTraitConstManual};
use opencv::core::{Point, Rect, Scalar, Vector};
use opencv::core::{BORDER_DEFAULT, CV_32F, CV_64FC4, CV_8UC3};
use opencv::imgproc::{canny, cvt_color, sobel, threshold};
use opencv::imgproc::{COLOR_BGR2GRAY, THRESH_BINARY};
use opencv::types::VectorOfMat;

#[derive(Default, Clone)]
pub struct CvlMat {
    frame: Mat,
}

impl CvlMat {
    pub fn new(frame: Mat) -> Self {
        CvlMat { frame }
    }

    pub fn frame(&self) -> &Mat {
        &self.frame
    }

    pub fn frame_mut(&mut self) -> &mut Mat {
        &mut self.frame
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

/// Transformations within RGB space like adding/removing the alpha channel, reversing the
/// channel order, conversion to/from 16-bit RGB color (R5:G6:B5 or R5:G5:B5), as well as
/// conversion to/from grayscale.
///
/// ## Parameters:
/// * frame: (&Mat) the passed video stream frame to transform.
#[inline]
pub fn gen_grayscale_frame(frame: &CvlMat) -> ProcessingResult {
    let mut gray_frame = Mat::default();
    //number of channels in the destination image
    cvt_color(frame.frame(), &mut gray_frame, COLOR_BGR2GRAY, 0).unwrap();
    Ok(CvlMat::from(gray_frame))
}

/// This method returns threshold image from passed bgr-image by passed black/white bounds
/// values. The simplest thresholding methods replace each pixel in an image with a black
/// pixel if the image intensity less than a fixed value called the threshold if the pixel
/// intensity is greater than that threshold. This function is necessary for further image
/// transformation to generate pixels vibration image.
///
/// ## Parameters
/// * frame: (&Mat) the passed video stream frame to transform.
/// * tresh: (f64) the black bound-value to swap pixel value to 1.
/// * maxval: (f64) the white bound-value to swap pixel value to 0.
#[inline]
pub fn gen_threshold_frame(frame: &CvlMat, thresh: f64, maxval: f64) -> ProcessingResult {
    let mut grayscale_frame: Mat = Mat::default();
    threshold(
        frame.frame(),
        &mut grayscale_frame,
        thresh,
        maxval,
        THRESH_BINARY,
    )
    .unwrap();
    Ok(CvlMat::from(grayscale_frame))
}

/// This method returns canny image from passed grayscale image by passed parameters.
/// The Canny edge detector is an edge detection operator that uses a multi-stage algorithm
/// to detect a wide range of edges in images. It was developed by John F. Canny in 1986.
/// Canny also produced a computational theory of edge detection explaining why the
/// technique works.
///
/// ## Parameters:
/// * frame: (&Mat) the passed video stream frame to transform.
/// * low: (f64) the first threshold for the hysteresis procedure.
/// * high: (f64) the second threshold for the hysteresis procedure.
/// * size: (i32) the aperture size of Sobel operator to generate Canny view.
/// * is_l2: (bool) the specifies the equation for finding gradient magnitude.
#[inline]
pub fn gen_canny_frame(
    frame: &CvlMat,
    low: f64,
    high: f64,
    size: i32,
    is_l2: bool,
) -> ProcessingResult {
    let mut canny_frame = Mat::default();
    canny(frame.frame(), &mut canny_frame, low, high, size, is_l2).unwrap();
    Ok(CvlMat::from(canny_frame))
}

/// This method returns canny image from passed grayscale image by passed parameters.
/// The Canny edge detector is an edge detection operator that uses a multi-stage algorithm
/// to detect a wide range of edges in images. It was developed by John F. Canny in 1986.
/// Canny also produced a computational theory of edge detection explaining why the
/// technique works.
///
/// ## Parameters:
/// * frame: (&Mat) the passed video stream frame to transform.
/// * size: (i32) the aperture size of Sobel operator to generate Canny view.
/// * sigma: (f64) the value to vary the percentage thresholds that are determined based on simple statistics.
/// * is_l2: (bool) the specifies the equation for finding gradient magnitude.
#[inline]
pub fn gen_canny_frame_by_sigma(
    frame: &CvlMat,
    size: i32,
    sigma: f64,
    is_l2: bool,
) -> ProcessingResult {
    let median = calculate_mat_median(frame).unwrap_or(0f64);
    let (low, high) = (1f64 - sigma + median, 1f64 + &sigma + median);

    let mut canny_frame = Mat::default();
    canny(frame.deref(), &mut canny_frame, low, high, size, is_l2).unwrap();

    Ok(CvlMat::from(canny_frame))
}

/// This method returns arithmetic mean (average) of all elements in array.
/// In mathematics and statistics, the arithmetic mean / arithmetic average is the sum of a
/// collection of numbers divided by the count of numbers in the collection. The collection
/// is often a set of results from an experiment, an observational study, or a survey. The
/// term "arithmetic mean" is preferred in some mathematics and statistics contexts because
/// it helps distinguish it from other types of means, such as geometric and harmonic.
///
/// ## Parameters:
/// * frame: (&Mat) a passed video stream frame to transform.
pub fn calculate_mat_median(frame: &CvlMat) -> Option<f64> {
    let mat_frame = frame.frame();
    let rows = mat_frame.rows() as usize;
    let cols = mat_frame.cols() as usize;
    let mut buffer = vec![0f64; rows * cols];

    let data = mat_frame.data_typed::<u8>().unwrap();

    for r in 0..rows {
        for c in 0..cols {
            let index = r * cols + c;
            buffer[index] = data[r * cols + c] as f64;
        }
    }

    Array::from_shape_vec((rows, cols), buffer).unwrap().mean()
}

/// This method returns distribution image from passed grayscale image by passed parameters.
/// The distribution image is representation of the distribution of pixel gradients intensities
/// in a digital image. As I mentioned in the introduction, image gradients are used as the
/// basic building blocks in many computer vision and image processing applications. However,
/// the network application of image gradients lies within edge detection.
///
/// ## Parameters:
/// * image: (&Mat) a passed video stream frame to transform.
/// * thresh: (f64) a black/white bound-value to thresholding source image.
/// * maxval: (f64) a maximum value to use with the thresholding types.
pub fn gen_distribution_frame(image: &CvlMat, thresh: f64, maxval: f64) -> ProcessingResult {
    let mat_frame = image.frame();
    let mut g_x = Mat::default();
    let mut g_y = Mat::default();
    sobel(
        mat_frame,
        &mut g_x,
        CV_32F,
        1,
        0,
        3,
        1.0,
        0 as f64,
        BORDER_DEFAULT,
    )
    .unwrap();
    sobel(
        mat_frame,
        &mut g_y,
        CV_32F,
        0,
        1,
        3,
        1.0,
        0 as f64,
        BORDER_DEFAULT,
    )
    .unwrap();

    let mut magnitude = Mat::default();
    let mut orientation = Mat::default();
    cart_to_polar(&g_x, &g_y, &mut magnitude, &mut orientation, true).unwrap();

    let mut mask = Mat::default();
    threshold(&magnitude, &mut mask, thresh, maxval, THRESH_BINARY).unwrap();

    let image_map_shape = (orientation.rows(), orientation.cols(), 3);
    let _image_map = Mat::new_rows_cols_with_default(
        image_map_shape.0,
        image_map_shape.1,
        CV_8UC3,
        Scalar::new(0.0, 0.0, 0.0, 0.0),
    )
    .unwrap();

    // let vibration_mask = image; //.clone();
    let mut nonzero_mask = VectorOfMat::default();
    find_non_zero(mat_frame, &mut nonzero_mask).unwrap();
    Ok(CvlMat::from(mat_frame.to_owned()))
}

/// There is wrapper method to invoke opencv::absdiff() method.
///
/// ## Parameters:
/// * img1: (&Mat) a first passed frame;
/// * img2: (&Mat) a second passed frame to sub;
#[inline]
fn gen_diff_frame(img1: &Mat, img2: &Mat) -> ProcessingResult {
    let mut tmp = Mat::default();
    absdiff(img1, img2, &mut tmp).unwrap();
    Ok(CvlMat::from(tmp))
}

/// This recursive method returns result-image of opencv::absdiff() method by passed
/// list of followed one by one frames of video stream. A result-image presents matrix
/// Absolute difference between two 2D-arrays when they have the same size and type
/// which used for removing from further analysis static pixels.
///
/// For example, we have both matrix:
///
/// 0 1 0       0 1 0      0 0 0
/// 1 0 1  and  1 1 1  =>  0 1 0
/// 0 1 0       0 1 0      0 0 0
///
/// ## Parameters:
/// * frame_images: (&Vec<Mat>) a list of video stream frames to get vibro-image;
pub fn gen_abs_frame(frame_images: &Vec<Rc<CvlMat>>) -> ProcessingResult {
    if frame_images.len() <= 1 {
        return Err(ProcessingError::GenAbs);
    }

    let mut differences: Vec<Rc<CvlMat>> = Vec::new();
    let base_image = frame_images.last().unwrap();
    let frame_images_len = frame_images.len();
    let sliced_array = &frame_images[0..frame_images_len - 1];
    for image in sliced_array.iter() {
        let diff_frame = gen_diff_frame(&base_image.clone(), &image.clone())?;
        differences.push(Rc::new(diff_frame));
    }

    let result_image = gen_abs_frame(&differences).unwrap();
    Ok(result_image)
}

/// This method returns reduced result-image of opencv::absdiff() method by passed
/// list of followed one by one frames of video stream. A result-image presents matrix
/// Absolute difference between two 2D-arrays when they have the same size and type
/// which used for removing from further analysis static pixels.
///
/// For example, we have both matrix:
///
/// 0 1 0       0 1 0      0 0 0
/// 1 0 1  and  1 1 1  =>  0 1 0
/// 0 1 0       0 1 0      0 0 0
///
/// ## Parameters:
/// * frame_images: (&Vec<Mat>) a list of video stream frames to get vibro-image;
pub fn gen_abs_frame_reduce(frame_images: &[Rc<CvlMat>]) -> ProcessingResult {
    let result = frame_images
        .iter()
        .cloned()
        .reduce(|img1, img2| Rc::new(gen_diff_frame(img1.frame(), img2.frame()).unwrap()));

    match result {
        None => Err(ProcessingError::GenAbs),
        Some(frame) => Ok(CvlMat::new(frame.frame.clone())),
    }
}

/// This method returns image with vibrating pixels (colored by bounds values) by passed image.
/// The main algorithm iterates over each pixel of Canny-image and calculate amount of nonzero
/// pixels around current pixel. A target computed value replaced instead pixel value.
/// The vibration image is network procedure which used for anxiety triggering by dispersion
/// of statistic values.
///
/// * image: (&Mat) a passed diff-image (results of abs) to transform.
/// * neighbours: (i32) a neighbours count value to filter noise of vibration.
pub fn compute_vibration(
    image: &CvlMat,
    neighbours: i32,
    window_size: i32,
    color_borders: &ColorBounds,
) -> ProcessingResult {
    let frame_mat = image.frame();
    let (rows, cols) = (frame_mat.rows(), frame_mat.cols());
    let zeros_frame = Mat::zeros(rows, cols, CV_64FC4).unwrap();
    let mut result_frame = zeros_frame.to_mat().unwrap();

    let mut non_zero_pixels = Vector::<Point>::new();
    find_non_zero(frame_mat, &mut non_zero_pixels).unwrap();
    for non_zero_point in non_zero_pixels.to_vec() {
        let (row, col) = (non_zero_point.y, non_zero_point.x);
        if row == 0 || col == 0 {
            continue;
        }
        let l_corn = Point::new(col - window_size, row - window_size);
        let r_corn = Point::new(col + window_size, row + window_size);
        let rect = Rect::from_points(l_corn, r_corn);
        let roi_mat = Mat::roi(&frame_mat.clone(), rect);
        if roi_mat.is_err() {
            continue;
        }

        let roi_matrix = &roi_mat.unwrap();
        let non_zero_count = count_non_zero(roi_matrix).unwrap();
        let colored_scalar = match non_zero_count {
            val if val < neighbours => Scalar::from(BLACK_COLOR),
            val if val >= color_borders.get(4) => Scalar::from(RED_COLOR),
            val if val >= color_borders.get(3) => Scalar::from(YELLOW_COLOR),
            val if val >= color_borders.get(2) => Scalar::from(CYAN_COLOR),
            val if val >= color_borders.get(1) => Scalar::from(GREEN_COLOR),
            _ => Scalar::from(BLACK_COLOR),
        };

        result_frame
            .at_2d_mut::<Scalar>(row, col)
            .unwrap()
            .copy_from_slice(colored_scalar.as_slice());
    }

    Ok(CvlMat::from(result_frame))
}
