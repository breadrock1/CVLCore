use crate::core::bounds::ColorBounds;
use crate::core::colors::*;
use crate::core::mat::CvlMat;
use crate::errors::{ProcessingError, ProcessingResult};

use ndarray::Array;

use opencv::core::{absdiff, cart_to_polar, count_non_zero, find_non_zero};
use opencv::core::{Mat, MatExprTraitConst, MatTrait, MatTraitConst, MatTraitConstManual};
use opencv::core::{Point, Rect, Scalar, Vector};
use opencv::core::{BORDER_DEFAULT, CV_32F, CV_64FC4, CV_8UC3};
use opencv::imgproc::{canny, cvt_color, sobel, threshold};
use opencv::imgproc::{COLOR_BGR2GRAY, THRESH_BINARY};

use std::ops::Deref;
use std::rc::Rc;

/// Transformations within RGB space like adding/removing the alpha channel, reversing the
/// channel order, conversion to/from 16-bit RGB color (R5:G6:B5 or R5:G5:B5), as well as
/// conversion to/from grayscale.
///
/// ## Parameters:
/// * frame: (&Mat) the passed video stream frame to transform.
#[inline(always)]
pub fn gen_grayscale_frame(frame: &CvlMat) -> ProcessingResult {
    let mut gray_frame = Mat::default();
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
#[inline(always)]
pub fn gen_threshold_frame(frame: &CvlMat, thresh: f64, maxval: f64) -> ProcessingResult {
    let mut gray: Mat = Mat::default();
    threshold(frame.frame(), &mut gray, thresh, maxval, THRESH_BINARY).unwrap();
    Ok(CvlMat::from(gray))
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
#[inline(always)]
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
#[inline(always)]
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

/// This method returns new Mat object with zeros by passed rows, columns and type parameters.
/// There is wrapper for [Mat::zeros] method.
///
/// ## Parameters:
/// * rows: (i32) a rows of Mat.
/// * cols: (i32) a columns of Mat.
/// * cv_type: (i32) a Mat type (like CV_64FC4).
#[inline(always)]
fn create_zeros_mat(rows: i32, cols: i32, cv_type: i32) -> Option<Mat> {
    let zeros_frame = Mat::zeros(rows, cols, cv_type).unwrap();
    zeros_frame.to_mat().ok()
}

/// There is wrapper for [Mat::roi] method which returns a sub-Mat object from source Mat and Rect.
/// By window parameter we get a sub-Mat with center point of (row, column).
///
/// ## Parameters:
/// * frame: (&Mat) a Mat to roi.
/// * rows: (i32) a rows of Mat point.
/// * cols: (i32) a columns of Mat point.
/// * window: (i32) an offset size.
#[inline(always)]
fn create_roi_mat(frame: &Mat, row: i32, col: i32, window: i32) -> Option<Mat> {
    let l_corn = Point::new(col - window, row - window);
    let r_corn = Point::new(col + window, row + window);
    let rect = Rect::from_points(l_corn, r_corn);
    Mat::roi(frame, rect).ok()
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

    let buffer = frame
        .frame()
        .data_typed::<u8>()
        .unwrap()
        .iter()
        .map(|d| *d as f64)
        .collect();

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
    let sobel_frame = gen_sobel_frame(mat_frame).unwrap();
    let g_x = sobel_frame.frame().clone();
    let g_y = sobel_frame.frame().clone();

    let mut magnitude = Mat::default();
    let mut orientation = Mat::default();
    cart_to_polar(&g_x, &g_y, &mut magnitude, &mut orientation, true).unwrap();

    let mut mask = Mat::default();
    threshold(&magnitude, &mut mask, thresh, maxval, THRESH_BINARY).unwrap();

    let scalar = Scalar::new(0.0, 0.0, 0.0, 0.0);
    let shape = (orientation.rows(), orientation.cols(), 3);
    let img_map = Mat::new_rows_cols_with_default(shape.0, shape.1, CV_8UC3, scalar).unwrap();

    // let mut nonzero_mask = VectorOfMat::default();
    // println!("{} {}", mat_frame.channels(), mat_frame.dims());
    // find_non_zero(&mat_frame, &mut nonzero_mask).unwrap();

    // let non_zero_count = count_non_zero(&orientation).unwrap();
    // let colored_scalar = match non_zero_count {
    //     val if val < neighbours => Scalar::from(BLACK_COLOR),
    //     val if val >= color_borders.get(4) => Scalar::from(RED_COLOR),
    //     val if val >= color_borders.get(3) => Scalar::from(YELLOW_COLOR),
    //     val if val >= color_borders.get(2) => Scalar::from(CYAN_COLOR),
    //     val if val >= color_borders.get(1) => Scalar::from(GREEN_COLOR),
    //     _ => Scalar::from(BLACK_COLOR),
    // };

    Ok(CvlMat::from(img_map.to_owned()))
}

/// Calculates the first, second, third, or mixed image derivatives using an extended Sobel operator.
/// The Sobel operators combine Gaussian smoothing and differentiation, so the result is more or less
/// resistant to the noise. Most often, the function is called with ( xorder = 1, yorder = 0, ksize = 3)
/// or ( xorder = 0, yorder = 1, ksize = 3) to calculate the first x- or y- image derivative.
/// The first case corresponds to a kernel of:
///
/// ## Parameters:
/// * frame: (&Mat) the passed video stream frame to transform.
#[inline(always)]
fn gen_sobel_frame(frame: &Mat) -> ProcessingResult {
    let mut g_x = Mat::default();
    sobel(frame, &mut g_x, CV_32F, 1, 0, 3, 1.0, 0f64, BORDER_DEFAULT).unwrap();
    Ok(CvlMat::new(g_x.to_owned()))
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
        let frame = frame_images.first().unwrap();
        let own_frame = frame.as_ref().to_owned();
        return Ok(own_frame);
    }

    let base_image = frame_images.last().unwrap();
    let sliced_array = &frame_images[0..frame_images.len() - 1];
    let differences: Vec<Rc<CvlMat>> = sliced_array
        .iter()
        .map(|m| gen_diff_frame(base_image.frame(), m.frame()).unwrap())
        .map(Rc::new)
        .collect();

    let result_image = gen_abs_frame(&differences)?;
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
        Some(frame) => Ok(frame.as_ref().to_owned()),
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
    color_bounds: &ColorBounds,
) -> ProcessingResult {
    let frame_mat = image.frame();
    let mut result_frame = create_zeros_mat(frame_mat.rows(), frame_mat.cols(), CV_64FC4).unwrap();

    let mut non_zero_pixels = Vector::<Point>::new();
    find_non_zero(frame_mat, &mut non_zero_pixels).unwrap();

    for non_zero_point in non_zero_pixels.to_vec() {
        let (row, col) = (non_zero_point.y, non_zero_point.x);
        if row == 0 || col == 0 {
            continue;
        }

        let roi_mat = create_roi_mat(frame_mat, row, col, window_size);
        if roi_mat.is_none() {
            continue;
        }

        let roi_matrix = &roi_mat.unwrap();
        let non_zero_count = count_non_zero(roi_matrix).unwrap();
        if non_zero_count < neighbours {
            continue;
        }

        let colored_scalar = match non_zero_count {
            val if val >= color_bounds.get(4) => Scalar::from(RED_COLOR),
            val if val >= color_bounds.get(3) => Scalar::from(YELLOW_COLOR),
            val if val >= color_bounds.get(2) => Scalar::from(CYAN_COLOR),
            val if val >= color_bounds.get(1) => Scalar::from(GREEN_COLOR),
            _ => Scalar::from(BLACK_COLOR),
        };

        result_frame
            .at_2d_mut::<Scalar>(row, col)
            .unwrap()
            .copy_from_slice(colored_scalar.as_slice());
    }

    Ok(CvlMat::from(result_frame))
}
