#[cfg(not(doctest))]
pub mod api;
pub mod core;
pub mod errors;
pub mod ui;

use crate::core::bounds::*;
use crate::core::mat::CvlMat;
use crate::core::statistic::{Dispersion, Statistic};
use crate::errors::{ProcessingError, ProcessingResult};

use ndarray::{Array, Array1};

use opencv::boxed_ref::BoxedRef;
use opencv::core::{absdiff, cart_to_polar, count_non_zero, find_non_zero};
use opencv::core::{Mat, MatExprTraitConst, MatTrait, MatTraitConst, MatTraitConstManual};
use opencv::core::{Point, Rect, Scalar, Vector};
use opencv::core::{BORDER_DEFAULT, CV_32F, CV_64FC4, CV_8UC3};
use opencv::imgproc::{canny, cvt_color, sobel, threshold};
use opencv::imgproc::{COLOR_BGR2GRAY, THRESH_BINARY};

use std::ops::Deref;
use std::rc::Rc;

const CHANNELS_COUNT: usize = 4;
const POW_DIFF_VALUE: u32 = 2;
pub const BGR_CV_IMAGE: i32 = 16;
pub const ANY_2_DIM_IMAGE: i32 = 0;

/// Transformations within RGB space like adding/removing the alpha channel, reversing the
/// channel order, conversion to/from 16-bit RGB color (R5:G6:B5 or R5:G5:B5), as well as
/// conversion to/from grayscale.
///
/// ![grayscale](/resources/grayscale.jpg "Example of Grayscale image")
///
/// ## Parameters:
/// * frame: (&CvlMat) the passed video stream frame to transform.
///
/// ## Returns:
/// Returns `Ok(CvlMat)` on success, otherwise returns an error.
///
/// ## Errors:
/// Returns [`GenGrayScale`](ProcessingError::GenGrayScale) if failed while trying to
/// transform passed image to grayscale image.
#[inline(always)]
pub fn gen_grayscale_frame(frame: &CvlMat) -> ProcessingResult {
    let mut gray_frame = Mat::default();
    if let Err(err) = cvt_color(frame.frame(), &mut gray_frame, COLOR_BGR2GRAY, 0) {
        return Err(ProcessingError::GenGrayScale(err.message));
    }

    Ok(CvlMat::from(gray_frame))
}

/// This method returns threshold image from passed bgr-image by passed black/white bounds
/// values. The simplest thresholding methods replace each pixel in an image with a black
/// pixel if the image intensity less than a fixed value called the threshold if the pixel
/// intensity is greater than that threshold. This function is necessary for further image
/// transformation to generate pixels vibration image.
///
/// ## Parameters
/// * frame: (&CvlMat) the passed video stream frame to transform.
/// * tresh: (f64) the black bound-value to swap pixel value to 1.
/// * maxval: (f64) the white bound-value to swap pixel value to 0.
///
/// ## Returns:
/// Returns `Ok(CvlMat)` on success, otherwise returns an error.
///
/// ## Errors:
/// Returns [`GenThreshold`](ProcessingError::GenThreshold) if failed while trying to
/// transform passed image to threshold image.
#[inline(always)]
pub fn gen_threshold_frame(frame: &CvlMat, thresh: f64, maxval: f64) -> ProcessingResult {
    let mut gray: Mat = Mat::default();
    if let Err(err) = threshold(frame.frame(), &mut gray, thresh, maxval, THRESH_BINARY) {
        return Err(ProcessingError::GenThreshold(err.message));
    }

    Ok(CvlMat::from(gray))
}

/// This method returns canny image from passed grayscale image by passed parameters.
/// The Canny edge detector is an edge detection operator that uses a multi-stage algorithm
/// to detect a wide range of edges in images. It was developed by John F. Canny in 1986.
/// Canny also produced a computational theory of edge detection explaining why the
/// technique works.
///
/// ![canny](/resources/canny.jpg "Example of Canny image")
///
/// ## Parameters:
/// * frame: (&CvlMat) the passed video stream frame to transform.
/// * low: (f64) the first threshold for the hysteresis procedure.
/// * high: (f64) the second threshold for the hysteresis procedure.
/// * size: (i32) the aperture size of Sobel operator to generate Canny view.
/// * is_l2: (bool) the specifies the equation for finding gradient magnitude.
///
/// ## Returns:
/// Returns `Ok(CvlMat)` on success, otherwise returns an error.
///
/// ## Errors:
/// Returns [`GenCanny`](ProcessingError::GenCanny) if failed while trying to
/// transform passed image to canny image.
#[inline(always)]
pub fn gen_canny_frame(
    frame: &CvlMat,
    low: f64,
    high: f64,
    size: i32,
    is_l2: bool,
) -> ProcessingResult {
    let mut canny_frame = Mat::default();
    if let Err(err) = canny(frame.frame(), &mut canny_frame, low, high, size, is_l2) {
        return Err(ProcessingError::GenCanny(err.message))
    }

    Ok(CvlMat::from(canny_frame))
}

/// This method returns canny image from passed grayscale image by passed parameters.
/// The Canny edge detector is an edge detection operator that uses a multi-stage algorithm
/// to detect a wide range of edges in images. It was developed by John F. Canny in 1986.
/// Canny also produced a computational theory of edge detection explaining why the
/// technique works.
///
/// ![canny](/resources/canny.jpg "Example of Canny image")
///
/// ## Parameters:
/// * frame: (&CvlMat) the passed video stream frame to transform.
/// * size: (i32) the aperture size of Sobel operator to generate Canny view.
/// * sigma: (f64) the value to vary the percentage thresholds that are determined based on simple statistics.
/// * is_l2: (bool) the specifies the equation for finding gradient magnitude.
///
/// ## Returns:
/// Returns `Ok(CvlMat)` on success, otherwise returns an error.
///
/// ## Errors:
/// Returns [`GenCanny`](ProcessingError::GenCanny) if failed while trying to
/// transform passed image to canny image.
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
    if let Err(err) = canny(frame.deref(), &mut canny_frame, low, high, size, is_l2) {
        return Err(ProcessingError::GenCanny(err.message))
    }

    Ok(CvlMat::from(canny_frame))
}

/// This method returns new Mat object with zeros by passed rows, columns and type parameters.
/// There is wrapper for [Mat::zeros] method.
///
/// ## Parameters:
/// * rows: (i32) a rows of Mat.
/// * cols: (i32) a columns of Mat.
/// * cv_type: (i32) a Mat type (like CV_64FC4).
///
/// ## Returns:
/// Returns `Option<Mat>` of executing [`Mat::zeros`] method from opencv library.
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
///
/// ## Returns:
/// Returns `Option<Mat>` of executing [`Mat::roi`] method from opencv library.
#[inline(always)]
fn create_roi_mat(frame: &Mat, row: i32, col: i32, window: i32) -> Option<BoxedRef<Mat>> {
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
/// * frame: (&CvlMat) a passed video stream frame to transform.
///
/// ## Results:
/// Returns `Option<f64>` of executing [`Array::mean`] method from ndarray library.
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
/// * image: (&CvlMat) a passed video stream frame to transform.
/// * thresh: (f64) a black/white bound-value to thresholding source image.
/// * maxval: (f64) a maximum value to use with the thresholding types.
///
/// ## Returns:
/// Returns `Ok(CvlMat)` on success, otherwise returns an error.
///
/// ## Errors:
/// Returns [`GenDistribution`](ProcessingError::GenDistribution) if failed while trying to
/// transform passed image to distribution image.
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
///
/// ## Returns:
/// Returns `Ok(CvlMat)` of executing [`sobel`] method of opencv library.
///
/// ## Errors:
/// Returns [`GenSobel`](ProcessingError::GenSobel) if failed while trying to
/// transform passed image to distribution image.
#[inline(always)]
fn gen_sobel_frame(frame: &Mat) -> ProcessingResult {
    let mut g_x = Mat::default();
    if let Err(err) = sobel(frame, &mut g_x, CV_32F, 1, 0, 3, 1.0, 0f64, BORDER_DEFAULT) {
        return Err(ProcessingError::GenSobel(err.message));
    }

    Ok(CvlMat::new(g_x.to_owned()))
}

/// There is wrapper method to invoke opencv::absdiff() method.
///
/// ## Parameters:
/// * img1: (&Mat) a first passed frame;
/// * img2: (&Mat) a second passed frame to sub;
///
/// ## Returns:
/// Returns `Ok(CvlMat)` on success, otherwise returns an error.
///
/// ## Errors:
/// Returns [`GenDifferences`](ProcessingError::GenDifferences) if failed while trying to
/// execute [`absdiff`] method for passed images.
#[inline]
fn gen_diff_frame(img1: &Mat, img2: &Mat) -> ProcessingResult {
    let mut tmp = Mat::default();
    if let Err(err) = absdiff(img1, img2, &mut tmp) {
        return Err(ProcessingError::GenDifferences(err.message))
    }

    Ok(CvlMat::from(tmp))
}

/// This recursive method returns result-image of opencv::absdiff() method by passed
/// list of followed one by one frames of video stream. A result-image presents matrix
/// Absolute difference between two 2D-arrays when they have the same size and type
/// which used for removing from further analysis static pixels.
///
/// ![difference](/resources/difference.jpg "Example of Difference image")
///
/// For example, we have both matrix:
///
///     0 1 0       0 1 0      0 0 0
///     1 0 1  and  1 1 1  =>  0 1 0
///     0 1 0       0 1 0      0 0 0
///
/// ## Parameters:
/// * frame_images: (&[`Rc<CvlMat>`]) a list of video stream frames to get difference-image;
///
/// ## Returns:
/// Returns `Ok(CvlMat)` on success, otherwise returns an error.
///
/// ## Errors:
/// Returns [`GenAbs`](ProcessingError::GenAbs) if failed while trying to generate difference
/// image from passed set of canny images.
pub fn gen_abs_frame(frame_images: &[Rc<CvlMat>]) -> ProcessingResult {
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
/// ![difference](/resources/difference.jpg "Example of Difference image")
///
/// For example, we have both matrix:
///
///     0 1 0       0 1 0      0 0 0
///     1 0 1  and  1 1 1  =>  0 1 0
///     0 1 0       0 1 0      0 0 0
///
/// ## Parameters:
/// * frame_images: (&[`Rc<CvlMat>`]) a list of video stream frames to get difference-image;
///
/// ## Returns:
/// Returns `Ok(CvlMat)` on success, otherwise returns an error.
///
/// ## Errors:
/// Returns [`GenAbs`](ProcessingError::GenAbs) if failed while trying to generate difference
/// image from passed set of canny images.
pub fn gen_abs_frame_reduce(frame_images: &[Rc<CvlMat>]) -> ProcessingResult {
    let result = frame_images
        .iter()
        .cloned()
        .reduce(|img1, img2| {
            Rc::new(gen_diff_frame(img1.frame(), img2.frame()).unwrap())
        });

    result
        .map(|it| it.as_ref().to_owned())
        .ok_or(ProcessingError::GenAbs)
}

/// This method returns image with vibrating pixels (colored by bounds values) by passed image.
/// The main algorithm iterates over each pixel of Canny-image and calculate amount of nonzero
/// pixels around current pixel. A target computed value replaced instead pixel value.
/// The vibration image is network procedure which used for anxiety triggering by dispersion
/// of statistic values.
///
/// ![vibration](/resources/vibration.jpg "Example of Vibration image")
///
/// ## Parameters:
/// * image: (&CvlMat) a passed diff-image (results of abs) to transform.
/// * neighbours: (i32) a neighbours count value to filter noise of vibration.
/// * window_size: (i32) a offset from central pixel to compute non-null pixel neighbours.
/// * color_bounds: (&ColorBounds) a object with channels values to set color for pixels.
///
/// ## Returns:
/// Returns `Ok(CvlMat)` on success, otherwise returns an error.
///
/// ## Errors:
/// Returns [`ComputeVibration`](ProcessingError::ComputeVibration) if failed while trying to
/// transform difference image to vibration image.
pub fn compute_vibration(
    image: &CvlMat,
    neighbours: i32,
    window_size: i32,
    color_bounds: &ColorBounds,
) -> ProcessingResult {
    let frame_mat = image.frame();
    let mut statistic = Statistic::default();
    let Some(mut result_frame) = create_zeros_mat(frame_mat.rows(), frame_mat.cols(), CV_64FC4) else {
        let msg = "returned empty zeros mat".to_string();
        return Err(ProcessingError::ComputeVibration(msg));
    };

    let mut non_zero_pixels = Vector::<Point>::new();
    find_non_zero(frame_mat, &mut non_zero_pixels).unwrap();

    for non_zero_point in non_zero_pixels.into_iter() {
        let (row, col) = (non_zero_point.y, non_zero_point.x);
        if row == 0 || col == 0 {
            continue;
        }

        let Some(roi_mat) = create_roi_mat(frame_mat, row, col, window_size) else {
            continue;
        };

        let Ok(non_zero_count) = count_non_zero(&roi_mat) else {
            continue;
        };

        let colored_scalar = match non_zero_count {
            val if val >= color_bounds.get(4) => {
                statistic.ch4 += 1;
                Scalar::from(RED_COLOR)
            }
            val if val >= color_bounds.get(3) => {
                statistic.ch3 += 1;
                Scalar::from(YELLOW_COLOR)
            }
            val if val >= color_bounds.get(2) => {
                statistic.ch2 += 1;
                Scalar::from(CYAN_COLOR)
            }
            val if val >= color_bounds.get(1) => {
                statistic.ch1 += 1;
                Scalar::from(GREEN_COLOR)
            }
            _ => Scalar::from(BLACK_COLOR),
        };

        let Ok(scalar) = result_frame.at_2d_mut::<Scalar>(row, col) else {
            continue;
        };

        scalar.copy_from_slice(colored_scalar.as_slice());
    }

    let mut cvlmat = CvlMat::from(result_frame);
    cvlmat.set_statistic(statistic);

    Ok(cvlmat)
}

///
pub fn compute_statistic(history_stats: Vec<&Statistic>, normalization: f32) -> Dispersion {
    let stats_arrays: Vec<_> = history_stats
        .into_iter()
        .map(|st| [st.ch1, st.ch2, st.ch3, st.ch4])
        .map(|sl| Array1::from_shape_vec(CHANNELS_COUNT, sl.to_vec()).unwrap())
        .collect();

    let stats_medians: &Vec<u16> = &stats_arrays
        .iter()
        .map(Array::mean)
        .map(Option::unwrap)
        .collect();

    let mut tmp_slice = [0f32; CHANNELS_COUNT];
    stats_arrays.into_iter().for_each(|array| {
        compute_math_expectation(&mut tmp_slice, &array, stats_medians);
    });

    Dispersion::from(
        tmp_slice
            .into_iter()
            .map(f32::sqrt)
            .map(|val| val / normalization)
            .collect::<Vec<f32>>(),
    )
}

///
fn compute_math_expectation(tmp_slice: &mut [f32; 4], array: &Array1<u16>, medians: &[u16]) {
    (0..CHANNELS_COUNT).for_each(|index| {
        let carr = *array.get(index).unwrap() as i32;
        let cmed = *medians.get(index).unwrap() as i32;
        let diff = (carr - cmed).pow(POW_DIFF_VALUE) as f32;
        *tmp_slice.get_mut(index).unwrap() += diff;
    });
}
