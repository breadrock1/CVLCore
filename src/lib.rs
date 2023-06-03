pub mod cvldetector {
    use std::ops::Deref;

    use ndarray::prelude::Array;
    use opencv::core::{absdiff, cart_to_polar, count_non_zero, find_non_zero};
    use opencv::core::{Mat, MatExprTraitConst, MatTrait, MatTraitConst, MatTraitConstManual};
    use opencv::core::{Point, Rect, Scalar, Vector};
    use opencv::core::{BORDER_DEFAULT, CV_32F, CV_64FC4, CV_8UC3};
    use opencv::imgproc;
    use opencv::types::VectorOfMat;

    /// A red color pixel value used for marking magnitude and vibration Mat object.
    pub const RED_COLOR: (f64, f64, f64, f64) = (0.0, 0.0, 255.0, 0.0);

    /// A cyan color pixel value used for marking magnitude and vibration Mat object.
    pub const CYAN_COLOR: (f64, f64, f64, f64) = (255.0, 255.0, 0.0, 0.0);

    /// A green color pixel value used for marking magnitude and vibration Mat object.
    pub const GREEN_COLOR: (f64, f64, f64, f64) = (0.0, 255.0, 0.0, 0.0);

    /// A yellow color pixel value used for marking magnitude and vibration Mat object.
    pub const YELLOW_COLOR: (f64, f64, f64, f64) = (0.0, 255.0, 255.0, 0.0);

    /// A black color pixel value used for marking magnitude and vibration Mat object.
    pub const BLACK_COLOR: (f64, f64, f64, f64) = (0.0, 0.0, 0.0, 0.0);

    pub struct ColorBounds {
        channel_1: i32,
        channel_2: i32,
        channel_3: i32,
        channel_4: i32
    }

    impl ColorBounds {
        pub fn new(ch1: i32, ch2: i32, ch3: i32, ch4: i32) -> Self {
            ColorBounds {
                channel_1: ch1,
                channel_2: ch2,
                channel_3: ch3,
                channel_4: ch4
            }
        }
    }

    impl Default for ColorBounds {
        fn default() -> Self {
            ColorBounds {
                channel_1: 8,
                channel_2: 9,
                channel_3: 10,
                channel_4: 11
            }
        }
    }

    /// Transformations within RGB space like adding/removing the alpha channel, reversing the
    /// channel order, conversion to/from 16-bit RGB color (R5:G6:B5 or R5:G5:B5), as well as
    /// conversion to/from grayscale.
    ///
    /// ## Parameters:
    /// * frame: (&Mat) the passed video stream frame to transform.
    #[inline]
    pub fn gen_grayscale_frame(frame: &Mat) -> opencv::Result<Mat> {
        let mut gray_frame = Mat::default();
        imgproc::cvt_color(&frame, &mut gray_frame, imgproc::COLOR_BGR2GRAY, 0).unwrap();
        Ok(gray_frame)
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
    pub fn gen_threshold_frame(frame: &Mat, thresh: f64, maxval: f64) -> opencv::Result<Mat> {
        let mut grayscale_frame: Mat = Mat::default();
        imgproc::threshold(
            &frame,
            &mut grayscale_frame,
            thresh,
            maxval,
            imgproc::THRESH_BINARY,
        )
        .unwrap();

        Ok(grayscale_frame)
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
        frame: &Mat,
        low: f64,
        high: f64,
        size: i32,
        is_l2: bool,
    ) -> opencv::Result<Mat> {
        let mut canny_frame = Mat::default();
        imgproc::canny(&frame, &mut canny_frame, low, high, size, is_l2).unwrap();

        Ok(canny_frame)
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
        frame: &Mat,
        size: i32,
        sigma: f64,
        is_l2: bool,
    ) -> opencv::Result<Mat> {
        let median = calculate_mat_median(frame);
        let (low, high) = (1.0 - sigma + median, 1.0 + &sigma + &median);

        let mut canny_frame = Mat::default();
        imgproc::canny(&frame, &mut canny_frame, low, high, size, is_l2).unwrap();

        Ok(canny_frame)
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
    pub fn calculate_mat_median(frame: &Mat) -> f64 {
        let rows = frame.rows() as usize;
        let cols = frame.cols() as usize;
        let mut buffer = vec![0.0; rows * cols];
        let data = frame.data_typed::<u8>().expect("");

        for r in 0..rows {
            for c in 0..cols {
                let index = &r * &cols + &c;
                buffer[index] = data[&r * &cols + &c] as f64;
            }
        }

        Array::from_shape_vec((rows, cols), buffer)
            .unwrap()
            .mean()
            .unwrap()
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
    pub fn gen_distribution_frame(
        image: &Mat,
        thresh: f64,
        maxval: f64,
    ) -> Result<Mat, opencv::Error> {
        let mut g_x = Mat::default();
        let mut g_y = Mat::default();
        imgproc::sobel(
            image,
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
        imgproc::sobel(
            image,
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
        imgproc::threshold(
            &magnitude,
            &mut mask,
            thresh,
            maxval,
            imgproc::THRESH_BINARY,
        )
        .unwrap();

        let image_map_shape = (orientation.rows(), orientation.cols(), 3);
        let mut _image_map = Mat::new_rows_cols_with_default(
            image_map_shape.0,
            image_map_shape.1,
            CV_8UC3,
            Scalar::new(0.0, 0.0, 0.0, 0.0),
        )
        .unwrap();

        let vibration_mask = image.clone();
        let mut nonzero_mask = VectorOfMat::default();
        find_non_zero(image, &mut nonzero_mask).unwrap();
        Ok(vibration_mask)
    }

    /// There is wrapper method to invoke opencv::absdiff() method.
    ///
    /// ## Parameters:
    /// * img1: (&Mat) a first passed frame;
    /// * img2: (&Mat) a second passed frame to sub;
    #[inline]
    fn gen_diff_frame(img1: &Mat, img2: &Mat) -> opencv::Result<Mat> {
        let mut tmp = Mat::default();
        absdiff(&img1, &img2, &mut tmp).unwrap();
        Ok(tmp)
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
    pub fn gen_abs_frame(frame_images: &Vec<Mat>) -> opencv::Result<Mat> {
        if frame_images.len() <= 1 {
            let result_image = frame_images.first().unwrap();
            let result_img = result_image.deref();
            return Ok(result_img.clone());
        }

        let mut differences: Vec<Mat> = Vec::new();
        let base_image = frame_images.last().unwrap();
        let frame_images_len = frame_images.len();
        let sliced_array = &frame_images[0..frame_images_len - 1];
        for image in sliced_array.iter() {
            let test = gen_diff_frame(base_image, image).unwrap();
            differences.push(test);
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
    pub fn gen_abs_frame_reduce(frame_images: &[Mat]) -> opencv::Result<Mat> {
        let result = frame_images
            .iter()
            .cloned()
            .reduce(|img1, img2| gen_diff_frame(&img1, &img2).unwrap());

        Ok(result.unwrap())
    }

    /// This method returns image with vibrating pixels (colored by bounds values) by passed image.
    /// The main algorithm iterates over each pixel of Canny-image and calculate amount of nonzero
    /// pixels around current pixel. A target computed value replaced instead pixel value.
    /// The vibration image is network procedure which used for anxiety triggering by dispersion
    /// of statistic values.
    ///
    /// * image: (&Mat) a passed diff-image (results of abs) to transform.
    /// * neighbours: (i32) a neighbours count value to filter noise of vibration.
    pub fn compute_vibrating_pixels(
        image: &Mat,
        neighbours: i32,
        window_size: i32,
        color_borders: &ColorBounds,
    ) -> Result<Mat, opencv::Error> {
        let (rows, cols) = (image.rows(), image.cols());
        let zeros_frame = Mat::zeros(rows, cols, CV_64FC4).unwrap();
        let mut result_frame = zeros_frame.to_mat().unwrap();

        let mut non_zero_pixels = Vector::<Point>::new();
        find_non_zero(&image, &mut non_zero_pixels).unwrap();
        for non_zero_point in non_zero_pixels.to_vec() {
            let (row, col) = (non_zero_point.y, non_zero_point.x);
            if row == 0 || col == 0 {
                continue;
            }
            let l_corn = Point::new(col - &window_size, row - &window_size);
            let r_corn = Point::new(col + &window_size, row + &window_size);
            let rect = Rect::from_points(l_corn, r_corn);
            let roi_mat = Mat::roi(image, rect);
            if roi_mat.is_err() {
                continue;
            }

            let roi_matrix = &roi_mat.unwrap();
            let non_zero_count = count_non_zero(roi_matrix).unwrap();
            let colored_scalar = match non_zero_count {
                val if val < neighbours => Scalar::from(BLACK_COLOR),
                val if &val >= &color_borders.channel_4 => Scalar::from(RED_COLOR),
                val if &val >= &color_borders.channel_3 => Scalar::from(YELLOW_COLOR),
                val if &val >= &color_borders.channel_2 => Scalar::from(CYAN_COLOR),
                val if &val >= &color_borders.channel_1 => Scalar::from(GREEN_COLOR),
                _ => Scalar::from(BLACK_COLOR),
            };

            result_frame
                .at_2d_mut::<Scalar>(row, col)
                .unwrap()
                .copy_from_slice(colored_scalar.as_slice());
        }

        Ok(result_frame)
    }
}
