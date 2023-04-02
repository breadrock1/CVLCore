pub mod cvldetector {
    use ndarray::prelude::*;

    use opencv::core::{cart_to_polar, count_non_zero, find_non_zero, Mat, MatTrait, MatTraitConst, MatTraitConstManual, Point, Scalar, Vector, BORDER_DEFAULT, CV_32F, CV_64FC4, CV_8UC3, Rect};
    use opencv::imgproc;
    use opencv::types::VectorOfMat;

    /// The red color pixel value used for marking magnitude and vibration Mat object.
    pub const RED_COLOR: (f64, f64, f64, f64) = (0.0, 0.0, 255.0, 0.0);

    /// The cyan color pixel value used for marking magnitude and vibration Mat object.
    pub const CYAN_COLOR: (f64, f64, f64, f64) = (255.0, 255.0, 0.0, 0.0);

    /// The green color pixel value used for marking magnitude and vibration Mat object.
    pub const GREEN_COLOR: (f64, f64, f64, f64) = (0.0, 255.0, 0.0, 0.0);

    /// The yellow color pixel value used for marking magnitude and vibration Mat object.
    pub const YELLOW_COLOR: (f64, f64, f64, f64) = (0.0, 255.0, 255.0, 0.0);

    /// The black color pixel value used for marking magnitude and vibration Mat object.
    pub const BLACK_COLOR: (f64, f64, f64, f64) = (0.0, 0.0, 255.0, 0.0);

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
        let (low, high) = (1.0 - sigma + median, 1.0 + sigma + median);

        let mut canny_frame = Mat::default();
        imgproc::canny(&frame, &mut canny_frame, low, high, size, is_l2).unwrap();

        Ok(canny_frame)
    }

    /// This mehtod returns arithmetic mean (average) of all elements in array.
    /// In mathematics and statistics, the arithmetic mean / arithmetic average is the sum of a
    /// collection of numbers divided by the count of numbers in the collection. The collection
    /// is often a set of results from an experiment, an observational study, or a survey. The
    /// term "arithmetic mean" is preferred in some mathematics and statistics contexts because
    /// it helps distinguish it from other types of means, such as geometric and harmonic.
    ///
    /// ## Parameters:
    /// * frame: (&Mat) the passed video stream frame to transform.
    pub fn calculate_mat_median(frame: &Mat) -> f64 {
        let rows = frame.rows() as usize;
        let cols = frame.cols() as usize;
        let mut buffer = vec![0.0; rows * cols];
        let data = frame.data_typed::<u8>().expect("");

        for r in 0..rows {
            for c in 0..cols {
                let index = r * cols + c;
                buffer[index] = data[r * cols + c] as f64;
            }
        }

        Array::from_shape_vec((rows, cols), buffer)
            .unwrap()
            .mean()
            .unwrap()
    }


    pub fn calculate_vibrating_image(
        image: &Mat,
        _neighbours_bound: i32,
    ) -> Result<Mat, opencv::Error> {
        let mut non_zero_pixels = Vector::<Point>::new();
        find_non_zero(&image, &mut non_zero_pixels).unwrap();

        let (rows, cols) = (image.rows(), image.cols());
        let data = image.data_typed::<u8>().expect("Failed to get types!");
        let mut result_mat = Mat::new_rows_cols_with_default(rows, cols, CV_64FC4, Scalar::default()).unwrap();

        for non_z_point in non_zero_pixels.to_vec() {
            let index = non_z_point.y * cols + non_z_point.x;
            let mut colored_scalar = match data[index as usize] {
                val if val > 200 => Scalar::from(RED_COLOR),
                val if val > 150 => Scalar::from(CYAN_COLOR),
                val if val > 100 => Scalar::from(GREEN_COLOR),
                val if val > 50 => Scalar::from(YELLOW_COLOR),
                _val => Scalar::from(BLACK_COLOR),
            };

            result_mat
                .at_2d_mut::<Scalar>(non_z_point.y, non_z_point.x)
                .unwrap()
                .copy_from_slice(colored_scalar.as_slice());
        }

        Ok(result_mat)
    }

    ///   0 1 2
    /// 0 0 0 0
    /// 1 0 x 0
    /// 2 0 0 0
    pub fn compute_vibrating_pixels(image: &Mat, neighbours: i32) -> Result<Mat, opencv::Error> {
        let mut non_zero_pixels = Vector::<Point>::new();
        find_non_zero(&image, &mut non_zero_pixels).unwrap();

        let mut result_frame = image.clone();
        for non_z_point in non_zero_pixels.to_vec() {
            let (row, col) = (non_z_point.y, non_z_point.x);
            let l_corn = Point::new(row - 1, col - 1);
            let r_corn = Point::new(row + 1, col + 1);
            let rect = Rect::from_points(l_corn, r_corn);
            let roi_mat = Mat::roi(&image, rect);
            if roi_mat.is_err() {
                continue;
            }

            let mut non_zero_count = match count_non_zero(&roi_mat.unwrap()) {
                Ok(count) if count < neighbours => 0,
                Ok(count) => count - 1,
                _ => 0
            };

            let mut kek = result_frame.at_2d_mut::<u8>(row, col).unwrap();
            kek = &mut (non_zero_count as u8);
        }

        Ok(result_frame)
    }






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
            image_map_shape.0 as i32,
            image_map_shape.1 as i32,
            CV_8UC3,
            Scalar::new(0.0, 0.0, 0.0, 0.0),
        )
            .unwrap();

        // let (r, g, y, c) = (
        //     Scalar::new(0.0, 0.0, 255.0, 0.0),
        //     Scalar::new(0.0, 255.0, 0.0, 0.0),
        //     Scalar::new(255.0, 0.0, 0.0, 0.0),
        //     Scalar::new(0.0, 0.0, 0.0, 0.0),
        // );

        let mut vibration_mask = image.clone();
        let mut nonzero_mask = VectorOfMat::default();
        find_non_zero(image, &mut nonzero_mask).unwrap();
        Ok(vibration_mask)
    }

    pub fn gen_abs_frame(_frames: Vec<&Mat>) -> opencv::Result<Mat> {
        todo!()
    }
}
