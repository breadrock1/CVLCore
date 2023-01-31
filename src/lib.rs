pub mod cvldetector {
    use opencv::core;
    use opencv::core::MatTraitConstManual;
    use opencv::imgproc;
    use std::ops::Deref;

    /// The red color pixel value used for marking magnitude and vibration Mat object.
    pub const RED_COLOR: (u8, u8, u8) = (255, 0, 0);

    /// The cyan color pixel value used for marking magnitude and vibration Mat object.
    pub const CYAN_COLOR: (u8, u8, u8) = (255, 255, 0);

    /// The green color pixel value used for marking magnitude and vibration Mat object.
    pub const GREEN_COLOR: (u8, u8, u8) = (0, 255, 0);

    /// The yellow color pixel value used for marking magnitude and vibration Mat object.
    pub const YELLOW_COLOR: (u8, u8, u8) = (0, 255, 255);

    /// Transformations within RGB space like adding/removing the alpha channel, reversing the
    /// channel order, conversion to/from 16-bit RGB color (R5:G6:B5 or R5:G5:B5), as well as
    /// conversion to/from grayscale.
    ///
    /// ## Parameters:
    /// * frame: (&core::Mat) the passed video stream frame to transform.
    #[inline]
    pub fn gen_grayscale_frame(frame: &core::Mat) -> opencv::Result<core::Mat> {
        let mut gray_frame = core::Mat::default();
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
    /// * frame: (&core::Mat) the passed video stream frame to transform.
    /// * tresh: (f64) the black bound-value to swap pixel value to 1.
    /// * maxval: (f64) the white bound-value to swap pixel value to 0.
    #[inline]
    pub fn gen_threshold_frame(
        frame: &core::Mat,
        thresh: f64,
        maxval: f64,
    ) -> opencv::Result<core::Mat> {
        let mut grayscale_frame: core::Mat = core::Mat::default();
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
    /// * frame: (&core::Mat) the passed video stream frame to transform.
    /// * low: (f64) the first threshold for the hysteresis procedure.
    /// * high: (f64) the second threshold for the hysteresis procedure.
    /// * size: (i32) the aperture size of Sobel operator to generate Canny view.
    /// * is_l2: (bool) the specifies the equation for finding gradient magnitude.
    #[inline]
    pub fn gen_canny_frame(
        frame: &core::Mat,
        low: f64,
        high: f64,
        size: i32,
        is_l2: bool,
    ) -> opencv::Result<core::Mat> {
        let mut canny_frame = core::Mat::default();
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
    /// * frame: (&core::Mat) the passed video stream frame to transform.
    /// * size: (i32) the aperture size of Sobel operator to generate Canny view.
    /// * sigma: (f64) the value to vary the percentage thresholds that are determined based on simple statistics.
    /// * is_l2: (bool) the specifies the equation for finding gradient magnitude.
    #[inline]
    pub fn gen_canny_frame_by_sigma(
        frame: &core::Mat,
        size: i32,
        sigma: f64,
        is_l2: bool,
    ) -> opencv::Result<core::Mat> {
        let median = 0.0;
        let low = 1.0 - sigma + median;
        let high = 1.0 + sigma + median;

        let mut canny_frame = core::Mat::default();
        imgproc::canny(&frame, &mut canny_frame, low, high, size, is_l2).unwrap();

        Ok(canny_frame)
    }

    pub fn gen_abs_frame(frames: Vec<&core::Mat>) -> opencv::Result<core::Mat> {
        todo!()
    }
}
