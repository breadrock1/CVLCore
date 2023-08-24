extern crate cvlcore;

#[cfg(test)]
mod main_test {
    use cvlcore::api::chain::*;
    use cvlcore::core::bounds::*;
    use cvlcore::core::cvl::*;
    use cvlcore::core::mat::*;
    use opencv::core::{Mat, MatTraitConst};
    use opencv::imgcodecs::imread;
    use std::path::Path;
    use std::rc::Rc;

    #[test]
    pub fn test_grayscale() {
        let frames = load_resource_frames();
        let mat = frames.first().unwrap();
        let cvlmat = CvlMat::new(mat.clone());
        let gray = gen_grayscale_frame(&cvlmat).unwrap();
        assert_eq!(gray.frame().channels(), 1);
        assert_eq!(gray.frame().dims(), 2);
    }

    #[test]
    fn test_threshold() {
        let frames = load_resource_frames();
        let mat = frames.first().unwrap();
        let cvlmat = CvlMat::new(mat.clone());
        let gray = gen_grayscale_frame(&cvlmat).unwrap();
        let thresh = gen_threshold_frame(&gray, 100.0, 255.0).unwrap();
        assert_eq!(thresh.frame().channels(), 1);
        assert_eq!(thresh.frame().dims(), 2);
    }

    #[test]
    fn test_canny() {
        let frames = load_resource_frames();
        let mat = frames.first().unwrap();
        let cvlmat = CvlMat::new(mat.clone());
        let gray = gen_grayscale_frame(&cvlmat).unwrap();
        let canny = gen_canny_frame(&gray, 100.0, 255.0, 3, true).unwrap();
        assert_eq!(canny.frame().channels(), 1);
        assert_eq!(canny.frame().dims(), 2);
    }

    #[test]
    fn test_canny_by_sigma() {
        let frames = load_resource_frames();
        let mat = frames.first().unwrap();
        let cvlmat = CvlMat::new(mat.clone());
        let gray = gen_grayscale_frame(&cvlmat).unwrap();
        let canny = gen_canny_frame_by_sigma(&gray, 3, 0.05, true).unwrap();
        assert_eq!(canny.frame().channels(), 1);
        assert_eq!(canny.frame().dims(), 2);
    }

    #[test]
    fn test_distribution() {
        let frames = load_resource_frames();
        let mat = frames.first().unwrap();
        let cvlmat = CvlMat::new(mat.clone());
        let gray = gen_grayscale_frame(&cvlmat).unwrap();
        let _distrib = gen_distribution_frame(&gray, 100.0, 255.0).unwrap();
        // assert_eq!(distrib.frame().channels(), 1);
        // assert_eq!(distrib.frame().dims(), 2);
    }

    #[test]
    fn test_compute_median() {
        let frames = load_resource_frames();
        let mat = frames.first().unwrap();
        let cvlmat = CvlMat::new(mat.clone());
        let gray = gen_grayscale_frame(&cvlmat).unwrap();
        let median = calculate_mat_median(&gray).unwrap_or(0f64);
        assert_eq!(median, 45.84036024305556);
    }

    #[test]
    fn test_diff_frame() {
        let frames = load_resource_frames()
            .into_iter()
            .map(CvlMat::new)
            .map(|m| gen_grayscale_frame(&m).unwrap())
            .map(|m| gen_canny_frame_by_sigma(&m, 3, 0.05, true).unwrap())
            .map(Rc::new)
            .collect::<Vec<Rc<CvlMat>>>();

        let abs_frame = gen_abs_frame(&frames).unwrap();
        assert_eq!(abs_frame.frame().channels(), 1);
        assert_eq!(abs_frame.frame().dims(), 2);
    }

    #[test]
    fn test_diff_frame_reduce() {
        let frames = load_resource_frames()
            .into_iter()
            .map(CvlMat::new)
            .map(|m| gen_grayscale_frame(&m).unwrap())
            .map(|m| gen_canny_frame_by_sigma(&m, 3, 0.05, true).unwrap())
            .map(Rc::new)
            .collect::<Vec<Rc<CvlMat>>>();

        let abs_frame = gen_abs_frame_reduce(&frames).unwrap();
        assert_eq!(abs_frame.frame().channels(), 1);
        assert_eq!(abs_frame.frame().dims(), 2);
    }

    #[test]
    fn test_diff_frame_accuracy() {
        let frames = load_resource_frames()
            .into_iter()
            .map(CvlMat::new)
            .map(|m| gen_grayscale_frame(&m).unwrap())
            .map(|m| gen_canny_frame_by_sigma(&m, 3, 0.05, true).unwrap())
            .map(Rc::new)
            .collect::<Vec<Rc<CvlMat>>>();

        let abs_frame = gen_abs_frame(&frames).unwrap();
        let abs_frame_reduce = gen_abs_frame_reduce(&frames).unwrap();

        let abs_mean = calculate_mat_median(&abs_frame).unwrap();
        let abs_mean_reduce = calculate_mat_median(&abs_frame_reduce).unwrap();

        println!("ABS: {}\nABS (Reduce): {}\n", abs_mean, abs_mean_reduce);
        assert_ne!(abs_mean, abs_mean_reduce);
    }

    #[test]
    fn test_compute_vibrating() {
        let frames = load_resource_frames()
            .into_iter()
            .map(CvlMat::new)
            .map(|m| gen_grayscale_frame(&m).unwrap())
            .map(|m| gen_canny_frame_by_sigma(&m, 3, 0.05, true).unwrap())
            .map(Rc::new)
            .collect::<Vec<Rc<CvlMat>>>();

        let abs_frame = gen_abs_frame_reduce(&frames).unwrap();
        let color_bounds = ColorBounds::default();
        let result = compute_vibration(&abs_frame, 8, 2, &color_bounds).unwrap();
        assert_eq!(result.frame().channels(), 4);
        assert_eq!(result.frame().dims(), 2);
    }

    #[test]
    fn test_chain_processing() {
        let frames = load_resource_frames();
        let mat = frames.first().unwrap();
        let cvlmat = CvlMat::new(mat.clone());

        let abs_frames = frames
            .into_iter()
            .map(CvlMat::new)
            .map(|m| gen_grayscale_frame(&m).unwrap())
            .map(|m| gen_canny_frame_by_sigma(&m, 3, 0.05, true).unwrap())
            .map(Rc::new)
            .collect::<Vec<Rc<CvlMat>>>();

        let mut own_chain = ChainProcessing::default();
        own_chain.set_frames(&abs_frames);
        let precessing_result = own_chain
            .run_chain(cvlmat)
            .grayscale()
            .canny()
            .append_frame()
            .reduce_abs()
            .vibrating();

        let chain_result = precessing_result.get_result();
        let result = chain_result.unwrap();
        assert_eq!(result.frame().channels(), 4);
        assert_eq!(result.frame().dims(), 2);
    }

    fn load_resource_frames() -> Vec<Mat> {
        let flags = 3;
        Path::new("resources/")
            .read_dir()
            .unwrap()
            .map(Result::unwrap)
            .filter(|f| f.file_name().to_str().unwrap().contains("test_frame_"))
            .map(|f| f.path().to_str().unwrap().to_string())
            .map(|f| imread(f.as_str(), flags).unwrap())
            .collect()
    }
}
