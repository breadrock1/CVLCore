#![feature(test)]
extern crate cvlcore;
extern crate test;

#[cfg(test)]
mod benchmark {
    use cvlcore::core::bounds::*;
    use cvlcore::core::mat::*;
    use cvlcore::core::statistic::*;
    use cvlcore::*;
    use opencv::core::Mat;
    use opencv::imgcodecs::imread;
    use std::path::Path;
    use std::rc::Rc;
    use test::Bencher;

    #[bench]
    fn bench_grayscale(b: &mut Bencher) {
        let frames = load_resource_frames();
        let mat = frames.first().unwrap();
        b.iter(|| {
            let cvlmat = CvlMat::new(mat.clone());
            let _ = gen_grayscale_frame(&cvlmat).unwrap();
        });
    }

    #[bench]
    fn bench_threshold(b: &mut Bencher) {
        let frames = load_resource_frames();
        let mat = frames.first().unwrap();
        b.iter(|| {
            let cvlmat = CvlMat::new(mat.clone());
            let gray = gen_grayscale_frame(&cvlmat).unwrap();
            let _ = gen_threshold_frame(&gray, 100.0, 255.0).unwrap();
        });
    }

    #[bench]
    fn bench_canny(b: &mut Bencher) {
        let frames = load_resource_frames();
        let mat = frames.first().unwrap();
        b.iter(|| {
            let cvlmat = CvlMat::new(mat.clone());
            let gray = gen_grayscale_frame(&cvlmat).unwrap();
            let _ = gen_canny_frame(&gray, 100.0, 255.0, 3, true).unwrap();
        });
    }

    #[bench]
    fn bench_canny_by_sigma(b: &mut Bencher) {
        let frames = load_resource_frames();
        let mat = frames.first().unwrap();
        b.iter(|| {
            let cvlmat = CvlMat::new(mat.clone());
            let gray = gen_grayscale_frame(&cvlmat).unwrap();
            let _ = gen_canny_frame_by_sigma(&gray, 3, 0.05, true).unwrap();
        });
    }

    #[bench]
    fn bench_compute_median(b: &mut Bencher) {
        let frames = load_resource_frames();
        let mat = frames.first().unwrap();
        b.iter(|| {
            let cvlmat = CvlMat::new(mat.clone());
            let gray = gen_grayscale_frame(&cvlmat).unwrap();
            let _ = calculate_mat_median(&gray).unwrap_or(0f64);
        });
    }

    #[bench]
    fn bench_compute_median_only(b: &mut Bencher) {
        let frames = load_resource_frames();
        let mat = frames.first().unwrap();
        let cvlmat = CvlMat::new(mat.clone());
        let gray = gen_grayscale_frame(&cvlmat).unwrap();
        b.iter(|| {
            let _ = calculate_mat_median(&gray).unwrap_or(0f64);
        });
    }

    #[bench]
    fn bench_diff_frame(b: &mut Bencher) {
        let frames = load_resource_frames()
            .into_iter()
            .map(CvlMat::new)
            .map(|m| gen_grayscale_frame(&m).unwrap())
            .map(|m| gen_canny_frame_by_sigma(&m, 3, 0.05, true).unwrap())
            .map(Rc::new)
            .collect::<Vec<Rc<CvlMat>>>();

        b.iter(|| {
            let _ = gen_abs_frame(&frames).unwrap();
        });
    }

    #[bench]
    fn bench_diff_reduce(b: &mut Bencher) {
        let frames = load_resource_frames()
            .into_iter()
            .map(CvlMat::new)
            .map(|m| gen_grayscale_frame(&m).unwrap())
            .map(|m| gen_canny_frame_by_sigma(&m, 3, 0.05, true).unwrap())
            .map(Rc::new)
            .collect::<Vec<Rc<CvlMat>>>();

        b.iter(|| {
            let _ = gen_abs_frame_reduce(&frames).unwrap();
        });
    }

    #[bench]
    fn bench_compute_vibrating(b: &mut Bencher) {
        let frames = load_resource_frames();
        let color_bounds = ColorBounds::default();

        b.iter(|| {
            let cvl_frames = frames
                .clone()
                .into_iter()
                .map(CvlMat::new)
                .map(|m| gen_grayscale_frame(&m).unwrap())
                .map(|m| gen_canny_frame_by_sigma(&m, 3, 0.05, true).unwrap())
                .map(Rc::new)
                .collect::<Vec<Rc<CvlMat>>>();

            let abs_frame = gen_abs_frame_reduce(&cvl_frames).unwrap();
            let _ = compute_vibration(&abs_frame, 8, 2, &color_bounds).unwrap();
        });
    }

    #[bench]
    fn bench_compute_vibrating_only(b: &mut Bencher) {
        let frames = load_resource_frames()
            .into_iter()
            .map(CvlMat::new)
            .map(|m| gen_grayscale_frame(&m).unwrap())
            .map(|m| gen_canny_frame_by_sigma(&m, 3, 0.05, true).unwrap())
            .map(Rc::new)
            .collect::<Vec<Rc<CvlMat>>>();

        let abs_frame = gen_abs_frame_reduce(&frames).unwrap();
        let color_bounds = ColorBounds::default();
        b.iter(|| {
            let _ = compute_vibration(&abs_frame, 8, 2, &color_bounds).unwrap();
        });
    }

    #[bench]
    fn bench_compute_statistic(b: &mut Bencher) {
        let stat_1 = Statistic::new(354, 256, 129, 80);
        let stat_2 = Statistic::new(879, 567, 280, 143);
        let stat_3 = Statistic::new(657, 452, 456, 111);
        let stat_4 = Statistic::new(200, 190, 160, 78);
        let stat_5 = Statistic::new(123, 100, 98, 65);

        b.iter(|| {
            let stat_list = vec![&stat_1, &stat_2, &stat_3, &stat_4, &stat_5];
            let _ = compute_statistic(stat_list, 10.0);
        });
    }

    fn load_resource_frames() -> Vec<Mat> {
        let flags = 3;
        Path::new("test/resources/")
            .read_dir()
            .unwrap()
            .map(Result::unwrap)
            .filter(|f| f.file_name().to_str().unwrap().contains("test_file_"))
            .map(|f| f.path().to_str().unwrap().to_string())
            .map(|f| imread(f.as_str(), flags).unwrap())
            .collect()
    }
}
