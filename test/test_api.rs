extern crate cvlcore;

#[cfg(test)]
mod main_test {
    use cvlcore::api::chain::ChainProcessing;
    use cvlcore::core::mat::CvlMat;
    use cvlcore::core::statistic::*;
    use cvlcore::*;
    use opencv::core::{Mat, MatTraitConst};
    use opencv::imgcodecs::imread;
    use std::path::Path;
    use std::rc::Rc;

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

    #[test]
    fn test_chain_statistic() {
        let frames = load_resource_frames();
        let all_frames = frames.into_iter().map(CvlMat::new).collect::<Vec<CvlMat>>();

        let mut dispertion = Dispersion::default();
        let mut own_chain = ChainProcessing::default();
        for cvlmat in all_frames {
            let precessing_result = own_chain
                .run_chain(cvlmat)
                .grayscale()
                .canny()
                .append_frame()
                .reduce_abs()
                .vibrating()
                .statistic();

            match precessing_result.get_dispersion() {
                None => continue,
                Some(result) => dispertion = result.clone(),
            }
        }

        assert_eq!(dispertion.ch1, 177.78374);
        assert_eq!(dispertion.ch2, 78.44896);
        assert_eq!(dispertion.ch3, 198.52461);
        assert_eq!(dispertion.ch4, 141.05609);
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
