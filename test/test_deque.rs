extern crate cvlcore;

#[cfg(test)]
mod test {
    use cvlcore::core::deque::CvlMatDeque;
    use cvlcore::core::mat::CvlMat;
    use opencv::imgcodecs::imread;
    use std::path::Path;

    #[test]
    fn test_cvlmat_deque_i32() {
        let mut cvl_deque: CvlMatDeque<i32> = CvlMatDeque::default();
        for int_value in 1..10 {
            let _ = &cvl_deque.push(int_value);
        }

        assert_eq!(5, *cvl_deque.get(0).unwrap());
        assert_eq!(6, *cvl_deque.get(1).unwrap());
        assert_eq!(7, *cvl_deque.get(2).unwrap());
        assert_eq!(8, *cvl_deque.get(3).unwrap());
        assert_eq!(9, *cvl_deque.get(4).unwrap());
        assert_eq!(cvl_deque.max_size(), 5);
        assert_eq!(cvl_deque.length(), 5);
    }

    #[test]
    fn test_cvlmat_deque_mat() {
        let cvl_deque = CvlMatDeque::from(load_resource_frames());
        assert_eq!(cvl_deque.length(), 15);
    }

    fn load_resource_frames() -> Vec<CvlMat> {
        let flags = 3;
        Path::new("test/resources/")
            .read_dir()
            .unwrap()
            .map(Result::unwrap)
            .filter(|f| f.file_name().to_str().unwrap().contains("test_file_"))
            .map(|f| f.path().to_str().unwrap().to_string())
            .map(|f| imread(f.as_str(), flags).unwrap())
            .map(CvlMat::new)
            .collect()
    }
}
