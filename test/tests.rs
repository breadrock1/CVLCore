extern crate cvlcore;

use crate::cvlcore::cvlcore::*;

use opencv::core::{Mat, Vector};
use opencv::imgcodecs::imread;

#[test]
fn test_calculate_mat_median() {
    let image_path = std::path::Path::new("resources");
    let test: Vec<_> = image_path.read_dir()
        .unwrap()
        .into_iter()
        .map(|val| val.unwrap().file_name())
        .collect();

    let file_name = test.first().unwrap().to_str().unwrap();
    // let full_name = format!("{}.jpeg", file_name.clone()).as_str();
    println!("{}", &file_name);
    let flags = 3;
    let image: Mat = imread(file_name, flags).unwrap();
    // calculate_mat_median()
}


