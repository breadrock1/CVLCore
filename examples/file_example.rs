extern crate cvldetector;

use crate::cvldetector::cvldetector::*;

use opencv::prelude::VideoCaptureTrait;
use opencv::{core, highgui, videoio};

fn main() {
    let neighbours = 8;
    let window_size = 2;
    let frames_set_size = 5;
    let is_reduced_abs = true;
    let color_borders = ColorBounds::default();

    let mut frames_to_abs: Vec<core::Mat> = Vec::new();
    let file_path_arg = std::env::args()
        .last()
        .expect("Video file path has not been passed!");

    let mut cam =
        videoio::VideoCapture::from_file(file_path_arg.as_str(), videoio::CAP_ANY).unwrap();

    loop {
        let mut frame = core::Mat::default();
        cam.read(&mut frame).unwrap();

        let gray_frame = gen_grayscale_frame(&frame).unwrap();
        let canny_frame = gen_canny_frame_by_sigma(&gray_frame, 3, 0.05, true).unwrap();
        frames_to_abs.push(canny_frame.clone());
        if frames_to_abs.len() < frames_set_size {
            continue;
        }

        frames_to_abs.remove(0);
        let abs_image = match is_reduced_abs {
            true => gen_abs_frame_reduce(&frames_to_abs),
            false => gen_abs_frame(&frames_to_abs),
        }
            .unwrap();

        let computed_image =
            compute_vibrating_pixels(&abs_image, neighbours, window_size, &color_borders).unwrap();
        highgui::imshow("window", &computed_image).unwrap();
        match highgui::wait_key(1) {
            Ok(key) => match key {
                113 => {
                    exit_app();
                    break;
                }
                _ => (),
            },
            _ => continue,
        }
    }
}

fn exit_app() {
    highgui::destroy_window("window").unwrap()
}
