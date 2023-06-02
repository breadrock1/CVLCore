extern crate cvldetector;

use crate::cvldetector::cvldetector::*;

use opencv::prelude::VideoCaptureTrait;
use opencv::{core, highgui, videoio};

fn main() {
    let mut frames_to_abs: Vec<core::Mat> = Vec::new();
    highgui::named_window("Simple using", highgui::WINDOW_FULLSCREEN).unwrap();
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY).unwrap();

    loop {
        let mut frame = core::Mat::default();
        cam.read(&mut frame).unwrap();

        let gray_frame = gen_grayscale_frame(&frame).unwrap();
        let canny_frame = gen_canny_frame_by_sigma(&gray_frame, 3, 0.05, true).unwrap();
        frames_to_abs.push(canny_frame.clone());
        if frames_to_abs.len() < 5 {
            continue;
        }

        frames_to_abs.remove(0);
        let abs_image = gen_abs_frame(&frames_to_abs).unwrap();
        let computed_image = compute_vibrating_pixels(&abs_image, 5).unwrap();
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
