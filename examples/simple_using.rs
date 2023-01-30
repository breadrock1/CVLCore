extern crate cvldetector;
use crate::cvldetector::cvldetector::*;

use opencv::prelude::VideoCaptureTrait;
use opencv::{core, highgui, prelude, videoio};

fn main() {
    let mut maxval = 150.0;
    let mut thresh1 = 150.0;
    let mut thresh2 = 100.0;

    let mut frames_to_abs: Vec<Box<core::Mat>> = Vec::new();

    highgui::named_window("Simple using", highgui::WINDOW_FULLSCREEN).unwrap();
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY).unwrap();

    loop {
        let mut frame = prelude::Mat::default();
        cam.read(&mut frame).unwrap();

        let gray_frame = gen_grayscale_frame(&frame).unwrap();
        let thresh_frame = gen_threshold_frame(&gray_frame, thresh1, maxval).unwrap();
        let canny_frame = gen_canny_frame(&thresh_frame, thresh1, thresh2, 5, true).unwrap();

        highgui::imshow("window", &canny_frame).unwrap();
        match highgui::wait_key(1) {
            Ok(key) => match key {
                113 => {
                    exit_app();
                    break;
                }
                109 => maxval += 10.0,
                110 => maxval -= 10.0,
                116 => thresh1 += 10.0,
                117 => thresh1 -= 10.0,
                121 => thresh2 += 10.0,
                12 => thresh2 -= 10.0,
                _ => (),
            },
            _ => continue,
        }
    }
}

fn exit_app() {
    highgui::destroy_window("window").unwrap()
}
