extern crate cvldetector;

use crate::cvldetector::cvldetector::*;

use opencv::prelude::VideoCaptureTrait;
use opencv::{core, highgui, prelude, videoio};
use opencv::core::Mat;

fn main() {
    let mut maxval = 150.0;
    let mut thresh1 = 150.0;
    let mut thresh2 = 100.0;

    let mut frames_to_abs: Vec<Mat> = Vec::new();

    highgui::named_window("Simple using", highgui::WINDOW_FULLSCREEN).unwrap();
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY).unwrap();

    loop {
        let mut frame = prelude::Mat::default();
        cam.read(&mut frame).unwrap();

        let gray_frame = gen_grayscale_frame(&frame).unwrap();
        // let thresh_frame = gen_threshold_frame(&gray_frame, thresh1, maxval).unwrap();
        // let canny_frame = gen_canny_frame(&thresh_frame, thresh1, thresh2, 5, true).unwrap();
        // let mut canny_frame = gen_canny_frame_by_sigma(&thresh_frame, 5, 0.33, true).unwrap();
        let canny_frame = gen_canny_frame_by_sigma(&gray_frame, 3, 0.01, true).unwrap();
        frames_to_abs.push(canny_frame.clone());
        // let distrib_frame = gen_distribution_frame(&canny_frame, thresh1, maxval).unwrap();
        // let vibro_frame = compute_vibrating_pixels(&canny_frame, 4).unwrap();
        // let distrib_frame = calculate_vibrating_image(&vibro_frame, 4).unwrap();

        if (frames_to_abs.len() < 5) {
            continue;
        }

        frames_to_abs.remove(0);
        let abs_image = gen_abs_frame(&frames_to_abs).unwrap();
        let distrib_frame = calculate_vibrating_image(&abs_image, 4).unwrap();

        highgui::imshow("window", &distrib_frame).unwrap();
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
