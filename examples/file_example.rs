extern crate cvlcore;

use crate::cvlcore::cvlcore::*;

use opencv::core;
use opencv::core::Vector;
use opencv::highgui::{destroy_window, imshow, named_window, wait_key};
use opencv::highgui::{WINDOW_AUTOSIZE, WINDOW_GUI_NORMAL};
use opencv::imgcodecs::imwrite;
use opencv::prelude::VideoCaptureTrait;
use opencv::videoio::{VideoCapture, CAP_ANY};
use opencv::imgcodecs::ImwriteFlags;

fn main() {
    let neighbours = 8;
    let window_size = 2;
    let frames_set_size = 5;
    let is_reduced_abs = true;
    let bounds = ColorBounds::default();
    let mut frames_to_abs: Vec<core::Mat> = Vec::new();

    let file_path_arg = std::env::args()
        .last()
        .expect("Video file path has not been passed!");

    let main_win_name = "CVLDetector Demo";
    named_window(main_win_name, WINDOW_AUTOSIZE | WINDOW_GUI_NORMAL).unwrap();
    let mut cam = VideoCapture::from_file(file_path_arg.as_str(), CAP_ANY).unwrap();

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

        let vibro_image = compute_vibration(&abs_image, neighbours, window_size, &bounds).unwrap();
        imshow(main_win_name, &vibro_image).unwrap();
        if let Ok(key) = wait_key(10) {
            if key == 113 {
                exit_app(main_win_name);
                break;
            }
        }
    }
}

fn exit_app(window_name: &str) {
    destroy_window(window_name).unwrap()
}
