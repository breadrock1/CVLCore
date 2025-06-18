extern crate cvlcore;

use cvlcore::api::capture::*;
use cvlcore::api::chain::*;
use cvlcore::errors::*;
use cvlcore::ui::window::*;

fn main() -> CaptureResult {
    let window_name = "CVLDetector Demo";
    let window = MainWindow::new(window_name);
    window.create_window();

    let mut vcap = CvlCapture::default();
    vcap.open_stream("0", StreamSource::WebCamera)?;
    processing_stream(&mut vcap, &window);

    window.close_window();
    Ok(())
}

fn processing_stream(vcap: &mut CvlCapture, window: &MainWindow) {
    let mut own_chain = ChainProcessing::default();
    while let Ok(frame) = vcap.read_frame() {
        let processing = own_chain
            .run_chain(frame)
            .grayscale()
            .canny()
            .append_frame()
            .reduce_abs()
            .vibrating();

        let cvl_mat = match processing.get_result() {
            Ok(mat) => mat,
            Err(err) => {
                println!("failed while processing frame: {err:#?}");
                continue;
            }
        };

        window.show_frame(&cvl_mat);
        match window.wait_event() {
            WindowSignals::KeepProcessing => {}
            WindowSignals::CLoseApplication => break,
        }
    }
}
