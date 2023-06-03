# CVLDetector

![GitHub version](https://img.shields.io/badge/version-v1.0.1-green?style=plastic&labelColor=dark)

[![Building Project](https://github.com/breadrock1/CVLDetector/actions/workflows/rust.yml/badge.svg?branch=master)](https://github.com/breadrock1/CVLDetector/actions/workflows/build-project-action.yml)

[![Creating Release](https://github.com/breadrock1/CVLDetector/actions/workflows/release.yml/badge.svg?branch=master&event=create)](https://github.com/breadrock1/CVLDetector/actions/workflows/create-release-action.yml)

### Program Description

The CVLDetector project is an opportunity for continuous analysis of a video stream with the functionality of calibrating parameters to generate a vibro-image, including calculating statistics of changes in vibrating pixels. Based on this information, the final client software provides the ability to detect macro/micro movement in the shooting area of both the video file and the broadcast.

![cvrimg.png](resources%2Fcvrimg.png)

Detailed adjustment of video stream processing parameters provides an opportunity to calibrate the parameters of vibro-image generation, information about which is later used to calculate statistics.

For example, the user has the ability to change the size of the video frame set, which is used to calculate the vibro-pixels between these frames. Also, the user has the ability to set the absolute value of the neighbors, which is used to filter noise.

This library allows you to:
- Broadcast video stream from ip/RTSP/webcam, as well as from video files;
- Transmission of alarms via API, calculated during the calculation of vibration pixel statistics.
