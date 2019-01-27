import os
import moviepy.video.io.ffmpeg_writer as ffmpeg_writer
import atexit
import numpy as np


class VideoWriter(object):
    """ A class for FFMPEG-based video writing.

        A class to write videos using ffmpeg. ffmpeg will write in a large
        choice of formats.

        Parameters
        -----------

        filename
          Any filename like 'video.mp4' etc. but if you want to avoid
          complications it is recommended to use the generic extension
          '.avi' for all your videos.

        size
          Size (width,height) of the output video in pixels.

        fps
          Frames per second in the output video file.

        codec
          FFMPEG codec. It seems that in terms of quality the hierarchy is
          'rawvideo' = 'png' > 'mpeg4' > 'libx264'
          'png' manages the same lossless quality as 'rawvideo' but yields
          smaller files. Type ``ffmpeg -codecs`` in a terminal to get a list
          of accepted codecs.

          Note for default 'libx264': by default the pixel format yuv420p
          is used. If the video dimensions are not both even (e.g. 720x405)
          another pixel format is used, and this can cause problem in some
          video readers.

        audiofile
          Optional: The name of an audio file that will be incorporated
          to the video.

        preset
          Sets the time that FFMPEG will take to compress the video. The slower,
          the better the compression rate. Possibilities are: ultrafast,superfast,
          veryfast, faster, fast, medium (default), slow, slower, veryslow,
          placebo.

        bitrate
          Only relevant for codecs which accept a bitrate. "5000k" offers
          nice results in general.

        withmask
          Boolean. Set to ``True`` if there is a mask in the video to be
          encoded.

        """

    def __init__(self, filename, size=(1920, 1080), fps=30, codec="libx264", audiofile=None,
                 preset="medium", bitrate=None, withmask=False,
                 logfile=None, threads=None, ffmpeg_params=None):

        self._filename = filename
        self._size = list(size)
        self._fps = fps
        self._codec = codec
        self._audiofile = audiofile
        self._preset = preset
        self._bitrate = bitrate
        self._withmask = withmask
        self._logfile = logfile
        self._threads = threads
        self._ffmpeg_params = ffmpeg_params

        # set the writer to None to start
        self._writer = None

        # keep track of frame count
        self.frame_count = 0

        # register to finalize the video on exit
        atexit.register(self._will_exit)

    def write(self, frame_or_frames):
        if isinstance(frame_or_frames, list):
            for f in frame_or_frames:
                self.write(f)
        elif isinstance(frame_or_frames, np.ndarray):
            if len(list(frame_or_frames.shape)) == 4:
                # batch of images, split up the batches
                frames = [frame_or_frames[x, :, :, :] for x in range(0, frame_or_frames.shape[0])]
                for f in frames:
                    self._write_single_frame(f)
            elif len(list(frame_or_frames.shape)) == 3:
                self._write_single_frame(frame_or_frames)
            else:
                raise ValueError('Invalid frame shape: {} . Expected a shape with 3 dimensions'.format(
                    frame_or_frames.shape))
        else:
            raise TypeError('Unknown frame type. Expected an ndarray or list. Received a {}'.format(
                type(frame_or_frames)))

    def close(self):
        if self._writer is not None:
            self._writer.close()

    def _write_single_frame(self, img):
        # make sure we are initialized
        self._init()
        clipped_frame = np.clip(img, 0, 255).astype(np.uint8)
        self._writer.write_frame(clipped_frame)
        self.frame_count += 1

    def _init(self):
        if self._writer is None:
            # make parent dirs if they do not exist
            abs_path = os.path.abspath(self._filename)
            dir_path = os.path.dirname(abs_path)
            os.makedirs(dir_path, exist_ok=True)

            self._writer = ffmpeg_writer.FFMPEG_VideoWriter(
                filename=abs_path,
                size=self._size,
                fps=self._fps,
                codec=self._codec,
                audiofile=self._audiofile,
                preset=self._preset,
                bitrate=self._bitrate,
                withmask=self._withmask,
                logfile=self._logfile,
                threads=self._threads,
                ffmpeg_params=self._ffmpeg_params
            )

    def _will_exit(self):
        self.close()




