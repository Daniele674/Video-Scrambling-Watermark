#QUESTO FA LA CONVERSIONE MP4 - MPG (NON SERVE A NIENTE)

import os
import ffmpeg

def compress_to_mpeg1(input_file, output_file, video_bitrate='800k', audio_bitrate='192k'):
    try:
        (
            ffmpeg.input(input_file)
            .output(
                'temp.mpg',
                codec='mpeg2video',
                **{'b:v': video_bitrate, 'c:a': 'mp2', 'b:a': audio_bitrate}
            )
            .run(overwrite_output=True)
        )

        # Converti il file temporaneo MPEG2 in MPEG1
        (
            ffmpeg.input('temp.mpg')
            .output(
                output_file,
                codec='mpeg1video',
                **{'b:v': video_bitrate, 'c:a': 'mp2', 'b:a': audio_bitrate}
            )
            .run(overwrite_output=True)
        )

        # Elimina il file temporaneo MPEG2
        os.remove('temp.mpg')

        print(f"Compressione completata. Il file compresso è salvato in: {output_file}")
    except ffmpeg.Error as e:
        print(f"Errore durante la compressione: {e.stderr}")

if __name__ == "__main__":
    input_video = '/home/vboxuser/Desktop/Compressione/Progetto_Compressione(Python)/Progetto_Compressione/Human_safari.mp4'
    output_video = '/home/vboxuser/Desktop/Compressione/Progetto_Compressione(Python)/Progetto_Compressione/Human_safari_compresso.mpg'
    
    compress_to_mpeg1(input_video, output_video)
