import argparse
from scramblery import scramblery as sc

# Default values
default_key = b'supersegreto1236'
default_input_path = './output_video.mp4'
default_output_path = './output_video_descrambled.mp4'

# Set up argument parser
parser = argparse.ArgumentParser(description='Descramble a video file.')
parser.add_argument('-k', '--key', type=str, help='Encryption key (16, 24, or 32 bytes)')
parser.add_argument('-i', '--input_path', type=str, help='Path to the input video file')
parser.add_argument('-o', '--output_path', type=str, help='Path to the output video file')
args = parser.parse_args()

# Get parameters from command line arguments if provided
key = args.key.encode() if args.key else default_key
input_path = args.input_path if args.input_path else default_input_path
output_path = args.output_path if args.output_path else default_output_path

print("Inizio descrambling del video...")
sc.descramblevideo(input_path, output_path, key)
print("Processo di descrambling completato!")
