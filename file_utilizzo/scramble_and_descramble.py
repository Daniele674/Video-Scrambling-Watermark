import argparse
from scramblery import scramblery as sc

# Default values
default_key = b'supersegreto1236'
default_scramble_type = 'permutation'
default_num_to_flip = 64
default_input_path = 'testSet/videoBreve.mp4'
default_scrambled_path = 'output_video.mp4'
default_descrambled_path = 'output_video_descrambled.mp4'

# Set up argument parser
parser = argparse.ArgumentParser(description='Scramble and descramble a video file.')
parser.add_argument('-k', '--key', type=str, help='Encryption key (16, 24, or 32 bytes)')
parser.add_argument('-t', '--scramble_type', type=str, choices=['signFlip', 'permutation'], help='Type of scrambling')
parser.add_argument('-n', '--num_to_flip', type=int, help='Number of coefficients to modify (only for signFlip)')
parser.add_argument('-i', '--input_path', type=str, help='Path to the input video file')
parser.add_argument('-s', '--scrambled_path', type=str, help='Path to the scrambled video file')
parser.add_argument('-d', '--descrambled_path', type=str, help='Path to the descrambled video file')
args = parser.parse_args()

# Get parameters from command line arguments if provided
key = args.key.encode() if args.key else default_key
scramble_type = args.scramble_type if args.scramble_type else default_scramble_type
num_to_flip = args.num_to_flip if args.num_to_flip else default_num_to_flip
input_path = args.input_path if args.input_path else default_input_path
scrambled_path = args.scrambled_path if args.scrambled_path else default_scrambled_path
descrambled_path = args.descrambled_path if args.descrambled_path else default_descrambled_path

# Scramble settings
scramble_settings = {
    'scramble_type': scramble_type,
    'num_to_flip': num_to_flip if scramble_type == 'signFlip' else 0,
}

print("Inizio scrambling del video...")
sc.scramblevideo(input_path, scrambled_path, scramble_settings, key)
print("Processo di scrambling completato!")

print("Inizio descrambling del video...")
sc.descramblevideo(scrambled_path, descrambled_path, key)
print("Processo di descrambling completato!")
