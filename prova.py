from scramblery import scramblery as sc
import pickle

scramble_settings = {
    'splits': 25,
    'type': 'pixel',  # 'pixel', 'stack', 'fourier'
    'bg': True,
    'seamless': False,  # se settato a True, non si vede il bordo della ROI
    # 'scramble_ratio': 0.5, # Only for 'fourier' scrambling
    'seed': 1,
    'write': False  # Should always be False for video processing
}

video_path = 'testSet/girl.mp4'

pixel_swaps = sc.scramblevideo(video_path, "output_video.mp4", scramble_settings)

# with open("test", "wb") as fp:  # Pickling
#     pickle.dump(pixel_swaps, fp)
#
# with open("test", "rb") as fp:  # Unpickling
#     b = pickle.load(fp)

sc.descrambleVideo("output_video.mp4", "output_video_descrambled.mp4", pixel_swaps)
