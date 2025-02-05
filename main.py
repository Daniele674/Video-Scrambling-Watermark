from scramblery import scramblery as sc
import time

type = 'signFlip'  # 'signFlip' or 'permutation'
seed = 1234
password_img = 1
password_wm = 1

scramble_settings = {
    'type': type,  # 'signFlip' or 'permutation'
    'num_to_flip': 63,  # Number of coefficients to flip for signFlip type from 0 to 64
    'seed': seed,
    'password_img': password_img,
    'password_wm': password_wm,
    'write': False  # Should always be False for video processing
}

descramble_settings = {
    'wm_shape': 222,
    'seed': seed,
    'password_img': password_img,
    'password_wm': password_wm,
    'write': False  # Should always be False for video processing
}

video_path = 'testSet/videoBreve.mp4'
scrambled_video_path = "output_video.mp4"
descrambled_video_path = "output_video_descrambled.mp4"

start_time = time.time()

sc.scramblevideo(video_path, scrambled_video_path, scramble_settings)

end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time)

sc.descramblevideo(scrambled_video_path, descrambled_video_path, descramble_settings)